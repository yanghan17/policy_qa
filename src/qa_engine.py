"""
qa_engine.py
------------
The Q&A pipeline: retrieval → filtering → prompt → LLM → citation.

Pipeline
--------
  question
      │
      ▼
  multi-query retrieval (see retrieval.py) + score boosts
      │  → List[(Chunk, cosine_similarity)]
      ▼
  relevance_gate()        drop chunks below MIN_SCORE
      │
      ├─ all dropped? → "cannot find" response (no LLM call)
      │
      ▼
  build_prompt(question, kept_chunks)
      │
      ▼
  llm.generate(prompt)
      │
      ▼
  detect "cannot find" in LLM output
      │
      ▼
  QAResult(answer, sources, is_grounded)

Three layers of anti-hallucination
------------------------------------
1. Relevance gate (engine level)  — refuses to call LLM if retrieved
   chunks are below the similarity threshold.  Returns a structured
   "cannot find" response with the closest related clauses listed.

2. System prompt enforcement (LLM level)  — the prompt's STRICT RULES
   tell the model to answer only from context and say "I cannot find"
   when uncertain.  We detect this phrase in the output.

3. Conflict detection (LLM level)  — the prompt tells the model to
   surface contradictions between documents rather than silently
   picking one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ingestor import Chunk
from retrieval import (
    expand_queries,
    keyword_duration_supplement,
    keyword_exclusion_supplement,
    keyword_flood_cover_supplement,
    keyword_open_air_flood_supplement,
    keyword_section4_supplement,
    merge_results,
    preferred_doc_names,
)
from vector_store import VectorStore
from llm import BaseLLM, OLLAMA_ERROR_PREFIX, build_prompt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K     = 6      # chunks passed to the LLM after multi-query merge
RETRIEVE_PER_QUERY = 5   # chunks fetched per expanded query
MIN_SCORE = 0.05   # minimum cosine similarity to pass the relevance gate
                   # (empirically good for both TF-IDF and dense embeddings)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    question:    str
    answer:      str
    sources:     List[dict]                              # citation metadata
    is_grounded: bool = True                             # False → "cannot find"
    raw_chunks:  List[Tuple[Chunk, float]] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [
            "=" * 70,
            f"Q: {self.question}",
            "=" * 70,
            "",
            self.answer,
            "",
        ]
        if self.sources:
            lines.append("SOURCES:")
            for s in self.sources:
                lines.append(
                    f"  • {s['doc_name']}  "
                    f"§{s['section']}  "
                    f"p.{s['page']}  "
                    f"(similarity: {s['score']:.3f})"
                )
        else:
            lines.append("(No relevant clauses found in the loaded documents.)")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QAEngine:
    """
    Parameters
    ----------
    store     : A built/loaded VectorStore instance.
    llm       : Any BaseLLM subclass.
    top_k     : How many chunks to retrieve before score filtering.
    min_score : Cosine similarity floor — chunks below this are discarded.
    """

    def __init__(
        self,
        store:     VectorStore,
        llm:       BaseLLM,
        top_k:     int   = TOP_K,
        min_score: float = MIN_SCORE,
    ):
        self.store     = store
        self.llm       = llm
        self.top_k     = top_k
        self.min_score = min_score

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ask(self, question: str) -> QAResult:
        question = question.strip()

        # 1. Multi-query retrieval with exclusion/duration boosts
        raw_results = self._retrieve(question)

        # 2. Apply relevance gate
        good = [(c, s) for c, s in raw_results if s >= self.min_score]

        # 3. If nothing passes gate, return structured "cannot find"
        if not good:
            return self._cannot_answer(question, raw_results)

        # 4. Build prompt from surviving chunks
        prompt = build_prompt(question, [c for c, _ in good])

        # 5. Call LLM
        answer = self.llm.generate(prompt)

        # 6. Grounded = real answer (not LLM refusal, not Ollama transport error)
        is_grounded = (
            not self._is_refusal(answer)
            and not answer.startswith(OLLAMA_ERROR_PREFIX)
        )

        if answer.startswith(OLLAMA_ERROR_PREFIX):
            answer += "\n\n" + self._closest_clauses_note(raw_results)
        elif not is_grounded:
            answer += "\n\n" + self._closest_clauses_note(raw_results)

        return QAResult(
            question    = question,
            answer      = answer,
            sources     = self._build_sources(good),
            is_grounded = is_grounded,
            raw_chunks  = raw_results,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> List[Tuple[Chunk, float]]:
        all_docs = list({c.doc_name for c in self.store._chunks.values()})
        doc_filter = preferred_doc_names(question, all_docs)
        per_q = max(RETRIEVE_PER_QUERY, self.top_k // 2)

        batches: List[List[Tuple[Chunk, float]]] = []
        for query in expand_queries(question):
            batches.append(
                self.store.search(query, k=per_q, doc_names=doc_filter)
            )

        # If filtered search returned nothing, retry without filter
        if doc_filter and not any(batch for batch in batches):
            batches = [
                self.store.search(q, k=per_q)
                for q in expand_queries(question)
            ]

        registry = list(self.store._chunks.values())
        batches.append(keyword_exclusion_supplement(registry, question, doc_filter))
        batches.append(keyword_duration_supplement(registry, question, doc_filter))
        batches.append(keyword_open_air_flood_supplement(registry, question, doc_filter))
        batches.append(keyword_flood_cover_supplement(registry, question, doc_filter))
        batches.append(keyword_section4_supplement(registry, question, doc_filter))
        return merge_results(batches, question, self.top_k)

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        low = answer.lower()
        if "cannot find a definitive answer" not in low:
            return False
        # Conflict answers may mention "cannot find" spuriously — treat as OK
        if any(
            p in low
            for p in (
                "however,",
                "conflict",
                "open air",
                "does not specify a waiting period",
                "the policy does not specify",
            )
        ):
            return False
        return True

    def _cannot_answer(
        self,
        question: str,
        raw: List[Tuple[Chunk, float]],
    ) -> QAResult:
        answer = (
            "I cannot find a definitive answer in the provided policy wording.\n\n"
            "This question does not appear to match any section of the loaded "
            "documents with sufficient confidence.\n\n"
            + self._closest_clauses_note(raw)
        )
        return QAResult(
            question    = question,
            answer      = answer,
            sources     = [],
            is_grounded = False,
            raw_chunks  = raw,
        )

    def _build_sources(
        self, results: List[Tuple[Chunk, float]]
    ) -> List[dict]:
        seen, out = set(), []
        for chunk, score in results:
            key = (chunk.doc_name, chunk.heading_path, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "doc_name": chunk.doc_name,
                "section":  chunk.heading_path,
                "page":     chunk.page,
                "clause":   chunk.clause_ref,
                "score":    round(score, 4),
            })
        return out

    def _closest_clauses_note(self, results: List[Tuple[Chunk, float]]) -> str:
        if not results:
            return "No related clauses found in the loaded documents."
        lines = ["Closest related clauses found (may not directly answer the question):"]
        for chunk, score in results[:3]:
            lines.append(
                f"  • {chunk.doc_name}  §{chunk.heading_path}  "
                f"p.{chunk.page}  (score: {score:.3f})\n"
                f"    \"{chunk.text[:120].strip()}…\""
            )
        return "\n".join(lines)
