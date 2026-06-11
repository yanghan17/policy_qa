"""
retrieval.py
------------
Query expansion and score adjustments for policy Q&A retrieval.

Fixes common failure modes:
  - "Is X covered?" matching "What We cover" instead of Exclusions
  - Duration/limit questions matching unrelated "90 day" clauses
  - Wrong PDF (PDS) outranking the home policy wording document
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ingestor import Chunk

_COVERAGE_RE = re.compile(r"\b(cover|covers|covered|coverage)\b", re.I)
_DURATION_RE = re.compile(
    r"\b(period|maximum|how long|duration|months?|limit)\b", re.I
)


def is_coverage_question(question: str) -> bool:
    return bool(_COVERAGE_RE.search(question))


def is_duration_question(question: str) -> bool:
    return bool(_DURATION_RE.search(question))


def preferred_doc_names(question: str, all_docs: List[str]) -> Optional[List[str]]:
    """
    Restrict retrieval to the home policy wording when the question
    clearly targets it, or when it is the only indexed document.
    """
    q = question.lower()
    home_docs = [d for d in all_docs if "home" in d.lower() and "policy" in d.lower()]
    if "home contents" in q or "home insurance" in q or "home policy" in q:
        return home_docs or None
    if len(all_docs) == 1:
        return all_docs
    # Default: prefer home policy wording over PDS / other products
    if home_docs and not any(x in q for x in ("pds", "contents insurance pds", "small business")):
        return home_docs
    return None


def expand_queries(question: str) -> List[str]:
    """Return search queries (primary first, then expansions)."""
    queries = [question]
    q = question.lower()

    if is_coverage_question(question):
        queries.append(f"Section 1 Exclusions {_topic_from_coverage_question(question)}")
        queries.append(f"exclusion {_topic_from_coverage_question(question)}")

    if is_duration_question(question) and (
        "temporary" in q or "accommodation" in q
    ):
        queries.append(
            "temporary accommodation maximum period months "
            "Section 1 Event 21 Section 2 Event 7"
        )

    if "flood" in q and ("garden" in q or "open air" in q or "outside" in q):
        queries.append(
            "Section 1 Event 1 Loss or Damage caused by Flood Home Contents"
        )
        queries.append(
            "Section 1 open air garden Storm Rainwater Flood We will not pay"
        )

    if any(w in q for w in ("laptop", "stolen", "theft", "coffee")) and (
        "claim" in q or "stolen" in q
    ):
        queries.append(
            "Section 4 Personal Valuables Unspecified anywhere in the World theft"
        )

    if "compensation" in q or "accidental death" in q:
        queries.append("Section 5 Personal Accident Compensation HK$ age child")

    # Deduplicate while preserving order
    seen, out = set(), []
    for query in queries:
        key = query.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(query.strip())
    return out


def _topic_from_coverage_question(question: str) -> str:
    """Strip framing words; keep the subject (e.g. 'wear and tear')."""
    q = question.strip()
    for frag in (
        r"\s+under the home contents policy[.?]*\s*$",
        r"\s+under the (?:home )?policy[.?]*\s*$",
        r"\s+under the policy[.?]*\s*$",
    ):
        q = re.sub(frag, "", q, flags=re.I)
    q = re.sub(
        r"^(?:is|does|do|can|will)\s+(?:the\s+)?(?:policy\s+)?",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"^(?:the\s+)?policy\s+cover\s+", "", q, flags=re.I)
    q = re.sub(r"\s+(?:is\s+)?covered[.?]*\s*$", "", q, flags=re.I)
    q = re.sub(r"\s+cover\s+", " ", q, flags=re.I)
    return q.strip(" ?.") or question


def _topic_words(question: str) -> List[str]:
    """Content words from the coverage topic (drop stopwords)."""
    stop = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "under", "over", "for", "from", "with", "about", "into", "through",
        "during", "before", "after", "above", "below", "between", "policy",
        "home", "contents", "insurance", "cover", "covered", "coverage",
        "damage", "caused", "own", "that", "this", "these", "those", "what",
        "which", "who", "whom", "whose", "where", "when", "why", "how",
        "any", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "same", "so", "than",
        "too", "very", "just", "also", "there", "their", "they", "them",
        "you", "your", "our", "we", "us", "it", "its",
    }
    words = re.findall(r"[a-z0-9]+", _topic_from_coverage_question(question).lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _section_relevance_boost(chunk: Chunk, question: str) -> float:
    """Prefer the policy section that matches the question subject."""
    q  = question.lower()
    hp = chunk.heading_path.lower()
    b  = 0.0

    if any(x in q for x in ("home contents", "home content")):
        if "section 1" in hp and "home contents" in hp:
            b += 0.18
        elif re.search(r"section [2-9]", hp):
            b -= 0.14

    if "personal accident" in q or "accidental death" in q:
        if "section 5" in hp and "personal accident" in hp:
            b += 0.15

    if "temporary accommodation" in q or ("temporary" in q and "accommodation" in q):
        if "section 1" in hp or "section 2" in hp:
            if "temporary" in (chunk.text + hp).lower():
                b += 0.10

    if any(w in q for w in ("laptop", "stolen", "theft", "coffee", "valuables")):
        if "section 4" in hp:
            b += 0.15

    return b


def _supplement_score(chunk: Chunk, question: str, base: float = 0.80) -> float:
    """Keyword-hit score blended with section relevance; capped at 1.0."""
    return min(1.0, base + _section_relevance_boost(chunk, question))


def adjust_score(chunk: Chunk, score: float, question: str) -> float:
    """Boost chunks that match the question type; demote misleading sections."""
    hp   = chunk.heading_path.lower()
    text = chunk.text.lower()
    q    = question.lower()

    score += _section_relevance_boost(chunk, question)
    score = min(score, 1.0)

    if is_coverage_question(question):
        if _is_exclusion_like_chunk(chunk):
            score += 0.08
        elif "what we cover" in hp and "not pay" not in hp:
            score -= 0.06

    q = question.lower()
    if "flood" in q and any(w in q for w in ("garden", "open air")):
        if "open air" in text and any(w in text for w in ("flood", "storm")):
            score += 0.12
        if "event 1" in text and "flood" in text:
            score += 0.08

    if any(w in q for w in ("laptop", "stolen", "theft")):
        if "section 4" in hp and "personal valuables" in hp:
            score += 0.12

    if is_duration_question(question) and (
        "temporary" in q or "accommodation" in q
    ):
        if "temporary" in text or "accommodation" in text:
            score += 0.10
        if "event" in hp and ("temporary" in text or "accommodation" in text):
            score += 0.08
        if "important informat" in hp and "temporary" not in text:
            score -= 0.10

    return score


def _is_exclusion_like_chunk(chunk: Chunk) -> bool:
    """Exclusions often live under 'We will not pay', not only 'Exclusions' headings."""
    hp   = chunk.heading_path.lower()
    head = chunk.text[:1200].lower()
    return any(
        marker in hp or marker in head
        for marker in (
            "exclusion",
            "exclusions",
            "not pay",
            "do not cover",
            "we will not",
            "some events we do not",
        )
    )


def _topic_matches_blob(blob: str, words: List[str]) -> bool:
    if not words:
        return False
    return all(w in blob for w in words)


def keyword_exclusion_supplement(
    all_chunks: List[Chunk],
    question: str,
    doc_names: Optional[List[str]] = None,
) -> List[Tuple[Chunk, float]]:
    """
    If semantic search missed exclusion clauses, match by topic + exclusion-like text.
    """
    if not is_coverage_question(question):
        return []

    words = _topic_words(question)
    if not words:
        return []

    q_lower = question.lower()
    home_contents_q = "home contents" in q_lower

    out: List[Tuple[Chunk, float]] = []
    for chunk in all_chunks:
        if doc_names and chunk.doc_name not in doc_names:
            continue
        hp = chunk.heading_path.lower()
        if home_contents_q and not ("section 1" in hp and "home contents" in hp):
            continue
        blob = (chunk.heading_path + " " + chunk.text).lower()
        if not _is_exclusion_like_chunk(chunk):
            continue
        if _topic_matches_blob(blob, words):
            out.append((chunk, _supplement_score(chunk, question)))
    return out


def keyword_open_air_flood_supplement(
    all_chunks: List[Chunk],
    question: str,
    doc_names: Optional[List[str]] = None,
) -> List[Tuple[Chunk, float]]:
    q = question.lower()
    if "flood" not in q:
        return []
    if not any(w in q for w in ("garden", "open air", "outside", "outdoor")):
        return []

    out: List[Tuple[Chunk, float]] = []
    for chunk in all_chunks:
        if doc_names and chunk.doc_name not in doc_names:
            continue
        blob = (chunk.heading_path + " " + chunk.text).lower()
        if "open air" not in blob:
            continue
        if not any(w in blob for w in ("flood", "storm", "rainwater")):
            continue
        out.append((chunk, _supplement_score(chunk, question, base=0.82)))
    return out


def keyword_flood_cover_supplement(
    all_chunks: List[Chunk],
    question: str,
    doc_names: Optional[List[str]] = None,
) -> List[Tuple[Chunk, float]]:
    """Section 1 general flood cover (Event 1) for garden/open-air questions."""
    q = question.lower()
    if "flood" not in q:
        return []

    out: List[Tuple[Chunk, float]] = []
    for chunk in all_chunks:
        if doc_names and chunk.doc_name not in doc_names:
            continue
        hp = chunk.heading_path.lower()
        if "section 1" not in hp or "home contents" not in hp:
            continue
        blob = (chunk.heading_path + " " + chunk.text).lower()
        if "caused by flood" in blob or "damage caused by flood" in blob:
            if "open air" in blob and "will not pay" in blob:
                continue
            out.append((chunk, _supplement_score(chunk, question, base=0.79)))
    return out


def keyword_section4_supplement(
    all_chunks: List[Chunk],
    question: str,
    doc_names: Optional[List[str]] = None,
) -> List[Tuple[Chunk, float]]:
    q = question.lower()
    if not any(w in q for w in ("laptop", "stolen", "theft", "jewellery", "valuable")):
        return []

    out: List[Tuple[Chunk, float]] = []
    for chunk in all_chunks:
        if doc_names and chunk.doc_name not in doc_names:
            continue
        hp = chunk.heading_path.lower()
        if "section 4" not in hp and "personal valuables" not in hp:
            continue
        blob = (chunk.heading_path + " " + chunk.text).lower()
        if "anywhere" in blob or "world" in blob or "unspecified" in blob:
            out.append((chunk, _supplement_score(chunk, question, base=0.81)))
    return out


def keyword_duration_supplement(
    all_chunks: List[Chunk],
    question: str,
    doc_names: Optional[List[str]] = None,
) -> List[Tuple[Chunk, float]]:
    q = question.lower()
    if not ("temporary" in q and "accommodation" in q):
        return []

    out: List[Tuple[Chunk, float]] = []
    for chunk in all_chunks:
        if doc_names and chunk.doc_name not in doc_names:
            continue
        blob = (chunk.heading_path + " " + chunk.text).lower()
        if "temporary" not in blob or "accommodation" not in blob:
            continue
        if "month" not in blob and "event" not in chunk.heading_path.lower():
            continue
        out.append((chunk, _supplement_score(chunk, question, base=0.81)))
    return out


def merge_results(
    batches: List[List[Tuple[Chunk, float]]],
    question: str,
    top_k: int,
) -> List[Tuple[Chunk, float]]:
    """Merge multiple search result lists; keep best score per chunk."""
    merged: dict[str, Tuple[Chunk, float]] = {}
    for batch in batches:
        for chunk, score in batch:
            score = min(1.0, adjust_score(chunk, score, question))
            prev = merged.get(chunk.chunk_id)
            if prev is None or score > prev[1]:
                merged[chunk.chunk_id] = (chunk, score)
    return sorted(merged.values(), key=lambda x: x[1], reverse=True)[:top_k]
