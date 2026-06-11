# Design Notes — Policy Q&A Bot

---

## 1. Embedding model

**`all-MiniLM-L6-v2`** via `sentence-transformers`.

This is a 6-layer MiniLM model fine-tuned on 1B+ sentence pairs for
semantic similarity. It produces **384-dimensional L2-normalised vectors**
and runs in ~5ms per sentence on CPU. First use downloads ~22 MB from
HuggingFace; subsequent runs use the local cache.

**Why this model?**

Dense embeddings capture *meaning*, not just keywords. A query like
"compensation for death" matches "accidental death benefit" even with
zero word overlap — which matters because users paraphrase policy language.
A keyword-only approach (TF-IDF) would score this poorly.

`all-MiniLM-L6-v2` is the standard choice for retrieval tasks at this
size: strong benchmark performance, 22 MB, no GPU needed.

To swap models, change `DEFAULT_MODEL` in `embeddings.py`:
```python
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"   # stronger, still 130 MB
```

---

## 2. Vector store: ChromaDB

ChromaDB runs **fully in-process** — no Docker, no separate server, just
`pip install chromadb`. It uses an **HNSW index** for approximate
nearest-neighbour search in sub-linear time, scales to millions of chunks,
and persists to disk automatically.

ChromaDB also stores metadata alongside embeddings (`doc_name`, `page`,
`heading_path`, etc.), which enables future metadata filters like:
```python
collection.query(..., where={"doc_name": "qbe_home_policy.pdf"})
```

The collection is configured with `hnsw:space=cosine`. Because our
embeddings are L2-normalised (unit length), dot product equals cosine
similarity, so ChromaDB's distance output is `1 - similarity`.

---

## 3. LLM backends

No cloud API is used. All backends are local:

| Class | How | Notes |
|-------|-----|-------|
| `MockLLM` | Returns a placeholder | Default; no model needed |
| `OllamaLLM` | HTTP to `localhost:11434` | Recommended; `ollama pull mistral` |
| `HuggingFaceLocalLLM` | `transformers` pipeline | Any HF model |
| `OpenAICompatibleLLM` | HTTP to local server | LM Studio, llama.cpp, vLLM |

All implement `BaseLLM.generate(prompt) → str`. Adding a new backend
(e.g. Anthropic) is one class with one method.

---

## 4. Chunking strategy

Insurance policies have a strict hierarchy that naive fixed-size chunking
would break — a chunk could contain the tail of "What We Cover" and the
start of "Exclusions", making accurate citation impossible.

**Our approach:**
1. `pdfplumber` extracts text page-by-page (better layout preservation
   than `pypdf`, important for heading detection).
2. Lines are classified as headings via three heuristics: all-caps short
   lines, lines matching an insurance keyword regex (`section`, `exclusion`,
   `definition`, …), and numbered section patterns (`Section 2 – Buildings`).
3. A **heading breadcrumb stack** (max depth 3) is maintained:
   `["Section 1 - Home Contents", "Exclusions", "1(c) Wear and Tear"]`
4. When a new heading is detected, the current buffer flushes *before*
   starting the new section — **chunks never straddle a heading boundary**.
5. Target chunk size: ~600 tokens (approximated as `len(text) // 4`).
6. 80-token overlap between consecutive chunks so clauses near a boundary
   appear fully in at least one chunk.
7. Every chunk carries `{chunk_id, doc_name, page, section, heading_path,
   clause_ref, text}`.

---

## 5. Prompt design

```
You are a precise insurance policy assistant.

STRICT RULES:
1. Answer ONLY from the CONTEXT provided.
2. Every factual claim → [Doc: ..., §..., p....]
3. Insufficient context → "I cannot find a definitive answer..."
4. Conflicts between documents → state them explicitly.
5. Bullet points for lists.
```

**Why explicit rules?** LLMs are trained on large insurance corpora and
will fill gaps with general knowledge — dangerous when users rely on
policy specifics. Rule 1 is the most critical.

**Context block format:**
```
[CHUNK 2]  Doc: qbe_home_policy.pdf  |  Section: Section 1 > Exclusions
           |  Clause: 1(c)  |  Page: 9
<chunk text>
```

The metadata header is inside the prompt so the LLM can form exact
citations without guessing file names or page numbers.

---

## 6. Retrieval improvements (`retrieval.py`)

Coverage questions (*"Is X covered?"*) used to match **What We cover**
headings instead of **Exclusions**. The engine now:

1. **Expands queries** — e.g. adds `Section 1 Exclusions wear and tear`.
2. **Merges** top results from each query (deduped by chunk id).
3. **Boosts** Exclusion chunks (+0.12) and demotes misleading **What We cover**
   chunks (−0.06) for coverage questions.
4. **Prefers** `qbe_home_policy_wording_mil.pdf` over PDS/Small Business PDFs
   unless the question names another product.
5. **Duration questions** about temporary accommodation get an extra query
   targeting Event 21 / Event 7 and demote unrelated Important Information clauses.

After changing files in `docs/`, rebuild: `python main.py --build`.

---

## 7. Anti-hallucination: three layers

| Layer | Location | Mechanism |
|-------|----------|-----------|
| Relevance gate | `qa_engine.py` | If no chunk scores ≥ 0.05 cosine similarity, return "cannot find" without calling the LLM at all |
| LLM self-report | system prompt Rule 3 | Model instructed to say "I cannot find" — detected in output |
| Conflict surfacing | system prompt Rule 4 | Model instructed to flag contradictions between documents |

The relevance gate is the most robust layer. It prevents the LLM from
receiving a low-quality context and confabulating an answer.

---

## 8. Test set design

| ID | Why chosen |
|----|-----------|
| IN-1 (wear & tear) | Most common policyholder misconception; clear exclusion |
| IN-2 (money-back) | Specific time-limited right; tests exact number extraction |
| IN-3 (domestic animals) | Common exclusion; tests negative answer retrieval |
| IN-4 (personal accident) | Two compensation amounts by age group |
| IN-5 (temp accommodation) | Duration limit with specific section reference |
| NM-1 (flood + garden) | Two clauses contradict each other — the open-air sub-clause overrides general flood cover. Most dangerous user trap in these docs. |
| NM-2 (waiting period) | Tests that the model does not confabulate a waiting period that does not exist |
| NM-3 (laptop theft) | Multi-condition answer: Section 4 covers it globally, but tenant/limit caveats apply |
| OOS-1 (motor vehicle) | Vehicles are explicitly excluded — related text exists but "no coverage" is the answer |
| OOS-2 (overseas travel) | "Medical" appears in Section 5 (burglary injuries at home) — model must not conflate this with travel cover |
