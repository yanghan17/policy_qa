# Policy Q&A Bot

A Retrieval-Augmented Generation (RAG) system that answers questions about
insurance policy documents with grounded, clause-level citations.
Runs entirely locally — no cloud API keys required.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put your PDFs in docs/
ls docs/
#  QM8698-1124_QBE_Contents_Insurance_PDS.pdf
#  QM9264-0723_Small_Business_Insurance.pdf
#  qbe_home_policy_wording_mil.pdf

# 4. Build the index  (run once; re-run after changing docs/)
python main.py --build

# 5. Ask a question  (MockLLM by default — shows pipeline structure)
python main.py --ask "Is wear and tear covered?"

# 6. Get real answers with Ollama
ollama pull mistral
python main.py --ask "Is wear and tear covered?" --ollama

# 7. Interactive session
python main.py --interactive --ollama

# 8. Run the 10-question test suite
python test_qa.py                  # MockLLM
python test_qa.py --ollama         # Ollama (requires mistral)
```

---

## Project Structure

```
policy_qa/
├── main.py              CLI entry point
├── test_qa.py           10-question test suite
├── requirements.txt
├── README.md
├── DESIGN_NOTES.md
│
└── src/
    ├── ingestor.py      PDF loading + section-aware chunking
    ├── embeddings.py    sentence-transformers (all-MiniLM-L6-v2)
    ├── vector_store.py  ChromaDB index (HNSW, cosine similarity)
    ├── llm.py           Pluggable LLM backends
    └── qa_engine.py     RAG pipeline: retrieve → filter → generate
```

---

## Embeddings

**Model: `all-MiniLM-L6-v2`** via `sentence-transformers`.

- 384-dimensional dense vectors, L2-normalised
- Fine-tuned on 1B+ sentence pairs for semantic similarity
- 22 MB, runs on CPU, ~5ms per sentence
- Auto-downloads from HuggingFace on first use

Captures meaning, not just keywords — "compensation for death" matches
"accidental death benefit" even with no word overlap.

---

## Vector Store: ChromaDB

- Runs in-process, no server needed
- HNSW index for fast approximate nearest-neighbour search
- Auto-persists to `chroma_store/`
- Stores metadata alongside embeddings (doc_name, page, heading_path)

---

## LLM Backends


| Flag              | Class                 | Requires                                             |
| ----------------- | --------------------- | ---------------------------------------------------- |
| *(default)*       | `MockLLM`             | Nothing — shows pipeline structure                   |
| `--ollama`        | `OllamaLLM`           | [Ollama](https://ollama.com) + `ollama pull mistral` |
| `--hf-model NAME` | `HuggingFaceLocalLLM` | `pip install transformers torch`                     |
| `--lmstudio`      | `OpenAICompatibleLLM` | [LM Studio](https://lmstudio.ai)                     |


No Anthropic API or any cloud service required.

Recommended Ollama models:

```bash
ollama pull mistral        # 4 GB, good quality
ollama pull llama3         # 4.7 GB, best quality
ollama pull phi3:mini      # 2.3 GB, fast on CPU
```

---

## CLI Reference

```bash
python main.py --build                          # ingest + index
python main.py --ask "..."                      # single question, MockLLM
python main.py --ask "..." --ollama             # Ollama (default: mistral)
python main.py --ask "..." --ollama --model llama3
python main.py --ask "..." --hf-model microsoft/phi-2
python main.py --ask "..." --lmstudio
python main.py --interactive --ollama           # chat session
python main.py --ask "..." --verbose            # show retrieved chunks
```

---

## Test results

See **[RESULTS.md](RESULTS.md)** for a summary of the 10-question evaluation suite
(Ollama/mistral run, pass/fail notes, known limitations).

## Running Tests

```bash
python test_qa.py                    # MockLLM
python test_qa.py --ollama           # Ollama
python test_qa.py --build-first      # rebuild index from docs/ then test
python test_qa.py --verbose          # show retrieved chunks per question
```

Re-run `--build` after adding or removing PDFs in `docs/` so the index
does not retain stale documents (e.g. old PDS files).

If you see `[OllamaLLM ERROR] timed out`, Mistral on CPU is hitting the
HTTP wait limit on large prompts. Use `--ollama-timeout 900`, a smaller
model (`--model phi3:mini`), or fewer chunks (`--top-k 4` in `main.py`).

10 test cases: 5 in-domain, 3 near-miss, 2 out-of-scope.

---

## Example Output

```
======================================================================
Q: Is wear and tear covered under the home contents policy?
======================================================================

Wear and tear is NOT covered.

The policy explicitly excludes loss or damage caused by Wear, Tear.
[Doc: xxx_home_policy_wording_mil.pdf, §Section 1 > Exclusions > 1(c), p.9]

Exception: if wear and tear directly causes another covered event such
as fire or glass breakage, the resulting damage from that event IS covered.
[Doc: xxx_home_policy_wording_mil.pdf, §Section 1 > Exclusions > 1(c), p.9]

SOURCES:
  • xxx_home_policy_wording_mil.pdf  §Section 1 > Exclusions  p.9  (similarity: 0.74)
  • QM8698-1124_xxx_Contents_Insurance_PDS.pdf  §Exclusions  p.34  (similarity: 0.61)
```

