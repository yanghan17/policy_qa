# Submission checklist

## What to send

**Format:** One **ZIP file** (or GitHub repo link if they allow it), e.g. `policy_qa_yourname.zip`.

**Root folder inside zip:** `policy_qa/` (so reviewers unzip and see `README.md` immediately).

## Include

| Path | Purpose |
|------|---------|
| `README.md` | Setup, usage, architecture overview — **read first** |
| `DESIGN_NOTES.md` | Chunking, prompts, anti-hallucination — **read second** |
| `RESULTS.md` | Test-run evidence (optional but recommended) |
| `requirements.txt` | Dependencies |
| `main.py`, `test_qa.py`, `test_retrieval.py` | Entry points |
| `src/*.py` | Implementation |
| `docs/*.pdf` | All 3 policy PDFs |
| `chroma_store/` | Pre-built index (optional — speeds reviewer run; adds size) |

## Do not include

| Path | Why |
|------|-----|
| `.venv/` | Large, machine-specific |
| `__pycache__/` | Generated |
| `.env` | Secrets |
| `.idea/`, `.vscode/` | IDE settings |

## Reviewer flow (typical)

1. Skim **README.md** (~2 min) — can they run it?
2. Read **DESIGN_NOTES.md** (~5 min) — do you understand RAG design?
3. Glance **RESULTS.md** (~2 min) — does it work on the 10 tests?
4. Spot-check **code** (`ingestor.py`, `qa_engine.py`, `retrieval.py`) if time allows
5. Optionally run `python test_qa.py --ollama` themselves

## Email template

```
Subject: Policy Q&A Bot — Take-Home Submission — [Your Name]

Hi [Team],

Please find attached policy_qa_[yourname].zip — a local RAG Q&A system over
3 insurance policy PDFs with clause-level citations.

Quick start:
  pip install -r requirements.txt
  python main.py --build
  python test_qa.py --ollama    # requires: ollama pull mistral

Docs: README.md, DESIGN_NOTES.md, RESULTS.md

Thanks,
[Your Name]
```
