# Test Results

Evidence for the 10-question evaluation suite in `test_qa.py`.

## How to reproduce

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py --build            # or use bundled chroma_store/
ollama pull mistral               # optional — for real LLM answers
python test_qa.py --ollama --verbose
```

**Environment:** Windows, Python 3.11, Ollama `mistral` (local CPU), embeddings `all-MiniLM-L6-v2`.

**Documents indexed (3 PDFs in `docs/`):**

- `qbe_home_policy_wording_mil.pdf`
- `QM8698-1124 QBE Contents Insurance PDS.pdf`
- `QM9264-0723 Small Business Insurance.pdf`

**Index size:** 871 chunks (ChromaDB, `chroma_store/`).

---

## Summary table

| ID   | Category   | Substantive result | Notes |
|------|------------|-------------------|-------|
| IN-1 | in-domain  | Pass | NO — wear/tear excluded, Section 1 |
| IN-2 | in-domain  | Pass | 15-day money-back guarantee |
| IN-3 | in-domain  | Pass | NO — domestic animals owned by insured |
| IN-4 | in-domain  | Pass | HK$100k (16–65) and HK$20k (3–16), Section 5 |
| IN-5 | in-domain  | Pass | Up to 6 months temporary accommodation |
| NM-1 | near-miss  | Pass | NO for garden/open-air flood exclusion |
| NM-2 | near-miss  | Pass | No waiting period specified |
| NM-3 | near-miss  | Partial | Section 4 worldwide cover; theft/limit caveats need full clause read |
| OOS-1 | out-of-scope | Partial | States motor not covered; still mentions home claim steps (known limit) |
| OOS-2 | out-of-scope | Pass | Cannot find travel medical cover |

---

## Representative answers (Ollama / mistral)

### IN-1 — Wear and tear

**Q:** Is wear and tear covered under the home contents policy?

**A (excerpt):** NO — explicitly excluded under Section 1 Home Contents (“We will not pay for Loss or Damage… Wear, Tear”), with exception if another covered event results.

**Cited:** `qbe_home_policy_wording_mil.pdf` §Section 1 - Home Contents > What We cover > 1. We will not pay…, p.4

---

### IN-4 — Personal accident compensation

**Q:** What compensation is paid for accidental death under the personal accident section?

**A (excerpt):** HK$100,000 (ages 16–65); HK$20,000 (children 3–16). Section 5 - Personal Accident > Compensation, p.11.

---

### NM-1 — Flood + garden (near-miss)

**Q:** Is flood damage covered for my contents left in the garden?

**A (excerpt):** NO for garden/open-air contents — exclusion for Storm, Rainwater and Flood when contents are in the open air at Your Situation.

**Cited:** Section 1 open-air / emergency storage clauses, p.4

---

### OOS-2 — Travel medical (out-of-scope)

**Q:** Does this policy cover overseas travel medical emergencies?

**A (excerpt):** I cannot find a definitive answer… policy does not specify coverage for overseas travel medical emergencies (home policy only).

---

## Known limitations (honest)

1. **Out-of-scope (OOS-1):** May append unrelated home-insurance claim procedure after refusing motor cover.
2. **Near-miss (NM-1):** Answer emphasizes open-air exclusion; could also state general flood cover (Event 1(b)) before the garden exception.
3. **Citations:** Inline `[Doc: …, §…, p.…]` format; see `qa_engine.py` / `llm.py` for structured source metadata.
4. **LLM:** Results above used Ollama `mistral` on CPU. `python test_qa.py` without `--ollama` uses `MockLLM` for pipeline smoke tests only.

---

## Automated retrieval tests

```bash
python test_retrieval.py
```

Unit tests for query expansion and exclusion keyword supplements (no GPU / Ollama required).
