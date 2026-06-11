"""
test_qa.py
----------
10 Q&A test cases: 5 in-domain, 3 near-miss, 2 out-of-scope.

Run
---
  python test_qa.py                   # MockLLM (no model needed)
  python test_qa.py --ollama          # real answers via Ollama
  python test_qa.py --build-first     # build index then run tests
  python test_qa.py --verbose         # show retrieved chunks per question
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestor     import ingest_folder
from embeddings   import Embedder
from vector_store import VectorStore
from llm          import MockLLM, OllamaLLM
from qa_engine    import QAEngine


BASE      = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(BASE, "docs")
INDEX_DIR = os.path.join(BASE, "chroma_store")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ---- IN-DOMAIN (5): clear answers exist in the documents ---------------
    {
        "id": "IN-1", "category": "in-domain",
        "question": "Is wear and tear covered under the home contents policy?",
        "expected": (
            "NO. Explicitly excluded under Section 1 Exclusions 1(c)(ii). "
            "Cited from qbe_home_policy or QBE Contents PDS, p.9."
        ),
    },
    {
        "id": "IN-2", "category": "in-domain",
        "question": "What is the money-back guarantee period for a new policy?",
        "expected": (
            "15 days. Policyholder can cancel within 15 days of the policy "
            "start date for a full premium refund."
        ),
    },
    {
        "id": "IN-3", "category": "in-domain",
        "question": "Does the policy cover damage caused by domestic animals I own?",
        "expected": (
            "NO. Section 1 Exclusion 1(h) explicitly excludes damage caused "
            "by domestic animals owned by or in the care of the insured."
        ),
    },
    {
        "id": "IN-4", "category": "in-domain",
        "question": "What compensation is paid for accidental death under the personal accident section?",
        "expected": (
            "HK$100,000 for persons aged 16-65; HK$20,000 for children "
            "aged 3-16. Section 5 - Personal Accident."
        ),
    },
    {
        "id": "IN-5", "category": "in-domain",
        "question": "What is the maximum period for temporary accommodation cover?",
        "expected": (
            "Up to 6 months. Section 1 Event 21 / Section 2 Event 7."
        ),
    },

    # ---- NEAR-MISS (3): related content but answer needs careful reasoning --
    {
        "id": "NM-1", "category": "near-miss",
        "question": "Is flood damage covered for my contents left in the garden?",
        "expected": (
            "Partial/conflicting: flood IS covered for Home Contents generally "
            "(Event 1(b)), BUT the open-air sub-clause explicitly EXCLUDES "
            "Storm, Rainwater and Flood (Event 2(a)). Garden items not covered."
        ),
    },
    {
        "id": "NM-2", "category": "near-miss",
        "question": "What is the waiting period before I can make an accidental damage claim?",
        "expected": (
            "No waiting period is defined. Model should say it cannot find "
            "a specific waiting period and note that cover applies during "
            "the Period of Insurance."
        ),
    },
    {
        "id": "NM-3", "category": "near-miss",
        "question": "Can I claim for my laptop stolen from a coffee shop?",
        "expected": (
            "Laptops are Unspecified Personal Valuables under Section 4, "
            "covered anywhere in the world — but Section 4 is not operative "
            "when building is leased to a tenant, and per-item limits apply."
        ),
    },

    # ---- OUT-OF-SCOPE (2): topic absent from these documents ----------------
    {
        "id": "OOS-1", "category": "out-of-scope",
        "question": "What is the claims process for motor vehicle accidents?",
        "expected": (
            "Motor vehicle insurance is not in these documents. "
            "Model should return 'cannot find a definitive answer'."
        ),
    },
    {
        "id": "OOS-2", "category": "out-of-scope",
        "question": "Does this policy cover overseas travel medical emergencies?",
        "expected": (
            "Travel insurance is not covered. Model should say 'cannot find'. "
            "Must not confuse with Section 5 which covers burglary injuries at home."
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(engine: QAEngine, verbose: bool = False):
    print("\n" + "=" * 70)
    print("  POLICY Q&A BOT — TEST SUITE")
    print("=" * 70)

    summary = []

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] ({tc['category'].upper()})")
        print(f"Q: {tc['question']}")
        print(f"Expected: {tc['expected'][:100]}...")
        print("-" * 50)

        result = engine.ask(tc["question"])

        preview = result.answer[:400] + ("..." if len(result.answer) > 400 else "")
        print(f"A: {preview}")
        llm_failed = result.answer.startswith("[OllamaLLM ERROR]")
        print(f"Grounded: {result.is_grounded}" + ("  (LLM call failed)" if llm_failed else ""))
        for s in result.sources[:2]:
            print(f"  Cited: {s['doc_name']}  §{s['section'][:55]}  p.{s['page']}")

        if verbose:
            print("\nTop chunks:")
            for chunk, score in result.raw_chunks[:3]:
                print(f"  [{score:.3f}] {chunk.doc_name} p.{chunk.page} — {chunk.text[:80]}")

        summary.append({
            "id":          tc["id"],
            "category":    tc["category"],
            "grounded":    result.is_grounded,
            "has_sources": len(result.sources) > 0,
            "llm_failed":  llm_failed,
        })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':<8} {'Category':<14} {'Grounded':<12} {'LLM OK':<8} {'Has Sources'}")
    for r in summary:
        print(
            f"{r['id']:<8} {r['category']:<14} {str(r['grounded']):<12} "
            f"{str(not r['llm_failed']):<8} {r['has_sources']}"
        )

    in_d = [r for r in summary if r["category"] == "in-domain"]
    nm   = [r for r in summary if r["category"] == "near-miss"]
    oos  = [r for r in summary if r["category"] == "out-of-scope"]
    print(f"\nIn-domain grounded         : {sum(r['grounded'] for r in in_d)}/{len(in_d)}")
    print(f"Near-miss grounded         : {sum(r['grounded'] for r in nm)}/{len(nm)}")
    print(f"Out-of-scope NOT grounded  : {sum(not r['grounded'] for r in oos)}/{len(oos)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run the Q&A test suite.")
    p.add_argument("--build-first", action="store_true", help="Build index before testing.")
    p.add_argument("--ollama",      action="store_true", help="Use OllamaLLM.")
    p.add_argument("--model",       default="mistral",   help="Ollama model name.")
    p.add_argument("--docs",        default=DOCS_DIR)
    p.add_argument("--index",       default=INDEX_DIR)
    p.add_argument("--verbose",     action="store_true")
    p.add_argument(
        "--ollama-timeout",
        type=int,
        default=600,
        help="Seconds to wait per Ollama request (default: 600).",
    )
    args = p.parse_args()

    embedder = Embedder()
    store    = VectorStore(args.index)

    if args.build_first:
        chunks = ingest_folder(args.docs)
        store.build(chunks, embedder)
    else:
        store.load(embedder)

    llm = (
        OllamaLLM(model=args.model, timeout=args.ollama_timeout)
        if args.ollama
        else MockLLM()
    )
    engine = QAEngine(store, llm)
    run_tests(engine, verbose=args.verbose)


if __name__ == "__main__":
    main()
