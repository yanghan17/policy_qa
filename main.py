"""
main.py
-------
CLI entry point for the Policy Q&A Bot.

Usage
-----
  # Build the vector index (run once, or when docs change)
  python main.py --build

  # Ask a single question  (MockLLM by default)
  python main.py --ask "Is wear and tear covered?"

  # Use Ollama for real answers  (ollama pull mistral first)
  python main.py --ask "Is wear and tear covered?" --ollama
  python main.py --ask "Is wear and tear covered?" --ollama --model llama3

  # Use a local HuggingFace model
  python main.py --ask "..." --hf-model microsoft/phi-2

  # Use LM Studio or any OpenAI-compatible local server
  python main.py --ask "..." --lmstudio

  # Interactive session
  python main.py --interactive --ollama

  # Show retrieved chunks alongside the answer
  python main.py --ask "..." --verbose
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestor     import ingest_folder
from embeddings   import Embedder
from vector_store import VectorStore
from llm          import MockLLM, OllamaLLM, HuggingFaceLocalLLM, OpenAICompatibleLLM
from qa_engine    import QAEngine


BASE      = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(BASE, "docs")
INDEX_DIR = os.path.join(BASE, "chroma_store")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def cmd_build(docs_dir: str, index_dir: str):
    print(f"\n[BUILD] Ingesting PDFs from: {docs_dir}")
    chunks = ingest_folder(docs_dir)
    if not chunks:
        sys.exit("ERROR: No chunks produced — check the docs/ folder.")

    embedder = Embedder()
    store    = VectorStore(index_dir)
    store.build(chunks, embedder)

    print(f"\n[BUILD] Complete.")
    print(f"        Chunks   : {len(chunks)}")
    print(f"        Embedder : {embedder.name}")
    print(f"        Index    : {index_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(index_dir: str):
    embedder = Embedder()
    store    = VectorStore(index_dir)
    store.load(embedder)
    return store


def _make_llm(args) -> object:
    if getattr(args, "ollama", False):
        return OllamaLLM(
            model=getattr(args, "model", "mistral"),
            timeout=getattr(args, "ollama_timeout", 600),
        )
    if getattr(args, "hf_model", None):
        return HuggingFaceLocalLLM(model_name=args.hf_model)
    if getattr(args, "lmstudio", False):
        return OpenAICompatibleLLM()
    return MockLLM()


# ---------------------------------------------------------------------------
# Ask / Interactive
# ---------------------------------------------------------------------------

def cmd_ask(args):
    store  = _load(args.index)
    engine = QAEngine(store, _make_llm(args), top_k=args.top_k)
    result = engine.ask(args.ask)

    if args.verbose:
        print("\n--- Retrieved chunks ---")
        for chunk, score in result.raw_chunks:
            print(f"  [{score:.3f}] {chunk.doc_name}  p.{chunk.page}")
            print(f"           {chunk.text[:110]}...\n")

    print(result.pretty())


def cmd_interactive(args):
    store  = _load(args.index)
    llm    = _make_llm(args)
    engine = QAEngine(store, llm, top_k=args.top_k)

    print(f"\nPolicy Q&A Bot  |  LLM: {llm.name}")
    print("Type your question and press Enter.  'quit' to exit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        print(engine.ask(question).pretty())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Policy Q&A Bot — RAG over insurance PDFs")

    p.add_argument("--build",       action="store_true", help="Ingest PDFs and build index.")
    p.add_argument("--ask",         metavar="QUESTION",  help="Ask a single question.")
    p.add_argument("--interactive", action="store_true", help="Interactive Q&A session.")

    p.add_argument("--docs",  default=DOCS_DIR,  help=f"PDF folder  (default: {DOCS_DIR})")
    p.add_argument("--index", default=INDEX_DIR, help=f"Index folder (default: {INDEX_DIR})")
    p.add_argument("--top-k", type=int, default=6, help="Chunks to retrieve (default: 6)")

    llm_group = p.add_mutually_exclusive_group()
    llm_group.add_argument("--ollama",   action="store_true", help="Use OllamaLLM (local)")
    llm_group.add_argument("--hf-model", metavar="MODEL",     help="Use HuggingFace local model")
    llm_group.add_argument("--lmstudio", action="store_true", help="Use LM Studio local server")

    p.add_argument("--model",   default="mistral", help="Ollama model name (default: mistral)")
    p.add_argument(
        "--ollama-timeout",
        type=int,
        default=600,
        help="Seconds to wait per Ollama request (default: 600)",
    )
    p.add_argument("--verbose", action="store_true", help="Print retrieved chunks.")

    args = p.parse_args()

    if not (args.build or args.ask or args.interactive):
        p.print_help()
        sys.exit(0)

    if args.build:
        cmd_build(args.docs, args.index)
    if args.ask:
        cmd_ask(args)
    if args.interactive:
        cmd_interactive(args)


if __name__ == "__main__":
    main()
