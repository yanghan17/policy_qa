"""
vector_store.py
---------------
ChromaDB-backed vector store.

ChromaDB runs fully in-process (no server), persists to disk automatically,
and uses an HNSW index for fast approximate nearest-neighbour search.

Install: pip install chromadb
"""

import json
import os
from typing import List, Optional, Tuple

import chromadb

from ingestor import Chunk
from embeddings import Embedder


COLLECTION_NAME = "policy_chunks"


class VectorStore:
    """
    Stores chunk embeddings in ChromaDB and retrieves by cosine similarity.

    Usage
    -----
    store = VectorStore("chroma_store/")
    store.build(chunks, embedder)   # index all chunks — run once

    store = VectorStore("chroma_store/")
    store.load(embedder)            # reload existing index
    results = store.search("is wear and tear covered?", k=5)
    # → List[(Chunk, similarity_score)]
    """

    def __init__(self, chroma_dir: str = "chroma_store"):
        self.chroma_dir = chroma_dir
        os.makedirs(chroma_dir, exist_ok=True)
        self._client     = chromadb.PersistentClient(path=chroma_dir)
        self._collection = None
        self._embedder: Embedder | None = None
        # Local registry: chunk_id → Chunk (for fast lookup after query)
        self._chunks: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk], embedder: Embedder) -> None:
        """
        Embed all chunks and upsert into ChromaDB.

        Uses upsert so re-running --build is safe (idempotent).
        Deletes the existing collection first to avoid stale data.
        """
        self._embedder = embedder

        # Fresh rebuild: delete old collection if it exists
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name     = COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )

        print(f"[VectorStore] Embedding {len(chunks)} chunks...")

        # Batch upsert for speed
        BATCH = 200
        for i in range(0, len(chunks), BATCH):
            batch  = chunks[i : i + BATCH]
            ids    = [c.chunk_id for c in batch]
            texts  = [c.text for c in batch]
            metas  = [
                {
                    "doc_name":     c.doc_name,
                    "page":         c.page,
                    "section":      c.section,
                    "heading_path": c.heading_path,
                    "clause_ref":   c.clause_ref,
                }
                for c in batch
            ]
            vecs = embedder.encode(texts).tolist()
            self._collection.upsert(
                ids        = ids,
                embeddings = vecs,
                documents  = texts,
                metadatas  = metas,
            )
            print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)} chunks upserted")

        # Save chunk registry so we can reconstruct Chunk objects on load
        registry_path = os.path.join(self.chroma_dir, "chunks.json")
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False)

        self._chunks = {c.chunk_id: c for c in chunks}
        print(f"[VectorStore] Done. {len(chunks)} chunks indexed.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, embedder: Embedder) -> None:
        """Reload a previously built index from disk."""
        self._embedder   = embedder
        self._collection = self._client.get_collection(COLLECTION_NAME)

        registry_path = os.path.join(self.chroma_dir, "chunks.json")
        with open(registry_path, encoding="utf-8") as f:
            self._chunks = {
                d["chunk_id"]: Chunk.from_dict(d)
                for d in json.load(f)
            }
        print(f"[VectorStore] Loaded {len(self._chunks)} chunks from {self.chroma_dir}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query:      str,
        k:          int = 5,
        doc_names:  Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Return the top-k most relevant chunks for a query.

        ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
        We convert to similarity: similarity = 1 - distance.

        Parameters
        ----------
        doc_names : If set, restrict search to these document filenames.

        Returns
        -------
        List of (Chunk, similarity) sorted by descending similarity.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not built or loaded.")

        where = None
        if doc_names:
            if len(doc_names) == 1:
                where = {"doc_name": doc_names[0]}
            else:
                where = {"doc_name": {"$in": doc_names}}

        query_vec = self._embedder.encode([query]).tolist()
        kwargs = {
            "query_embeddings": query_vec,
            "n_results":        k,
            "include":          ["distances"],
        }
        if where is not None:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)

        output = []
        for chunk_id, distance in zip(
            results["ids"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - float(distance)
            chunk = self._chunks.get(chunk_id)
            if chunk:
                output.append((chunk, similarity))

        return sorted(output, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        count = self._collection.count() if self._collection else 0
        return {
            "num_chunks": count,
            "embedder":   self._embedder.name if self._embedder else "none",
            "docs":       list({c.doc_name for c in self._chunks.values()}),
        }
