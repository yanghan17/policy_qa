"""
embeddings.py
-------------
Embedding model using sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384-dimensional L2-normalised dense vectors
  - Fine-tuned on 1B+ sentence pairs for semantic similarity
  - 22 MB download, runs on CPU, ~5ms per sentence
  - First use auto-downloads from HuggingFace to ~/.cache/

Install: pip install sentence-transformers

To swap models, change DEFAULT_MODEL. Good alternatives:
  "all-mpnet-base-v2"              — better quality, 420 MB, 768-dim
  "multi-qa-MiniLM-L6-cos-v1"     — tuned specifically for Q&A retrieval
  "BAAI/bge-small-en-v1.5"        — strong, 130 MB, 384-dim
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps a SentenceTransformer model with a clean encode() interface.

    encode(texts) returns a float32 numpy array of shape (N, 384).
    Vectors are L2-normalised (unit length), so dot product == cosine
    similarity — this is what ChromaDB expects with hnsw:space=cosine.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Embedder] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._name  = model_name
        # Determine dim from a single test encode
        self._dim   = self._model.encode(["test"]).shape[1]
        print(f"[Embedder] Ready — model={model_name}, dim={self._dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into a (N, dim) float32 array.
        normalize_embeddings=True ensures unit-length vectors.
        """
        return self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name
