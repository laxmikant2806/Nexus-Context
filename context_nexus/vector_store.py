"""
context_nexus.vector_store
==========================
In-memory vector store integrating sentence-transformers (or TF-IDF fallback)
for vector embedding index management and similarity retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from context_nexus.hybrid_search import compute_cosine_distances

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store for chunk embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._embed_model: Any = self._load_model(model_name)

    def add_documents(self, doc_ids: list[str], texts: list[str]) -> None:
        """Embed and insert documents into the vector store."""
        if not doc_ids or not texts:
            return
        embeddings = self.encode(texts)
        for doc_id, text, emb in zip(doc_ids, texts, embeddings):
            self._doc_ids.append(doc_id)
            self._doc_texts.append(text)
            self._embeddings.append(emb)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search top_k documents ordered by cosine similarity descending."""
        if not self._embeddings:
            return []

        query_emb = self.encode([query])[0]
        distances = compute_cosine_distances(query_emb, self._embeddings)

        # Convert distance to similarity = 1.0 - distance
        scored = [
            (self._doc_ids[i], max(0.0, 1.0 - distances[i]))
            for i in range(len(self._doc_ids))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to float vectors."""
        if self._embed_model is not None:
            try:
                vecs = self._embed_model.encode(
                    texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                )
                return [v.tolist() for v in np.array(vecs, dtype=np.float32)]
            except Exception:
                pass

        # Fallback term-frequency vectorization
        return self._tfidf_fallback(texts)

    def _tfidf_fallback(self, texts: list[str]) -> list[list[float]]:
        from collections import Counter
        vocab: dict[str, int] = {}
        tokens_list = [t.lower().split() for t in texts]
        for tok_seq in tokens_list:
            for tok in tok_seq:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        dim = max(1, len(vocab))
        result = []
        for tok_seq in tokens_list:
            counts = Counter(tok_seq)
            vec = [0.0] * dim
            for tok, cnt in counts.items():
                if tok in vocab:
                    vec[vocab[tok]] = float(cnt)
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            result.append(vec)
        return result

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except Exception:
            return None
