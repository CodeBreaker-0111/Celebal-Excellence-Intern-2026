"""
retriever.py
------------
Stages 3, 4, 6 of the RAG pipeline: Embedding Creation, Vector
Database, and Context Retrieval.

Two backends are supported:

1. TF-IDF (default) — pure scikit-learn, runs fully offline, no model
   download needed. Great for a "simple beginner" setup and for
   environments without internet access to a model hub.

2. Sentence-embeddings (optional) — if `sentence-transformers` is
   installed and its model can be downloaded, this gives true dense
   semantic search (captures meaning/synonyms, not just keyword
   overlap). Falls back to TF-IDF automatically if unavailable.

Both backends expose the same interface: build(chunks) then
retrieve(query, top_k).
"""

from typing import List, Dict
import numpy as np


class TfidfRetriever:
    """Simple, dependency-light vector store using TF-IDF + cosine similarity."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunks: List[Dict] = []
        self.matrix = None

    def build(self, chunks: List[Dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.matrix is None or len(self.chunks) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


class SentenceEmbeddingRetriever:
    """Dense retriever using sentence-transformers, if available."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict] = []
        self.embeddings = None

    def build(self, chunks: List[Dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


def build_retriever(chunks: List[Dict], prefer_embeddings: bool = True):
    """
    Factory: tries sentence-transformers first (better quality) and
    transparently falls back to TF-IDF if the package/model isn't
    available (e.g. no internet access to the model hub).
    """
    if prefer_embeddings:
        try:
            retriever = SentenceEmbeddingRetriever()
            retriever.build(chunks)
            print("[retriever] Using sentence-transformers (dense embeddings).")
            return retriever
        except Exception as e:
            print(f"[retriever] sentence-transformers unavailable ({e}); falling back to TF-IDF.")

    retriever = TfidfRetriever()
    retriever.build(chunks)
    print("[retriever] Using TF-IDF retriever.")
    return retriever
