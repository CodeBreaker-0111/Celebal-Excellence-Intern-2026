"""
rag_pipeline.py
----------------
Ties together every stage of the pipeline into one RAGPipeline class:

  Document Ingestion -> Text Chunking -> Embedding Creation ->
  Vector Database -> Query Processing -> Context Retrieval ->
  Answer Generation
"""

from typing import List, Dict
from document_loader import load_documents
from chunking import chunk_documents
from retriever import build_retriever
from generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        top_k: int = 3,
        prefer_embeddings: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.prefer_embeddings = prefer_embeddings
        self.retriever = None
        self.chunks: List[Dict] = []

    def ingest(self, file_paths: List[str]) -> None:
        """Load documents, chunk them, embed them, and build the vector store."""
        docs = load_documents(file_paths)
        if not docs:
            raise ValueError("No documents could be loaded — check the file paths.")

        self.chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        print(f"[pipeline] Loaded {len(docs)} document(s) -> {len(self.chunks)} chunks.")

        self.retriever = build_retriever(self.chunks, prefer_embeddings=self.prefer_embeddings)

    def ask(self, query: str) -> Dict:
        """Run query processing, retrieval, and generation for one question."""
        if self.retriever is None:
            raise RuntimeError("No documents ingested yet — call ingest() first.")

        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        answer = generate_answer(query, retrieved)

        return {
            "question": query,
            "answer": answer,
            "sources": [{"source": c["source"], "score": c["score"]} for c in retrieved],
        }
