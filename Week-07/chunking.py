"""
chunking.py
-----------
Stage 2 of the RAG pipeline: Text Chunking.

Splits raw document text into smaller overlapping chunks. Chunking
matters a lot for retrieval quality: chunks too large dilute the
embedding/TF-IDF signal, chunks too small lose context. This uses a
simple sliding-window word-based splitter with overlap, and prefers
to break on sentence boundaries where possible.
"""

import re
from typing import List, Dict


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # naive sentence splitter (good enough for beginner-level RAG)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into chunks of ~chunk_size words, with `overlap` words
    repeated between consecutive chunks so context isn't lost at chunk
    boundaries.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_words: List[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # start next chunk with overlap from the end of the previous one
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_documents(
    docs: List[Dict[str, str]],
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Dict]:
    """
    Chunk a list of {"source", "text"} documents into a flat list of
    {"id", "source", "text"} chunk records ready for embedding.
    """
    all_chunks = []
    chunk_id = 0
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            all_chunks.append({
                "id": chunk_id,
                "source": doc["source"],
                "text": piece,
            })
            chunk_id += 1
    return all_chunks
