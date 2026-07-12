"""
document_loader.py
-------------------
Stage 1 of the RAG pipeline: Document Ingestion.

Loads PDFs, .txt, and .md files and converts them into raw text,
tagged with their source filename (used later for citations).
"""

from pathlib import Path
from typing import List, Dict


def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file page by page."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def load_text(path: str) -> str:
    """Load a plain text / markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_document(path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (supported: .pdf, .txt, .md)")


def load_documents(paths: List[str]) -> List[Dict[str, str]]:
    """
    Load multiple documents.

    Returns a list of {"source": filename, "text": raw_text} dicts,
    skipping empty/unreadable files with a warning instead of crashing
    the whole pipeline.
    """
    docs = []
    for p in paths:
        try:
            text = load_document(p)
            if text.strip():
                docs.append({"source": Path(p).name, "text": text})
            else:
                print(f"[warn] No extractable text in {p} (scanned/image PDF?) — skipping.")
        except Exception as e:
            print(f"[warn] Could not load {p}: {e}")
    return docs
