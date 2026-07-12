# Document Question Answering System (RAG)

A simple, dependency-light Retrieval-Augmented Generation pipeline for
asking questions over your own PDFs / text files.

## Pipeline stages

```
PDF/TXT files
     │  document_loader.py
     ▼
raw text  ───────────────► chunking.py ──► overlapping text chunks
                                                    │
                                          retriever.py (TF-IDF or
                                          sentence-transformers)
                                                    │
                                            vector store (in-memory)
                                                    │
question ─────► embed query ──► cosine similarity search ──► top-k chunks
                                                    │
                                          generator.py (Anthropic /
                                          OpenAI / extractive fallback)
                                                    │
                                                answer + sources
```

## Files

| File | Stage | What it does |
|---|---|---|
| `document_loader.py` | Ingestion | Loads `.pdf` / `.txt` / `.md` into raw text |
| `chunking.py` | Chunking | Splits text into ~200-word overlapping chunks on sentence boundaries |
| `retriever.py` | Embedding + Vector DB + Retrieval | TF-IDF by default, dense embeddings via `sentence-transformers` if installed |
| `generator.py` | Generation | Anthropic / OpenAI if an API key is set, else extractive fallback (no key needed) |
| `rag_pipeline.py` | Orchestration | `RAGPipeline` class wiring every stage together |
| `app.py` | CLI | Command-line entry point |
| `sample_docs/rag_notes.txt` | Demo data | A short document about RAG itself, so you can try the system with zero setup |

## Quickstart

```bash
pip install -r requirements.txt

# Interactive Q&A loop
python app.py sample_docs/rag_notes.txt

# One-shot question
python app.py sample_docs/rag_notes.txt --ask "What is the main motivation for RAG?"

# Use your own files (PDF, txt, md — mix and match, multiple files allowed)
python app.py resume.pdf notes.txt --ask "What internships are listed?"
```

## Getting real generated answers (optional)

By default, with no API key set, the system runs in **extractive
mode**: it shows you the raw retrieved passages instead of a
synthesized answer. This keeps the whole pipeline runnable offline
with zero cost. To get fluent, LLM-written answers instead:

```bash
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here
```

The generator auto-detects whichever key is set (Anthropic is tried
first) and falls back to extractive mode if the call fails.

## Getting dense (semantic) retrieval (optional)

By default retrieval uses TF-IDF (keyword-based, works fully
offline). For true semantic search — e.g. matching "car" to
"automobile" — install `sentence-transformers`:

```bash
pip install sentence-transformers
```

`retriever.py` will automatically detect it and switch to dense
embeddings; no code changes needed. If the package or its model
weights can't be downloaded, it transparently falls back to TF-IDF.

## Tuning

```bash
python app.py doc.pdf --chunk-size 300 --top-k 5
python app.py doc.pdf --no-embeddings   # force TF-IDF even if sentence-transformers is installed
```

## Suggested experiments (from the assignment)

- **Chunking**: try `--chunk-size 100` vs `300` and compare retrieval quality
- **Embeddings**: compare TF-IDF vs `sentence-transformers` results for the same question
- **Hybrid search**: combine TF-IDF and dense scores (e.g. weighted sum) in `retriever.py`
- **Re-ranking**: after retrieving top-10 with TF-IDF, re-score with a cross-encoder before picking top-3
- **Different LLMs**: swap the model name in `generator.py`, or add a local model backend (e.g. via `transformers`)
