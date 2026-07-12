"""
app.py
------
Command-line interface for the RAG system.

Usage:
    python app.py doc1.pdf doc2.txt              # interactive Q&A loop
    python app.py doc1.pdf --ask "What is X?"     # single question, then exit
"""

import argparse
from rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ask questions over your own documents (RAG).")
    parser.add_argument("files", nargs="+", help="Paths to PDF/.txt/.md documents to ingest")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question and exit")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Force TF-IDF retrieval instead of trying sentence-transformers first",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        prefer_embeddings=not args.no_embeddings,
    )
    pipeline.ingest(args.files)

    def ask_and_print(question: str):
        result = pipeline.ask(question)
        print("\n" + "=" * 60)
        print(f"Q: {result['question']}")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        srcs = ", ".join(f"{s['source']} ({s['score']:.2f})" for s in result["sources"])
        print(f"Sources: {srcs if srcs else 'none'}")
        print("=" * 60 + "\n")

    if args.ask:
        ask_and_print(args.ask)
        return

    print("\nRAG system ready. Type a question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        ask_and_print(question)


if __name__ == "__main__":
    main()
