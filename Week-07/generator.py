"""
generator.py
------------
Stage 7 of the RAG pipeline: Answer Generation.

Builds a prompt from the retrieved chunks ("augmentation") and asks a
language model to answer using only that context. Supports Anthropic
and OpenAI as LLM backends (auto-detected from environment variables),
and falls back to a simple extractive answer (no API key needed) so
the whole pipeline still runs end-to-end without any external service.
"""

import os
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from the user's documents. If the answer isn't "
    "in the context, say you don't have enough information — do not "
    "make anything up. Keep answers concise and cite the source file "
    "name(s) you used."
)


def build_prompt(query: str, chunks: List[Dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context from documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Answer the question using only the context above."
    )


def _generate_with_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _generate_with_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content


def _generate_extractive(query: str, chunks: List[Dict]) -> str:
    """
    No-LLM fallback: return the most relevant retrieved sentences
    directly, formatted as an answer. Keeps the pipeline runnable with
    zero external dependencies / API keys — useful for offline demos
    and as a baseline to compare LLM-generated answers against.
    """
    if not chunks:
        return "I couldn't find anything relevant to that question in the documents."

    lines = ["Based on the retrieved context (extractive mode — no LLM configured):\n"]
    for c in chunks:
        snippet = c["text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"- ({c['source']}, relevance {c['score']:.2f}): {snippet}")
    lines.append(
        "\n[Set ANTHROPIC_API_KEY or OPENAI_API_KEY to get a fluent, "
        "synthesized answer instead of raw excerpts.]"
    )
    return "\n".join(lines)


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Try Anthropic, then OpenAI, then fall back to extractive answering.
    Whichever backend is used, the answer is grounded only in `chunks`.
    """
    prompt = build_prompt(query, chunks)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _generate_with_anthropic(prompt)
        except Exception as e:
            print(f"[generator] Anthropic call failed ({e}); trying next backend.")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _generate_with_openai(prompt)
        except Exception as e:
            print(f"[generator] OpenAI call failed ({e}); falling back to extractive mode.")

    return _generate_extractive(query, chunks)
