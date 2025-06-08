"""Utilities for building and querying FAISS indices."""

from __future__ import annotations

import json
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .pdf_reader import load_pdf_chunks


def build_faiss_index(
    pdf_path: str,
    index_path: str = "faiss.index",
    metadata_path: str = "faiss.json",
    chunk_size: int = 500,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[faiss.Index, List[str]]:
    """Embed text from ``pdf_path`` and save a FAISS index."""

    chunks = load_pdf_chunks(pdf_path, chunk_size)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh)

    return index, chunks


def load_faiss_index(
    index_path: str = "faiss.index",
    metadata_path: str = "faiss.json",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """Load a FAISS index with its associated text chunks."""

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    model = SentenceTransformer(model_name)
    return index, chunks, model


def search_index(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return ``top_k`` text chunks that best match ``query``."""

    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results: List[Tuple[str, float]] = []
    for score, idx in zip(distances[0], indices[0]):
        results.append((chunks[idx], float(score)))
    return results


def build_query_prompt(
    user_query: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 3,
) -> str:
    """Return retrieved context and the user query as a single prompt."""

    results = search_index(user_query, index, chunks, model, top_k)
    context = "\n".join(text for text, _ in results)
    return f"{context}\n\n{user_query}"

