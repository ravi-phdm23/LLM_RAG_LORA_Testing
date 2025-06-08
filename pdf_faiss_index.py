import json
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from rwa_pdf_loader import load_basel_summary_chunks


def build_faiss_index(
    pdf_path: str = "RWA_docs/Basel_summary.pdf",
    index_path: str = "faiss.index",
    metadata_path: str = "faiss.json",
    chunk_size: int = 500,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[faiss.Index, List[str]]:
    """Embed PDF text and save a FAISS index to disk."""

    chunks = load_basel_summary_chunks(pdf_path, chunk_size)
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
    """Load a FAISS index and associated texts."""

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
    """Return the top ``top_k`` matching text chunks for ``query``."""

    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results: List[Tuple[str, float]] = []
    for score, idx in zip(distances[0], indices[0]):
        results.append((chunks[idx], float(score)))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query a FAISS index from a PDF")
    parser.add_argument("--build", action="store_true", help="Build the index and exit")
    parser.add_argument("--pdf", default="RWA_docs/Basel_summary.pdf", help="Path to PDF file")
    parser.add_argument("--index", default="faiss.index", help="Path to FAISS index file")
    parser.add_argument("--meta", default="faiss.json", help="Path to metadata file")
    parser.add_argument("--query", help="Query to search for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.build:
        build_faiss_index(args.pdf, args.index, args.meta)
    elif args.query:
        index, chunks, model = load_faiss_index(args.index, args.meta)
        for text, score in search_index(args.query, index, chunks, model, args.top_k):
            print(f"{score:.4f}\t{text[:200]}")
    else:
        parser.print_help()
