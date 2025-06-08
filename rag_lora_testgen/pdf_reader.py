"""PDF loading utilities."""

from __future__ import annotations

import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def load_pdf_chunks(pdf_path: str, chunk_size: int = 500) -> list[str]:
    """Read ``pdf_path`` and split text into roughly ``chunk_size`` token chunks."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    text_parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n".join(text_parts)

    sentences = sent_tokenize(full_text)
    chunks: list[str] = []
    current_tokens: list[str] = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if len(current_tokens) + len(tokens) > chunk_size:
            chunks.append(" ".join(current_tokens))
            current_tokens = tokens
        else:
            current_tokens.extend(tokens)
    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks

