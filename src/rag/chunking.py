"""
Text chunking with metadata preservation.

Every chunk carries enough metadata (title, page number, source URL, a
stable chunk ID) to produce a real citation later — retrieval.py and
generation.py never have to guess where a passage came from.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from src import config
from src.rag.ingestion import LoadedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    title: str
    source_url: Optional[str]
    page_number: Optional[int]
    source_file: str


def _stable_chunk_id(source_file: str, page_number: Optional[int], ordinal: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    page_part = f"p{page_number}" if page_number is not None else "p-"
    return f"{source_file}:{page_part}:{ordinal}:{digest}"


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple, dependency-free sliding-window chunker on whitespace-normalized
    text. Splits on paragraph boundaries where possible, falling back to a
    hard character window so no chunk ever exceeds chunk_size by much."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long: hard-split with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
            current = ""

    if current:
        chunks.append(current)

    # Apply character-level overlap between adjacent chunks for better recall
    # across a boundary, without re-splitting already-correct short chunks.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            prefix = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(f"{prefix} {cur}")
        return overlapped
    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    ordinal = 0
    for page in document.pages:
        for piece in _split_text(page.text, chunk_size, overlap):
            if not piece.strip():
                continue
            chunk_id = _stable_chunk_id(document.source_path.name, page.page_number, ordinal, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    title=document.title,
                    source_url=document.source_url,
                    page_number=page.page_number,
                    source_file=document.source_path.name,
                )
            )
            ordinal += 1
    return chunks
