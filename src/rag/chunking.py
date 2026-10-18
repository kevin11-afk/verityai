"""
Text chunking with metadata preservation.

Every chunk carries enough metadata (title, page number, source URL, a
stable chunk ID) to produce a real citation later — retrieval.py and
generation.py never have to guess where a passage came from.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

from src import config
from src.rag.ingestion import LoadedDocument


# Bump when chunking changes: persisted indexes built with an older version are rebuilt.
CHUNKER_VERSION = "2"


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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREV_TAIL = re.compile(r"(?:\b(?:e\.g|i\.e|et al|vs|etc|approx|fig|dr|mr|mrs|ms|no|cf)|\b[A-Z])\.$", re.IGNORECASE)


def _hard_wrap(text: str, size: int) -> List[str]:
    """Split an over-long run of text at whitespace so no piece exceeds `size` (never mid-word)."""
    words, pieces, cur = text.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            pieces.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        pieces.append(cur)
    return pieces


_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s", re.MULTILINE)
_LIST_SPLIT = re.compile(r"\n(?=\s*(?:[-*+]|\d+\.)\s)")


def _units(para: str, chunk_size: int) -> List[tuple]:
    """(separator-within-paragraph, text) units of one paragraph. Headings and list items keep
    their own lines (so Markdown structure survives); prose is re-flowed and, when longer than a
    chunk, split into sentences."""
    units: List[tuple] = []
    first_line, _, rest = para.partition("\n")
    if first_line.lstrip().startswith("#") and rest.strip():
        units.append(("", first_line.strip()))
        para, lead = rest.strip(), "\n"
    else:
        lead = ""
    if _LIST_LINE.search(para):
        for i, item in enumerate(_LIST_SPLIT.split(para)):
            text = " ".join(item.split())
            if not text:
                continue
            for j, part in enumerate(_hard_wrap(text, chunk_size) if len(text) > chunk_size else [text]):
                units.append((lead if not units else ("\n" if j == 0 else " "), part))
        return units
    flat = " ".join(para.split())
    if not flat:
        return units
    if len(flat) <= chunk_size:
        units.append((lead if not units else "\n", flat))
        return units
    merged: List[str] = []
    for piece in _SENTENCE_BOUNDARY.split(flat):
        if merged and _ABBREV_TAIL.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    for sent in merged:
        for part in (_hard_wrap(sent, chunk_size) if len(sent) > chunk_size else [sent]):
            units.append((lead if not units else " ", part))
    return units


def _atoms(text: str, chunk_size: int) -> List[tuple]:
    """Break text into (separator, atom) units. Joining separator+atom in order reproduces the
    text (modulo whitespace), so every chunk is contiguous source text."""
    atoms: List[tuple] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        if not para.strip():
            continue
        for i, (sep, unit) in enumerate(_units(para.strip(), chunk_size)):
            atoms.append(("\n\n" if atoms and i == 0 else sep, unit))
    return atoms


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sentence-aware sliding window. Every chunk is contiguous text from the source and starts
    at a sentence (or paragraph) boundary; overlap is whole trailing sentences, never a character
    slice, so a chunk can never begin mid-word or contain spliced/duplicated text."""
    atoms = _atoms(text, chunk_size)
    chunks: List[str] = []
    cur: List[tuple] = []
    fresh = 0  # atoms in `cur` that are new (not carried over as overlap)

    def length(items: List[tuple]) -> int:
        return sum(len(sep) + len(atom) for sep, atom in items)

    def render(items: List[tuple]) -> str:
        return "".join(sep + atom for sep, atom in items).strip()

    for atom in atoms:
        if cur and length(cur) + len(atom[0]) + len(atom[1]) > chunk_size:
            chunks.append(render(cur))
            carry: List[tuple] = []
            for prev in reversed(cur):
                if overlap > 0 and length(carry) + len(prev[0]) + len(prev[1]) <= overlap:
                    carry.insert(0, prev)
                else:
                    break
            # never let the carried sentences crowd out the new atom
            while carry and length(carry) + len(atom[0]) + len(atom[1]) > chunk_size:
                carry.pop(0)
            cur = carry
            fresh = 0
        cur.append(atom)
        fresh += 1
    if cur and fresh > 0:  # a tail made only of carried-over overlap would duplicate the previous chunk
        chunks.append(render(cur))
    return [c for c in chunks if c]


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
