"""Chunker guarantees: contiguous source text, sentence-aligned, size-bounded, deterministic."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.rag.chunking import CHUNKER_VERSION, _split_text  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731

PROSE = " ".join(f"Sentence number {i} explains detector behaviour in some detail for testing." for i in range(60))


def test_every_chunk_is_contiguous_source_text_within_size():
    chunks = _split_text(PROSE, 300, 80)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c) <= 300
        assert norm(c) in norm(PROSE), "chunk contains text that is not in the source"


def test_chunks_start_at_sentence_boundaries_and_never_mid_word():
    for c in _split_text(PROSE, 300, 80):
        assert c.startswith("Sentence number"), c[:40]


def test_overlap_is_whole_trailing_sentences():
    chunks = _split_text(PROSE, 300, 120)
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail in chunks[1], "the last sentence of a chunk should open the next one"


def test_no_text_is_lost_or_duplicated_inside_a_chunk():
    for c in _split_text(PROSE, 300, 80):
        sentences = re.findall(r"Sentence number (\d+)", c)
        assert len(sentences) == len(set(sentences)), "duplicated sentence inside one chunk"
        assert [int(x) for x in sentences] == sorted(int(x) for x in sentences)


def test_oversize_sentence_is_wrapped_at_whitespace():
    long_sentence = "word " * 400
    chunks = _split_text(long_sentence.strip(), 200, 0)
    assert all(len(c) <= 200 for c in chunks)
    assert all(not c.startswith(" ") and not c.endswith("wor") for c in chunks)
    assert norm(" ".join(chunks)) == norm(long_sentence)


def test_markdown_lists_and_headings_keep_their_lines():
    doc = "# Title\n\n## Models\n\n- **A:** first item is here.\n- **B:** second item\n  wraps onto another line.\n"
    out = "\n".join(_split_text(doc, 1000, 0))
    assert "## Models" in out and "- **A:** first item is here." in out
    assert "- **B:** second item wraps onto another line." in out


def test_real_knowledge_base_chunks_are_all_contiguous():
    for path in sorted(config.KNOWLEDGE_BASE_DIR.glob("0*.md")):
        text = path.read_text()
        for c in _split_text(text, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS):
            assert norm(c) in norm(text), f"{path.name}: spliced chunk"
            assert len(c) <= config.CHUNK_SIZE_CHARS


def test_chunking_is_deterministic():
    assert _split_text(PROSE, 300, 80) == _split_text(PROSE, 300, 80)


def test_empty_and_tiny_inputs():
    assert _split_text("", 300, 80) == []
    assert _split_text("   \n\n  ", 300, 80) == []
    assert _split_text("Just one short sentence.", 300, 80) == ["Just one short sentence."]


def test_stale_index_is_detected_and_rebuilt(tmp_path):
    pytest.importorskip("chromadb")
    from src.rag.chunking import Chunk
    from src.rag.vector_store import VectorStore

    def fake(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    with patch("src.rag.vector_store.embed_texts", side_effect=fake):
        store = VectorStore(persist_directory=tmp_path / "c", collection_name="chunker_test")
        store.add_chunks([Chunk("a", "text", "T", None, None, "a.md")])
        assert not store.stale()
        store._version_file().write_text("old")
        assert store.stale()
        store.rebuild([Chunk("b", "text", "T", None, None, "b.md")])
        assert not store.stale() and store.count() == 1
        store._version_file().unlink()  # an index with no version record counts as stale
        assert store.stale()
    assert CHUNKER_VERSION
