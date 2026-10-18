"""Extractive, citation-grounded answers. No text is generated: every claim is
a sentence taken verbatim (after Markdown cleanup) from a retrieved passage, so
the answer cannot contain a statement that is not in the knowledge base."""
from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

from src.rag.retrieval import STOPWORDS as _STOP

_WORD = re.compile(r"[a-z0-9]+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


_ABBREV_END = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|vs|etc|approx|fig|dr|mr|mrs|ms|no|cf|ca|inc|ltd)|\b[A-Z])\.$", re.IGNORECASE)
_DANGLING_START = re.compile(r"^(it|its|this|these|those|they|their|such|that|he|she|there)\b", re.IGNORECASE)


def _split_sentences(block: str) -> List[str]:
    """Sentence split that does not break after abbreviations ("e.g.", "et al.", initials)."""
    pieces = _SENT_END.split(block)
    merged: List[str] = []
    for piece in pieces:
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _inline_clean(text: str) -> str:
    text = re.sub(r"[*_`]{1,3}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _blocks(text: str) -> List[Tuple[str, str]]:
    """Split Markdown-ish text into (kind, cleaned text) blocks: "heading", "item" or "para".
    Hard-wrapped lines are joined into their paragraph or list item; headings stand alone."""
    out: List[Tuple[str, str]] = []
    kind = "para"
    lines: List[str] = []

    def flush() -> None:
        joined = _inline_clean(" ".join(lines))
        if joined:
            out.append((kind, joined))
        lines.clear()

    for line in text.splitlines():
        if not line.strip():
            flush()
            kind = "para"
        elif _HEADING.match(line):
            flush()
            out.append(("heading", _inline_clean(_HEADING.sub("", line))))
            kind = "para"
        elif _LIST_ITEM.match(line):
            flush()
            kind = "item"
            lines.append(_LIST_ITEM.sub("", line))
        else:
            lines.append(line.lstrip()[1:] if line.lstrip().startswith(">") else line)
    flush()
    return out


def clean_markdown(text: str) -> str:
    """Passage text as prose: Markdown syntax removed, whitespace collapsed. Claims are
    verified to occur verbatim inside this string."""
    return " ".join(t for _, t in _blocks(text))


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _terms(text: str) -> List[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def _candidate_sentences(text: str) -> List[str]:
    """Whole sentences usable as claims: not headings, not source/citation lines, not fragments
    that begin mid-sentence (chunk boundaries), not URLs."""
    out: List[str] = []
    for kind, block in _blocks(text):
        if kind == "heading" or block.lower().startswith("source:"):
            continue
        for s in _split_sentences(block):
            s = s.strip()
            if not (40 <= len(s) <= 420) or not (s[0].isupper() or s[0].isdigit() or s[0] in "\"'("):
                continue
            if "http://" in s or "https://" in s or s.lower().startswith("source:"):
                continue
            out.append(s)
    return out


def compose_answer(question: str, passages: Sequence, max_claims: int = 3) -> List[Tuple[str, int]]:
    """Return up to `max_claims` (sentence, citation_id) pairs, best match first,
    at most one per passage so the answer draws on distinct sources."""
    q_terms = set(_terms(question))
    if not q_terms:
        return []
    scored = []
    for passage in passages:
        best = None
        for sentence in _candidate_sentences(passage.text):
            overlap = len(q_terms & set(_terms(sentence)))
            if overlap == 0:
                continue
            # Reward coverage of the question; lightly penalise long sentences.
            score = overlap / math.sqrt(len(q_terms)) - 0.002 * len(sentence)
            if _DANGLING_START.match(sentence):
                score -= 0.6  # opens with a pronoun whose referent is outside the sentence
            if best is None or score > best[0]:
                best = (score, sentence)
        if best:
            scored.append((best[0], best[1], passage.citation_id))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(sentence, cid) for _, sentence, cid in scored[:max_claims]]
