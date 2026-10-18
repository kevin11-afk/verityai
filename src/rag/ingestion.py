"""
Document loading for the RAG knowledge base: PDF, TXT, and Markdown.

Treats every document as untrusted input (per the project's security
requirements): parsing failures are caught per-file rather than crashing
the whole ingestion run, and no content extracted from a document is ever
executed or interpolated into a prompt/template as anything other than
plain text evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LoadedPage:
    page_number: Optional[int]  # 1-indexed; None for formats without pages (TXT/MD)
    text: str


@dataclass
class LoadedDocument:
    source_path: Path
    title: str
    source_url: Optional[str]
    pages: List[LoadedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _read_source_url_sidecar(path: Path) -> Optional[str]:
    """Optional convention: a document "report.pdf" may ship with a
    "report.pdf.url" sidecar file containing its canonical source URL, so we
    can cite a real link without guessing one."""
    sidecar = path.with_suffix(path.suffix + ".url")
    if sidecar.exists():
        try:
            url = sidecar.read_text(encoding="utf-8").strip()
            return url or None
        except OSError:
            return None
    return None


def load_pdf(path: Path) -> LoadedDocument:
    import fitz  # PyMuPDF

    pages: List[LoadedPage] = []
    title = path.stem
    try:
        with fitz.open(path) as doc:
            metadata_title = (doc.metadata or {}).get("title")
            if metadata_title:
                title = metadata_title
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(LoadedPage(page_number=i, text=text))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse PDF '{path.name}': {exc}") from exc

    return LoadedDocument(
        source_path=path, title=title, source_url=_read_source_url_sidecar(path), pages=pages
    )


def load_text(path: Path) -> LoadedDocument:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Failed to read text file '{path.name}': {exc}") from exc

    title = path.stem
    # Markdown convention: first "# Heading" line becomes the title.
    if path.suffix.lower() == ".md":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

    return LoadedDocument(
        source_path=path,
        title=title,
        source_url=_read_source_url_sidecar(path),
        pages=[LoadedPage(page_number=None, text=text)],
    )


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md"):
        return load_text(path)
    raise ValueError(f"Unsupported document type '{suffix}' for '{path.name}'.")


def load_knowledge_base_directory(directory: Path) -> List[LoadedDocument]:
    """Load every supported document in a directory, skipping and reporting
    (rather than crashing on) any file that fails to parse."""
    documents: List[LoadedDocument] = []
    errors: List[str] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        if path.suffix.lower() == ".url":
            continue
        if path.parent == directory and path.name.lower() == "readme.md":
            continue  # the folder's own how-to is documentation, not a citable source
        try:
            documents.append(load_document(path))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        import logging
        for e in errors:
            logging.getLogger(__name__).warning(e)

    return documents
