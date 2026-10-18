"""Knowledge-base service: owns the vector store and hybrid retriever, and
implements document upload, duplicate detection, removal and re-indexing.

Seed documents ship in knowledge_base/ (read-only). Uploaded documents live in
the configurable upload directory and are the only removable ones."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api.errors import ApiError
from api.schemas import KBDocument
from api.settings import Settings
from src import config
from src.rag.chunking import Chunk, chunk_document
from src.rag.ingestion import load_document
from src.rag.retrieval import EvidencePassage, HybridRetriever, RetrievalDiagnostics
from src.rag.vector_store import VectorStore

log = logging.getLogger("verityai.kb")
_DOC_EXTS = (".pdf", ".txt", ".md")
_NAME_CLEAN = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class KnowledgeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._store: Optional[VectorStore] = None
        self._retriever: Optional[HybridRetriever] = None
        self.ready = False

    # ---- lifecycle ---------------------------------------------------------
    def _seed_files(self) -> List[Path]:
        root = config.KNOWLEDGE_BASE_DIR
        if not root.exists():
            return []
        return sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in _DOC_EXTS and p.name.lower() != "readme.md"
        )

    def _upload_files(self) -> List[Path]:
        root = self.settings.kb_upload_dir
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _DOC_EXTS)

    def _chunks_for(self, paths: List[Path]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for path in paths:
            try:
                chunks.extend(chunk_document(load_document(path)))
            except ValueError as exc:
                log.warning("Skipping unreadable document %s: %s", path.name, exc)
        return chunks

    def _ensure(self) -> Tuple[VectorStore, HybridRetriever]:
        with self._lock:
            if self._store is None:
                store = VectorStore()
                if store.count() == 0:
                    store.add_chunks(self._chunks_for(self._seed_files() + self._upload_files()))
                elif store.stale():
                    log.info("Index was built by an older chunker; rebuilding")
                    store.rebuild(self._chunks_for(self._seed_files() + self._upload_files()))
                self._store = store
                self._retriever = HybridRetriever(store)
            self.ready = True
            return self._store, self._retriever  # type: ignore[return-value]

    def warm(self) -> None:
        """Build/open the index ahead of the first request (run in a background thread)."""
        try:
            self._ensure()
        except Exception:  # noqa: BLE001 - surfaced through /health, not fatal at startup
            log.exception("Knowledge-base warm-up failed")

    # ---- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> Tuple[List[EvidencePassage], RetrievalDiagnostics]:
        _, retriever = self._ensure()
        with self._lock:
            return retriever.retrieve_with_diagnostics(query, top_k=top_k)

    # ---- documents ----------------------------------------------------------
    def list_documents(self) -> Tuple[List[KBDocument], int]:
        store, _ = self._ensure()
        with self._lock:
            indexed = store.list_sources()
            total = store.count()
        entries: List[Tuple[str, Path]] = [("seed", p) for p in self._seed_files()] + [
            ("upload", p) for p in self._upload_files()
        ]
        first_by_hash: Dict[str, str] = {}
        docs: List[KBDocument] = []
        for origin, path in entries:
            digest = _sha256(path)
            info = indexed.get(path.name)
            dup = first_by_hash.get(digest)
            first_by_hash.setdefault(digest, path.name)
            sidecar = path.with_suffix(path.suffix + ".url")
            url = None
            if sidecar.exists():
                try:
                    url = sidecar.read_text(encoding="utf-8").strip() or None
                except OSError:
                    url = None
            docs.append(KBDocument(
                source_file=path.name,
                title=(info or {}).get("title") or path.stem,
                origin=origin,  # type: ignore[arg-type]
                removable=origin == "upload",
                chunks=(info or {}).get("chunks", 0),
                indexed=info is not None,
                sha256=digest,
                size_bytes=path.stat().st_size,
                source_url=(info or {}).get("source_url") or url,
                duplicate_of=dup,
            ))
        return docs, total

    @staticmethod
    def _safe_name(filename: str) -> str:
        name = _NAME_CLEAN.sub("_", Path(filename or "").name).strip("._")
        return name[:120]

    def add_document(self, filename: str, data: bytes, source_url: Optional[str]) -> Tuple[str, str, int, Optional[KBDocument]]:
        name = self._safe_name(filename)
        suffix = Path(name).suffix.lower()
        if not name or suffix not in _DOC_EXTS:
            raise ApiError(422, "unsupported_type", f"Unsupported document type. Allowed: {', '.join(_DOC_EXTS)}.")
        if not data:
            raise ApiError(422, "empty_file", "The file is empty.")
        if len(data) > self.settings.max_kb_upload_mb * 1024 * 1024:
            raise ApiError(413, "file_too_large", f"Documents are limited to {self.settings.max_kb_upload_mb} MB.")
        if suffix == ".pdf" and not data.startswith(b"%PDF"):
            raise ApiError(422, "invalid_file", "The file does not look like a PDF (signature check failed).")
        if suffix != ".pdf" and b"\x00" in data[:4096]:
            raise ApiError(422, "invalid_file", "The file looks binary, not text.")
        if source_url:
            if not re.match(r"^https?://[^\s]{3,500}$", source_url):
                raise ApiError(422, "invalid_url", "The source URL must start with http:// or https://.")

        digest = hashlib.sha256(data).hexdigest()
        docs, _ = self.list_documents()
        for d in docs:
            if d.sha256 == digest:
                return "duplicate", f"Identical content is already indexed as '{d.source_file}'. Nothing was added.", 0, d
        if any(d.source_file == name for d in docs):
            raise ApiError(409, "name_conflict", f"A different document named '{name}' already exists. Rename the file and retry.")

        upload_dir = self.settings.kb_upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / name
        sidecar = path.with_suffix(path.suffix + ".url")
        path.write_bytes(data)
        if source_url:
            sidecar.write_text(source_url, encoding="utf-8")
        try:
            chunks = chunk_document(load_document(path))
        except ValueError as exc:
            self._unlink(path, sidecar)
            raise ApiError(422, "unreadable_document", str(exc)) from exc
        if not chunks:
            self._unlink(path, sidecar)
            raise ApiError(422, "no_text", "No extractable text was found (scanned PDFs need OCR, which is not supported).")

        store, retriever = self._ensure()
        with self._lock:
            added = store.add_chunks(chunks)
            retriever.refresh()
        docs, _ = self.list_documents()
        doc = next((d for d in docs if d.source_file == name), None)
        return "indexed", f"Indexed '{name}' as {added} passages.", added, doc

    @staticmethod
    def _unlink(*paths: Path) -> None:
        for p in paths:
            try:
                p.unlink()
            except OSError:
                pass

    def remove_document(self, source_file: str) -> int:
        name = self._safe_name(source_file)
        path = self.settings.kb_upload_dir / name
        if name != source_file or not path.is_file():
            if (config.KNOWLEDGE_BASE_DIR / source_file).is_file():
                raise ApiError(403, "read_only", "Built-in documents are read-only and cannot be removed here.")
            raise ApiError(404, "not_found", f"No uploaded document named '{source_file}'.")
        store, retriever = self._ensure()
        with self._lock:
            removed = store.delete_source(name)
            retriever.refresh()
        self._unlink(path, path.with_suffix(path.suffix + ".url"))
        return removed

    def reindex(self) -> int:
        store, retriever = self._ensure()
        with self._lock:
            chunks = self._chunks_for(self._seed_files() + self._upload_files())
            store.rebuild(chunks)
            retriever.refresh()
            return store.count()

    def counts(self) -> Tuple[int, int]:
        """(documents, chunks) without forcing a rebuild if not ready."""
        if self._store is None:
            return len(self._seed_files()) + len(self._upload_files()), 0
        with self._lock:
            return len(self._store.list_sources()), self._store.count()
