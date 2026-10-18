"""Report history as one JSON file per report. Only report metadata is kept:
uploaded media is deleted right after analysis and never stored."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

ID_RE = re.compile(r"^[0-9a-f]{32}$")


class HistoryStore:
    def __init__(self, directory: Path, max_items: int = 500):
        self.directory = directory
        self.max_items = max_items
        self._lock = threading.Lock()

    def _path(self, report_id: str) -> Optional[Path]:
        # The regex is what makes path traversal impossible: ids are never joined raw.
        return self.directory / f"{report_id}.json" if ID_RE.match(report_id) else None

    def save(self, report: Dict) -> None:
        path = self._path(report["id"])
        if path is None:
            raise ValueError("invalid report id")
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
            self._prune()

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[self.max_items:]:
            try:
                old.unlink()
            except OSError:
                pass

    def get(self, report_id: str) -> Optional[Dict]:
        path = self._path(report_id)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def delete(self, report_id: str) -> bool:
        path = self._path(report_id)
        if path is None or not path.exists():
            return False
        with self._lock:
            try:
                path.unlink()
                return True
            except OSError:
                return False

    def list(self) -> List[Dict]:
        if not self.directory.exists():
            return []
        out = []
        for p in self.directory.glob("*.json"):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return out
