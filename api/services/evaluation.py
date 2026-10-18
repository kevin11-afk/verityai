"""Reads evaluation artifacts produced by scripts/export_evaluation.py.
Nothing is computed or defaulted here: absent artifacts mean "not evaluated"."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_evaluation(eval_dir: Path) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        "detector": _load(eval_dir / "detector_eval.json"),
        "rag": _load(eval_dir / "rag_eval.json"),
    }
