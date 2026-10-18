from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from src import config

APP_VERSION = "1.0.0"


def _split(value: str) -> Tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    history_dir: Path
    kb_upload_dir: Path
    eval_dir: Path
    static_dir: Path
    cors_origins: Tuple[str, ...]
    persistence: str  # "ephemeral" | "persistent"
    max_kb_upload_mb: int = 20
    max_history: int = 500

    @classmethod
    def from_env(cls) -> "Settings":
        root = config.PROJECT_ROOT
        persistence = os.environ.get("VERITYAI_PERSISTENCE", "ephemeral").lower()
        if persistence not in ("ephemeral", "persistent"):
            persistence = "ephemeral"
        return cls(
            history_dir=Path(os.environ.get("VERITYAI_HISTORY_DIR", root / "data" / "history")),
            kb_upload_dir=Path(os.environ.get("VERITYAI_KB_UPLOAD_DIR", root / "data" / "kb_uploads")),
            eval_dir=Path(os.environ.get("VERITYAI_EVAL_DIR", root / "evaluation")),
            static_dir=Path(os.environ.get("VERITYAI_STATIC_DIR", root / "product" / "static")),
            # Same-origin deployments need no CORS. Set for a separately hosted frontend,
            # e.g. VERITYAI_CORS_ORIGINS=https://my-app.web.app,http://localhost:5173
            cors_origins=_split(os.environ.get("VERITYAI_CORS_ORIGINS", "")),
            persistence=persistence,
        )

    @property
    def persistence_note(self) -> str:
        if self.persistence == "persistent":
            return "Reports and uploaded knowledge-base documents are stored on this server's persistent disk."
        return (
            "Reports and uploaded knowledge-base documents are stored on this server's local disk, "
            "which can be erased whenever the server restarts or redeploys. "
            "Download any report you need to keep."
        )
