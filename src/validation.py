"""
Input validation and safe temp-file handling for user uploads.

Everything the user uploads is treated as untrusted:
  - extension AND file-signature (magic bytes) are checked, not just the
    filename the browser reports.
  - size and dimension/duration limits are enforced before any heavy
    processing (never trust Content-Length alone).
  - temp files are written with random names inside a fixed directory —
    the user-provided filename is never used to build a path, which is
    what prevents path traversal.
"""
from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from src import config

# --------------------------------------------------------------------------
# File signatures ("magic bytes") for the formats we claim to support.
# Checked in addition to the extension, so a renamed .exe cannot masquerade
# as a .png.
# --------------------------------------------------------------------------
_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",  # JPEG/JPG
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # followed by "WEBP" at offset 8; checked below
}

_VIDEO_SIGNATURES_FTYP_OFFSET = 4  # MP4/MOV both use an ISO-BMFF "ftyp" box


@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None
    detected_mime: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _ext_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_image_bytes(data: bytes, filename: str) -> ValidationResult:
    """Validate an uploaded image before it ever reaches the detector."""
    ext = _ext_of(filename)
    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        return ValidationResult(False, f"Unsupported file extension '{ext}'. "
                                        f"Allowed: {sorted(config.ALLOWED_IMAGE_EXTENSIONS)}")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_SIZE_MB:
        return ValidationResult(False, f"File is {size_mb:.1f}MB, exceeds the "
                                        f"{config.MAX_IMAGE_SIZE_MB}MB limit.")
    if len(data) == 0:
        return ValidationResult(False, "File is empty.")

    detected_mime = _sniff_image_mime(data)
    if detected_mime is None:
        return ValidationResult(False, "File content does not match a supported "
                                        "image format (signature check failed). "
                                        "The file may be corrupted or mislabeled.")

    # Decode with Pillow to get real dimensions and catch corrupt/decompression-bomb images.
    try:
        Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_DIMENSION_PX * config.MAX_IMAGE_DIMENSION_PX
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # cheap structural check
        with Image.open(io.BytesIO(data)) as img:  # verify() invalidates the handle; reopen
            width, height = img.size
    except Exception as exc:  # noqa: BLE001 - any decode failure is a validation failure
        return ValidationResult(False, f"Image could not be decoded safely: {exc}")

    if width < config.MIN_IMAGE_DIMENSION_PX or height < config.MIN_IMAGE_DIMENSION_PX:
        return ValidationResult(False, f"Image is {width}x{height}px, smaller than the "
                                        f"{config.MIN_IMAGE_DIMENSION_PX}px minimum.")
    if width > config.MAX_IMAGE_DIMENSION_PX or height > config.MAX_IMAGE_DIMENSION_PX:
        return ValidationResult(False, f"Image is {width}x{height}px, larger than the "
                                        f"{config.MAX_IMAGE_DIMENSION_PX}px maximum.")

    return ValidationResult(True, detected_mime=detected_mime, width=width, height=height)


def _sniff_image_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_video_bytes(data: bytes, filename: str) -> ValidationResult:
    """Validate an uploaded video's container/size before frame extraction.
    Duration is checked later with OpenCV once the file is on disk (container
    metadata is unreliable to parse safely from raw bytes alone).
    """
    ext = _ext_of(filename)
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        return ValidationResult(False, f"Unsupported file extension '{ext}'. "
                                        f"Allowed: {sorted(config.ALLOWED_VIDEO_EXTENSIONS)}")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_VIDEO_SIZE_MB:
        return ValidationResult(False, f"File is {size_mb:.1f}MB, exceeds the "
                                        f"{config.MAX_VIDEO_SIZE_MB}MB limit.")
    if len(data) == 0:
        return ValidationResult(False, "File is empty.")

    # ISO-BMFF signature check: bytes 4-8 should spell "ftyp" for MP4/MOV.
    if len(data) < 12 or data[4:8] != b"ftyp":
        return ValidationResult(False, "File content does not match a supported "
                                        "MP4/MOV container (signature check failed).")

    return ValidationResult(True, detected_mime="video/mp4" if ext == ".mp4" else "video/quicktime")


def safe_temp_path(suffix: str) -> Path:
    """Build a temp-file path with a random name inside the fixed upload
    directory. The caller's original filename is never used here, which is
    what makes path traversal impossible regardless of what the browser sent.
    """
    config.TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix if suffix in (config.ALLOWED_IMAGE_EXTENSIONS | config.ALLOWED_VIDEO_EXTENSIONS) else ""
    return config.TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}{safe_suffix}"


def cleanup_temp_file(path: Path) -> None:
    """Best-effort deletion of a temp file. Never raises."""
    try:
        if path.exists():
            os.remove(path)
    except OSError:
        pass
