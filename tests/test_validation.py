"""Tests for src/validation.py: file-type, size, dimension, and signature checks."""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from src import config, validation


def _make_jpeg_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 180, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


class TestImageValidation:
    def test_valid_jpeg_passes(self):
        result = validation.validate_image_bytes(_make_jpeg_bytes(), "photo.jpg")
        assert result.ok
        assert result.detected_mime == "image/jpeg"
        assert result.width == 64 and result.height == 64

    def test_valid_png_passes(self):
        result = validation.validate_image_bytes(_make_png_bytes(), "photo.png")
        assert result.ok
        assert result.detected_mime == "image/png"

    def test_rejects_unsupported_extension(self):
        result = validation.validate_image_bytes(_make_jpeg_bytes(), "photo.gif")
        assert not result.ok
        assert "extension" in result.reason.lower()

    def test_rejects_empty_file(self):
        result = validation.validate_image_bytes(b"", "photo.jpg")
        assert not result.ok

    def test_rejects_oversized_file(self):
        big = b"\xff\xd8\xff" + b"0" * (config.MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
        result = validation.validate_image_bytes(big, "photo.jpg")
        assert not result.ok
        assert "exceeds" in result.reason.lower()

    def test_rejects_renamed_non_image_file(self):
        """A .txt file renamed to .jpg must fail the signature check, not just
        be accepted because the extension looks right."""
        fake = b"this is definitely not an image, just text pretending to be one"
        result = validation.validate_image_bytes(fake, "fake.jpg")
        assert not result.ok
        assert "signature" in result.reason.lower() or "decoded" in result.reason.lower()

    def test_rejects_undersized_dimensions(self):
        tiny = _make_jpeg_bytes(width=4, height=4)
        result = validation.validate_image_bytes(tiny, "tiny.jpg")
        assert not result.ok
        assert "smaller" in result.reason.lower()

    def test_webp_signature_detected(self):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="WEBP")
        result = validation.validate_image_bytes(buf.getvalue(), "photo.webp")
        assert result.ok
        assert result.detected_mime == "image/webp"


class TestVideoValidation:
    def test_rejects_unsupported_extension(self):
        result = validation.validate_video_bytes(b"0" * 100, "clip.avi")
        assert not result.ok

    def test_rejects_bad_signature(self):
        fake_mp4 = b"not a real mp4 container" + b"0" * 100
        result = validation.validate_video_bytes(fake_mp4, "clip.mp4")
        assert not result.ok
        assert "signature" in result.reason.lower()

    def test_accepts_valid_ftyp_signature(self):
        # Minimal fake ISO-BMFF header: size(4) + 'ftyp' + rest
        fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 50
        result = validation.validate_video_bytes(fake_mp4, "clip.mp4")
        assert result.ok

    def test_rejects_oversized_file(self):
        fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * (config.MAX_VIDEO_SIZE_MB * 1024 * 1024 + 1)
        result = validation.validate_video_bytes(fake_mp4, "clip.mp4")
        assert not result.ok


class TestSafeTempPath:
    def test_ignores_user_filename_prevents_traversal(self):
        """The returned path must never incorporate the caller's filename —
        this is what makes path traversal via '../../etc/passwd' impossible."""
        path = validation.safe_temp_path(".jpg")
        assert path.parent == config.TMP_UPLOAD_DIR
        assert ".." not in str(path)
        assert path.suffix == ".jpg"

    def test_unknown_suffix_is_stripped(self):
        path = validation.safe_temp_path(".exe")
        assert path.suffix != ".exe"

    def test_cleanup_is_safe_on_missing_file(self):
        missing = config.TMP_UPLOAD_DIR / "does-not-exist-12345.jpg"
        validation.cleanup_temp_file(missing)  # must not raise
