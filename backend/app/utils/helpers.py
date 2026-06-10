"""Miscellaneous helper utilities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


def slugify(text: str, max_length: int = 80) -> str:
    """Convert arbitrary text to a URL-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_length]


def truncate(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Truncate text to max_chars characters."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def flatten(nested: list[list[Any]]) -> list[Any]:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in sublist]


def sha256_hex(data: bytes, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]
