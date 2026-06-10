"""Recursive semantic chunker for policy document text.

Splits a PolicyDocument into overlapping TextChunk objects that carry
full citation provenance: doc_id, page number, section heading, source URL,
and character offsets within the page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.document_loader import PolicyDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Matches common policy-document section markers:
#   Roman numerals:  "III. Governance"
#   Numbered:        "1.2.3 Targets"
#   ALL-CAPS labels: "EXECUTIVE SUMMARY"
#   Article/Section: "Article 12 — Compliance"
#   Lettered:        "A. Background"
_HEADING_RE = re.compile(
    r"^("
    r"(?:Article|Section|Annex|Schedule|Appendix|Chapter|Part)\s+[\dIVXA-Z][\w\s\-–—:]{0,80}"
    r"|[IVX]{1,5}\.\s+.{3,80}"
    r"|\d{1,3}(?:\.\d{1,3}){0,3}\.?\s+[A-Z].{2,79}"
    r"|[A-Z]{3}[A-Z\s]{0,60}:?"
    r"|[A-Z]\.\s+[A-Z].{2,79}"
    r")$",
    re.MULTILINE,
)


def _detect_heading(text: str) -> Optional[str]:
    """Return the first section heading found in a text chunk, or None."""
    for line in text.splitlines()[:5]:       # headings appear near the top of a chunk
        line = line.strip()
        if _HEADING_RE.match(line) and len(line) <= 120:
            return line
    return None


# ---------------------------------------------------------------------------
# Core split algorithm
# ---------------------------------------------------------------------------

def _split(text: str, chunk_size: int, overlap: int, seps: list[str]) -> list[str]:
    """Recursively split text along separator hierarchy until chunks fit chunk_size."""
    if not text.strip():
        return []

    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                tail = current[-overlap:] if overlap else ""
                current = (tail + sep if tail else "") + part

        if current.strip():
            chunks.append(current.strip())

        # Recursively split any chunk that still exceeds chunk_size
        remaining = seps[seps.index(sep) + 1:]
        result: list[str] = []
        for c in chunks:
            if len(c) > chunk_size and remaining:
                result.extend(_split(c, chunk_size, overlap, remaining))
            else:
                result.append(c)
        return result

    # Hard-split fallback: no separator matched; split at chunk_size boundary
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


# ---------------------------------------------------------------------------
# TextChunk data model
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """A single passage derived from a policy document.

    All citation-relevant fields are first-class attributes so the
    citation engine and vector-store metadata can reference them directly
    without digging into the metadata dict.

    Attributes:
        chunk_id:        Unique identifier: ``{doc_id}_p{page:04d}_c{idx:06d}``
        doc_id:          Stable document identifier (SHA-256 prefix).
        text:            The chunk text content.
        page_number:     1-based page number within the source document.
        chunk_index:     Global sequential index across all pages.
        char_start:      Start character offset within the page text.
        char_end:        End character offset within the page text.
        source_url:      Origin URL of the document (None for text payloads).
        section_heading: Nearest section heading detected within the chunk.
        metadata:        Supplementary key-value pairs for the vector store.
    """

    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    source_url: Optional[str] = None
    section_heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    def citation_label(self) -> str:
        """Short human-readable citation string."""
        parts = [self.metadata.get("doc_title", self.doc_id)]
        if self.section_heading:
            parts.append(self.section_heading)
        parts.append(f"p. {self.page_number}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def chunk_document(
    doc: PolicyDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chars: int = 50,
) -> list[TextChunk]:
    """Split a :class:`PolicyDocument` into overlapping :class:`TextChunk` objects.

    Args:
        doc:           The loaded policy document.
        chunk_size:    Target character count per chunk.
        chunk_overlap: Characters of context carried into the next chunk.
        min_chars:     Discard chunks shorter than this threshold.

    Returns:
        List of :class:`TextChunk` objects ordered by page, then position.

    Raises:
        ValueError: If chunk_size ≤ chunk_overlap (would produce infinite loop).
    """
    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than chunk_overlap ({chunk_overlap})."
        )

    chunks: list[TextChunk] = []
    global_idx = 0

    for page_num, page_text in doc.pages:
        if not page_text.strip():
            logger.debug("Skipping empty page %d in '%s'", page_num, doc.title)
            continue

        raw_chunks = _split(page_text, chunk_size, chunk_overlap, _DEFAULT_SEPARATORS)
        cursor = 0

        for raw in raw_chunks:
            if len(raw) < min_chars:
                continue

            # Locate char offset within the page (search forward from cursor)
            start = page_text.find(raw, max(0, cursor - 20))
            if start == -1:
                start = cursor
            end = start + len(raw)
            cursor = end

            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}_p{page_num:04d}_c{global_idx:06d}",
                    doc_id=doc.doc_id,
                    text=raw,
                    page_number=page_num,
                    chunk_index=global_idx,
                    char_start=start,
                    char_end=end,
                    source_url=doc.source_url,
                    section_heading=_detect_heading(raw),
                    metadata={
                        "doc_title": doc.title,
                        "source_url": doc.source_url or "",
                        "source_label": doc.source_label or "",
                        "page_number": page_num,
                        "section_heading": _detect_heading(raw) or "",
                        "doc_id": doc.doc_id,
                    },
                )
            )
            global_idx += 1

    logger.info(
        "Chunked '%s' -> %d chunks (chunk_size=%d, overlap=%d, pages=%d)",
        doc.title, len(chunks), chunk_size, chunk_overlap, doc.page_count,
    )
    return chunks
