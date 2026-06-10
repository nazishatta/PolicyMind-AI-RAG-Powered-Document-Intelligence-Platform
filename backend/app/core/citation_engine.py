"""Citation tracking and formatting.

The engine records which chunks contributed to an answer and produces
structured, human-readable citations in APA-style and JSON formats.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from app.schemas.response import Citation


@dataclass
class CitationRecord:
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    section_heading: Optional[str]
    excerpt: str
    relevance_score: float


class CitationEngine:
    """Accumulates retrieved chunks and exports them as citations."""

    def __init__(self) -> None:
        self._records: list[CitationRecord] = []

    def record(
        self,
        chunk_id: str,
        doc_id: str,
        doc_title: str,
        page_number: int,
        excerpt: str,
        relevance_score: float,
        section_heading: Optional[str] = None,
    ) -> None:
        self._records.append(
            CitationRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                page_number=page_number,
                section_heading=section_heading,
                excerpt=excerpt[:300],
                relevance_score=round(max(0.0, min(1.0, relevance_score)), 4),
            )
        )

    def to_schema(self) -> list[Citation]:
        """Return citations as :class:`Citation` Pydantic objects."""
        return [
            Citation(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                page_number=r.page_number,
                section_heading=r.section_heading,
                excerpt=r.excerpt,
                relevance_score=r.relevance_score,
            )
            for r in self._records
        ]

    def to_apa(self) -> list[str]:
        """Format citations in a simplified APA style."""
        apa = []
        today = date.today().strftime("%B %d, %Y")
        for r in self._records:
            heading = f", {r.section_heading}" if r.section_heading else ""
            apa.append(
                f"{r.doc_title}{heading}, p. {r.page_number}. "
                f"Retrieved {today}. [chunk {r.chunk_id}]"
            )
        return apa

    def clear(self) -> None:
        self._records.clear()
