"""Pydantic models for document ingestion."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class IngestRequest(BaseModel):
    """Payload for POST /api/v1/ingest.

    Exactly one of ``url`` or ``text`` must be provided.

    **Note:** When supplying a ``url``, the URL must be publicly accessible
    without authentication. Users are responsible for ensuring they have
    the right to ingest the referenced document and for complying with
    the source's licensing terms.
    """

    url: Optional[HttpUrl] = Field(
        default=None,
        description=(
            "Publicly accessible URL of a PDF or plain-text policy document. "
            "Must not require authentication. "
            "Example public sources: EUR-Lex, World Bank Open Knowledge Repository, "
            "UN document portal, government open-data portals."
        ),
        examples=["https://example.org/docs/climate_action_plan_2030.pdf"],
    )
    text: Optional[str] = Field(
        default=None,
        description="Raw document text (max 500 000 characters, ≈ 150–200 pages).",
        max_length=500_000,
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable document title. Auto-detected from URL or first line if omitted.",
        max_length=300,
    )
    source_label: Optional[str] = Field(
        default=None,
        description="Short label for the originating organisation or programme.",
        max_length=100,
        examples=["UNDP", "World Bank", "EU Commission", "UK FCDO"],
    )
    max_file_size_mb: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "Per-request override for the maximum download size in MB. "
            "Defaults to MAX_DOCUMENT_SIZE_MB from server configuration (50 MB). "
            "Applies to URL-based ingestion only."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "IngestRequest":
        if not self.url and not self.text:
            raise ValueError(
                "Provide either 'url' (a publicly accessible document URL) "
                "or 'text' (raw document content), not neither."
            )
        if self.url and self.text:
            raise ValueError(
                "Provide either 'url' or 'text', not both."
            )
        return self


class IngestResponse(BaseModel):
    """Result of a successful ingestion request."""

    doc_id: str = Field(
        description=(
            "Stable 16-character hex identifier for the ingested document. "
            "Derived from the SHA-256 of the raw content bytes. "
            "Re-ingesting the same document produces the same doc_id."
        )
    )
    title: str
    source_url: Optional[str] = Field(
        default=None,
        description="Origin URL, if the document was fetched remotely.",
    )
    page_count: int
    chunk_count: int
    entities_extracted: int
    graph_nodes_added: int
    processing_status: str = "completed"
    message: str = "Document ingested successfully."


class DocumentSummary(BaseModel):
    """Lightweight record returned by the document list endpoint."""

    doc_id: str
    doc_title: str
    source_url: Optional[str] = None
    chunk_count: int


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class DocumentDetail(BaseModel):
    """Full metadata record returned by GET /documents/{doc_id}."""

    doc_id: str
    doc_title: str
    source_url: Optional[str] = None
    source_label: Optional[str] = None
    chunk_count: int
    page_count: int
    sections: list[str] = Field(
        default_factory=list,
        description="Unique section headings detected during chunking.",
    )
    status: str = "indexed"


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str
