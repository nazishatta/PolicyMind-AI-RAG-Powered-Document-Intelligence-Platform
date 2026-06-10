"""Ingestion endpoints: POST /ingest, GET /documents, DELETE /documents/{doc_id}."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.dependencies import get_pipeline
from app.core.connectors import connector_from_request
from app.core.document_loader import (
    DocumentEmptyError,
    DocumentFetchError,
    DocumentParseError,
    DocumentSizeError,
    IngestionError,
    UnsupportedContentTypeError,
)
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentDetail,
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    ListDocumentsResponse,
)
from app.services.rag_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Error → HTTP status mapping
# ---------------------------------------------------------------------------

def _ingestion_error_to_http(exc: IngestionError) -> HTTPException:
    """Map typed IngestionError subclasses to appropriate HTTP status codes."""
    if isinstance(exc, DocumentSizeError):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        )
    if isinstance(exc, UnsupportedContentTypeError):
        return HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )
    if isinstance(exc, DocumentFetchError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )
    if isinstance(exc, (DocumentEmptyError, DocumentParseError)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    # Generic IngestionError fallback
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=str(exc),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a policy document",
    description=(
        "Accepts a **publicly accessible** document URL (PDF or plain text) "
        "or a raw text payload. The document is chunked, embedded, indexed "
        "in the vector store, and entity/relation data is written to the "
        "knowledge graph.\n\n"
        "**Users are responsible for supplying valid public URLs and for "
        "complying with the source document's licensing terms.** "
        "This endpoint does not store raw document bytes — only "
        "processed metadata, chunks, and embeddings are persisted."
    ),
)
async def ingest_document(
    body: IngestRequest,
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    from app.config import get_settings

    settings = get_settings()
    max_bytes = (
        body.max_file_size_mb * 1024 * 1024
        if body.max_file_size_mb
        else settings.max_document_size_bytes
    )

    # Build connector
    try:
        connector = connector_from_request(
            url=str(body.url) if body.url else None,
            text=body.text,
            title=body.title,
            source_label=body.source_label,
            max_bytes=max_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))

    # Fetch document
    try:
        doc = await connector.fetch()
    except IngestionError as exc:
        logger.warning("Ingestion fetch error [%s]: %s", type(exc).__name__, exc)
        raise _ingestion_error_to_http(exc)
    except Exception as exc:
        logger.error("Unexpected fetch error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected error loading document: {exc}",
        )

    # Run pipeline
    try:
        result = pipeline.ingest(doc)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
    except Exception as exc:
        logger.error("Pipeline ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline error: {exc}",
        )

    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        source_url=doc.source_url,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        entities_extracted=result.entities_extracted,
        graph_nodes_added=result.graph_nodes_added,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------

@router.get(
    "/documents",
    response_model=ListDocumentsResponse,
    summary="List ingested documents",
    description="Returns a summary of all documents currently indexed in the vector store.",
)
def list_documents(
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> ListDocumentsResponse:
    docs = pipeline.list_documents()
    summaries = [
        DocumentSummary(
            doc_id=d["doc_id"],
            doc_title=d.get("doc_title", ""),
            source_url=d.get("source_url") or None,
            chunk_count=d.get("chunk_count", 0),
        )
        for d in docs
    ]
    return ListDocumentsResponse(documents=summaries, total=len(summaries))


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{doc_id}
# ---------------------------------------------------------------------------

@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDetail,
    summary="Get document metadata",
    description=(
        "Returns detailed metadata for a single ingested document: title, "
        "source URL, chunk count, page count, and detected section headings."
    ),
)
def get_document(
    doc_id: str = Path(..., description="Document ID returned by the ingest endpoint."),
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> DocumentDetail:
    detail = pipeline.get_document(doc_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return DocumentDetail(**detail)


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{doc_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Remove an ingested document",
    description=(
        "Deletes all chunks and embeddings for the specified document from the "
        "vector store. Graph entities extracted from this document are not removed "
        "(they may be shared across multiple documents)."
    ),
)
def delete_document(
    doc_id: str = Path(..., description="Document ID returned by the ingest endpoint."),
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> DeleteDocumentResponse:
    deleted = pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in the vector store.",
        )
    return DeleteDocumentResponse(
        doc_id=doc_id,
        deleted=True,
        message=f"Document '{doc_id}' and all its chunks have been removed.",
    )
