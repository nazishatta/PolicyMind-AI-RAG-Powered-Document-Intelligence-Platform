"""GET / — project index and endpoint directory."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline
from app.config import get_settings
from app.schemas.response import RootResponse
from app.services.rag_pipeline import GraphRAGPipeline

router = APIRouter()

_ENDPOINT_MAP = {
    "root":               "GET  /",
    "health":             "GET  /health",
    "docs":               "GET  /docs  (Swagger UI)",
    "redoc":              "GET  /redoc",
    "ingest":             "POST /api/v1/ingest",
    "query":              "POST /api/v1/query",
    "list_documents":     "GET  /api/v1/documents",
    "document_detail":    "GET  /api/v1/documents/{doc_id}",
    "delete_document":    "DELETE /api/v1/documents/{doc_id}",
    "graph_stats":        "GET  /api/v1/graph/stats",
    "graph_neighbours":   "GET  /api/v1/graph/neighbours?entity=<name>&depth=1",
}


@router.get(
    "/",
    response_model=RootResponse,
    summary="Project index",
    description=(
        "Returns project metadata, the configured provider stack, "
        "and a directory of all available API endpoints."
    ),
    tags=["Root"],
)
def root(pipeline: GraphRAGPipeline = Depends(get_pipeline)) -> RootResponse:
    s = get_settings()
    return RootResponse(
        name=s.app_name,
        version=s.app_version,
        description=(
            "GraphRAG-powered policy document intelligence platform. "
            "Ingest public policy documents, extract knowledge graphs, "
            "and ask citation-backed questions over complex policy corpora."
        ),
        status="running",
        docs_url="/docs",
        redoc_url="/redoc",
        endpoints=_ENDPOINT_MAP,
        providers={
            "llm": s.llm_provider,
            "llm_model": s.resolved_llm_model,
            "vector_store": s.vector_store_provider,
            "graph": s.graph_provider,
            "embeddings": s.embedding_model,
        },
    )
