"""GET /health — liveness and readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline
from app.config import get_settings
from app.schemas.response import HealthResponse
from app.services.rag_pipeline import GraphRAGPipeline

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the health status of the API along with live statistics for "
        "the vector store (total indexed chunks), knowledge graph (entity and "
        "relation counts), and the configured LLM provider and model."
    ),
    tags=["Health"],
)
def health(pipeline: GraphRAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        version=s.app_version,
        llm_provider=s.llm_provider,
        llm_model=s.resolved_llm_model,
        vector_store=s.vector_store_provider,
        vector_store_chunks=pipeline._vs.count(),
        graph_provider=s.graph_provider,
        graph_enabled=s.graph_enabled,
        graph_entities=pipeline._gs.entity_count(),
        graph_relations=pipeline._gs.relation_count(),
    )
