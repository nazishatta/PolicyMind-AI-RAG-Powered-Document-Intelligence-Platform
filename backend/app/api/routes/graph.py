"""GET /api/v1/graph — knowledge graph inspection endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.dependencies import get_pipeline
from app.services.graph_service import GraphEdge, GraphNode
from app.services.rag_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class GraphNeighboursResponse(BaseModel):
    entity: str
    neighbours: list[GraphNode]
    edges: list[GraphEdge]
    total_entities: int
    total_relations: int


class GraphStatsResponse(BaseModel):
    total_entities: int
    total_relations: int
    graph_provider: str
    graph_enabled: bool


@router.get(
    "/graph/stats",
    response_model=GraphStatsResponse,
    summary="Knowledge graph statistics",
)
def graph_stats(pipeline: GraphRAGPipeline = Depends(get_pipeline)) -> GraphStatsResponse:
    from app.config import get_settings

    s = get_settings()
    return GraphStatsResponse(
        total_entities=pipeline._gs.entity_count(),
        total_relations=pipeline._gs.relation_count(),
        graph_provider=s.graph_provider,
        graph_enabled=s.graph_enabled,
    )


@router.get(
    "/graph/neighbours",
    response_model=GraphNeighboursResponse,
    summary="Retrieve graph neighbours for an entity",
    description=(
        "Returns entities within `depth` hops of the named entity "
        "along with all direct edges.  Useful for inspecting the "
        "knowledge graph interactively."
    ),
)
def graph_neighbours(
    entity: str = Query(..., description="Entity name to look up (case-sensitive)."),
    depth: int = Query(1, ge=1, le=3, description="Traversal depth (1–3)."),
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> GraphNeighboursResponse:
    neighbours = pipeline._gs.get_neighbours(entity, depth=depth)
    edges = pipeline._gs.get_edges(entity)
    return GraphNeighboursResponse(
        entity=entity,
        neighbours=neighbours,
        edges=edges,
        total_entities=pipeline._gs.entity_count(),
        total_relations=pipeline._gs.relation_count(),
    )
