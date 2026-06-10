"""Pydantic models for query requests."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /api/v1/query."""

    question: str = Field(
        description="Natural language question over the policy corpus.",
        min_length=5,
        max_length=2000,
        examples=["What emission reduction targets are committed to by 2030?"],
    )
    doc_id: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to a single document. "
            "If omitted, the question is answered across all ingested documents."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the default number of retrieved chunks.",
    )
    include_graph_evidence: bool = Field(
        default=True,
        description="Include knowledge-graph evidence in the response when available.",
    )
    graph_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description=(
            "Graph neighbourhood traversal depth for hybrid retrieval. "
            "1 (default) — direct edges only. "
            "2 — also collects edges from 1-hop neighbours; indirect evidence "
            "is included with confidence discounted by 0.8 to signal it is "
            "less certain than directly attached triples. "
            "3 — two layers of indirect evidence. "
            "Higher values improve recall for multi-hop policy relationships "
            "(e.g. agency → programme → deadline chains) at some added latency. "
            "Omit to use the server default (depth 1)."
        ),
    )
