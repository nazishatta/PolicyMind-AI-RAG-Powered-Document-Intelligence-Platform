"""POST /api/v1/query — question answering endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_pipeline
from app.schemas.query import QueryRequest
from app.schemas.response import AnswerResponse
from app.services.rag_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=AnswerResponse,
    summary="Ask a question over the policy corpus",
    description=(
        "Retrieves semantically relevant passages (and optional graph evidence), "
        "then generates a citation-backed answer using the configured LLM provider."
    ),
)
async def query_documents(
    body: QueryRequest,
    pipeline: GraphRAGPipeline = Depends(get_pipeline),
) -> AnswerResponse:
    try:
        return await pipeline.query(
            question=body.question,
            doc_id=body.doc_id,
            top_k=body.top_k,
            include_graph=body.include_graph_evidence,
            graph_depth=body.graph_depth,
        )
    except Exception as exc:
        logger.error("Query pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )
