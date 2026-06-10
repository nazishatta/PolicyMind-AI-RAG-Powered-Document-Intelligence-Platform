"""Pydantic models for API responses."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A traceable source reference produced by the citation engine."""

    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    section_heading: Optional[str] = None
    excerpt: str = Field(description="Verbatim excerpt (≤300 chars) supporting the answer.")
    relevance_score: float = Field(ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    """A raw passage returned from the vector store before answer generation."""

    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    section_heading: Optional[str] = None
    text: str = Field(description="Full chunk text (≤500 chars displayed).")
    relevance_score: float = Field(ge=0.0, le=1.0)


class GraphEvidence(BaseModel):
    """A knowledge-graph triple used to corroborate the answer."""

    entity: str
    relation: str
    target: str
    source_doc_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    """Full response from POST /api/v1/query."""

    query_id: str = Field(description="Unique identifier for this query session.")
    question: str
    answer: str = Field(description="LLM-generated answer grounded in retrieved passages.")

    # ── Trust layer ──────────────────────────────────────────────────────────

    answer_type: Literal["cited", "partial", "refused", "no_corpus"] = Field(
        default="cited",
        description=(
            "Classification of this response: "
            "'cited' — answer is grounded in retrieved passages; "
            "'partial' — evidence was found but is incomplete or weak (confidence < 0.35); "
            "'refused' — the LLM judged the context insufficient to answer; "
            "'no_corpus' — no documents were indexed when the query was made."
        ),
    )
    evidence_quality: Literal["strong", "moderate", "weak", "insufficient"] = Field(
        default="insufficient",
        description=(
            "Categorical evidence quality derived from confidence and passage count: "
            "'strong' (conf ≥ 0.7 and ≥ 3 passages), "
            "'moderate' (conf ≥ 0.5 and ≥ 2 passages), "
            "'weak' (conf ≥ 0.35), "
            "'insufficient' (conf < 0.35 or no passages retrieved)."
        ),
    )
    confidence_note: str = Field(
        default="",
        description=(
            "Human-readable explanation of the confidence score: number of passages, "
            "mean cosine similarity, graph-evidence contribution, and quality assessment."
        ),
    )

    # ── Sources ──────────────────────────────────────────────────────────────

    citations: list[Citation] = Field(
        description="Processed, citation-formatted source chunks that ground the answer."
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Raw retrieved passages before answer generation, with relevance scores.",
    )
    graph_evidence: list[GraphEvidence] = Field(
        description="Knowledge-graph triples that informed retrieval scoring and the answer."
    )

    # ── Confidence and caveats ───────────────────────────────────────────────

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Retrieval confidence in [0, 1]: mean top-3 chunk cosine similarity "
            "plus a small graph-evidence bonus. None when no passages were retrieved."
        ),
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Deterministic limitations for this response — mock provider in use, "
            "low retrieval confidence, empty graph, sparse passage coverage, etc."
        ),
    )

    # ── Metadata ─────────────────────────────────────────────────────────────

    latency_ms: float = Field(description="End-to-end query latency in milliseconds.")
    provider: str = Field(description="LLM provider used ('anthropic', 'openai', 'mock').")
    model: str = Field(description="LLM model identifier.")


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "ok"
    version: str
    llm_provider: str
    llm_model: str = ""
    vector_store: str
    vector_store_chunks: int = 0
    graph_provider: str
    graph_enabled: bool
    graph_entities: int = 0
    graph_relations: int = 0


class RootResponse(BaseModel):
    """Response from GET /."""

    name: str
    version: str
    description: str
    status: str
    docs_url: str
    redoc_url: str
    endpoints: dict[str, str]
    providers: dict[str, str]
