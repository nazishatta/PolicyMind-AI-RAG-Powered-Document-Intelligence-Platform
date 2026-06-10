"""GraphRAG pipeline — ingestion and query orchestration.

The pipeline is the single entry point for both document ingestion and
question answering.  It wires together the core modules and service layer,
keeping each component independently testable.

Query flow
----------
1. Embed the question.
2. HybridRetriever: vector search → entity extraction → graph expansion → score fusion.
3. CitationEngine: record retrieved chunks as traceable citations.
4. LLM: generate an answer from the fused context.
5. Confidence scoring and limitations inference.
6. Return a fully structured AnswerResponse.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.core.citation_engine import CitationEngine
from app.core.document_loader import PolicyDocument
from app.core.embeddings import EmbeddingModel
from app.core.entity_extraction import extract_entities, extract_relations
from app.core.retriever import HybridRetriever
from app.core.text_chunker import chunk_document
from app.schemas.response import AnswerResponse, GraphEvidence, RetrievedChunk
from app.services.graph_service import BaseGraphService
from app.services.llm_service import BaseLLMProvider
from app.services.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Confidence below this threshold triggers a "partial" answer type and injects
# a low-evidence preamble into the user prompt so the LLM qualifies its response.
_LOW_CONFIDENCE_THRESHOLD = 0.35

# These phrases are produced by the pipeline itself (not the LLM) so they can
# be detected programmatically for answer-type classification.
_REFUSAL_PHRASE = (
    "The available documents do not contain enough information to answer this question."
)
_PARTIAL_PHRASE = "Based on limited evidence:"

_SYSTEM_PROMPT = """\
You are a careful policy research assistant. Your task is to answer questions \
using ONLY the context passages provided below. You must follow these rules without exception:

1. CITATION-FIRST: Begin every answer by stating which document(s) and page(s) \
you are drawing from, e.g. "According to [Document Title, p.3]..." or \
"The following is based on passages from [Title] (pages 2 and 5)."

2. INLINE CITATIONS: After each factual claim, add a brief source reference in \
brackets, e.g. "(p.4, Section 2)" or "(p.7)".

3. REFUSE WHEN UNSUPPORTED: If the provided context passages do not contain \
sufficient information to answer the question, respond with exactly this phrase \
followed by a brief explanation: \
"The available documents do not contain enough information to answer this question."

4. PARTIAL ANSWERS: If the context contains relevant but incomplete information, \
begin your answer with "Based on limited evidence:" and clearly distinguish \
between what the documents say and what remains unaddressed.

5. NO SPECULATION: Do not infer, extrapolate, or add any information that is not \
explicitly stated in the provided passages. Do not use your general knowledge.

6. NO UNSUPPORTED CONCLUSIONS: Do not draw policy conclusions, make normative \
judgements, or predict outcomes beyond what the source text states.

7. AMBIGUITY DISCLOSURE: If the context passages contradict each other or leave \
the question ambiguous, say so explicitly before attempting an answer."""


@dataclass
class IngestResult:
    doc_id: str
    title: str
    page_count: int
    chunk_count: int
    entities_extracted: int
    graph_nodes_added: int


class GraphRAGPipeline:
    """Orchestrates ingestion and GraphRAG query answering."""

    def __init__(
        self,
        settings: Settings,
        vector_store: BaseVectorStore,
        graph_service: BaseGraphService,
        llm: BaseLLMProvider,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._settings = settings
        self._vs = vector_store
        self._gs = graph_service
        self._llm = llm
        self._emb = embedding_model
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            graph_service=graph_service,
            embedding_model=embedding_model,
            alpha=settings.graph_vector_alpha,
            graph_depth=1,
            max_graph_evidence=settings.top_k_graph * 3,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, doc: PolicyDocument) -> IngestResult:
        """Chunk, embed, index, and extract graph data from a document."""
        s = self._settings

        chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap)
        if not chunks:
            raise ValueError(f"Document '{doc.title}' produced no usable chunks.")

        embeddings = self._emb.embed([c.text for c in chunks])
        self._vs.upsert(chunks, embeddings)

        entity_count = 0
        node_count_before = self._gs.entity_count()
        for chunk in chunks:
            entities = extract_entities(
                chunk.text, chunk.doc_id, chunk.page_number, chunk.chunk_id
            )
            relations = extract_relations(entities, chunk.text, chunk.chunk_id, chunk.doc_id)
            for ent in entities:
                self._gs.add_entity(ent)
                entity_count += 1
            for rel in relations:
                self._gs.add_relation(rel)
        graph_nodes_added = self._gs.entity_count() - node_count_before

        logger.info(
            "Ingested '%s': %d chunks, %d entities, %d new graph nodes",
            doc.title, len(chunks), entity_count, graph_nodes_added,
        )
        return IngestResult(
            doc_id=doc.doc_id,
            title=doc.title,
            page_count=doc.page_count,
            chunk_count=len(chunks),
            entities_extracted=entity_count,
            graph_nodes_added=graph_nodes_added,
        )

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        return self._vs.list_documents()

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Return detailed metadata for a single document, or None if not found."""
        return self._vs.get_document_detail(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        existing = {d["doc_id"] for d in self._vs.list_documents()}
        if doc_id not in existing:
            return False
        self._vs.delete_document(doc_id)
        logger.info("Deleted document doc_id='%s' from vector store", doc_id)
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
        include_graph: bool = True,
        graph_depth: Optional[int] = None,
    ) -> AnswerResponse:
        """Answer a question using the full hybrid GraphRAG pipeline."""
        t0 = time.perf_counter()
        s = self._settings
        k = top_k or s.top_k_chunks
        query_id = str(uuid.uuid4())

        # 1. Hybrid retrieval — vector + graph expansion + score fusion
        result = self._retriever.retrieve(
            question=question,
            top_k=k,
            doc_id_filter=doc_id,
            include_graph=include_graph,
            graph_depth=graph_depth,
        )

        # ── Trust gate: refuse immediately when the corpus is empty ──────────
        # Calling an LLM with no context is the primary hallucination pathway.
        # Return a structured refusal without making any LLM API call.
        if not result.chunks:
            limitations = _infer_limitations(
                provider=self._llm.provider_name,
                chunks=[],
                graph_evidence=[],
                graph_nodes=self._gs.entity_count(),
                confidence=None,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Query refused (no passages) in %.0f ms (provider=%s)",
                latency_ms, self._llm.provider_name,
            )
            return AnswerResponse(
                query_id=query_id,
                question=question,
                answer=(
                    f"{_REFUSAL_PHRASE} "
                    "No relevant passages were found in the indexed corpus. "
                    "Please ingest the relevant policy documents and retry."
                ),
                answer_type="no_corpus",
                evidence_quality="insufficient",
                confidence_note=_compute_confidence_note([], [], None),
                citations=[],
                retrieved_chunks=[],
                graph_evidence=[],
                confidence=None,
                limitations=limitations,
                latency_ms=round(latency_ms, 1),
                provider=self._llm.provider_name,
                model=self._llm.model_id,
            )

        # 2. Build context, record citations, and build retrieved_chunks list
        citation_engine = CitationEngine()
        context_parts: list[str] = []
        retrieved_chunks: list[RetrievedChunk] = []

        for hit in result.chunks:
            meta = hit.metadata
            context_parts.append(f"[p.{meta.get('page_number', '?')}] {hit.text}")
            clamped_score = round(max(0.0, min(1.0, hit.score)), 4)
            heading = meta.get("section_heading") or None
            citation_engine.record(
                chunk_id=hit.chunk_id,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                page_number=int(meta.get("page_number", 0)),
                excerpt=hit.text[:250],
                relevance_score=hit.score,
                section_heading=heading,
            )
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    doc_title=meta.get("doc_title", ""),
                    page_number=int(meta.get("page_number", 0)),
                    section_heading=heading,
                    text=hit.text[:500],
                    relevance_score=clamped_score,
                )
            )

        if result.graph_evidence:
            graph_ctx = "\n".join(
                f"[GRAPH] {e.entity} —{e.relation}→ {e.target}"
                for e in result.graph_evidence
            )
            context_parts.append(f"\nKnowledge graph evidence:\n{graph_ctx}")

        # 3. Confidence scoring (computed before LLM so it can shape the prompt)
        confidence = _compute_confidence(result.chunks, result.graph_evidence)

        # ── Trust gate: inject low-evidence preamble when confidence is weak ──
        # The preamble instructs the LLM to prefix its answer with _PARTIAL_PHRASE
        # so that programmatic consumers can detect partial answers.
        user_prompt = f"Context:\n{'---'.join(context_parts)}\n\nQuestion: {question}"
        if confidence is not None and confidence < _LOW_CONFIDENCE_THRESHOLD:
            user_prompt = (
                f"[RETRIEVAL NOTE: Confidence score is low ({confidence:.2f}). "
                f"The retrieved passages may not fully address the question. "
                f"You MUST begin your answer with '{_PARTIAL_PHRASE}' and clearly "
                f"state which parts of the question the context cannot answer.]\n\n"
            ) + user_prompt

        # 4. LLM answer generation
        answer = await self._llm.acomplete(_SYSTEM_PROMPT, user_prompt)

        # 5. Trust-layer classification
        limitations = _infer_limitations(
            provider=self._llm.provider_name,
            chunks=result.chunks,
            graph_evidence=result.graph_evidence,
            graph_nodes=self._gs.entity_count(),
            confidence=confidence,
        )
        answer_type = _classify_answer_type(answer, result.chunks, confidence)
        evidence_quality = _classify_evidence_quality(result.chunks, confidence)
        confidence_note = _compute_confidence_note(
            result.chunks, result.graph_evidence, confidence
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Query answered in %.0f ms "
            "(provider=%s, type=%s, quality=%s, chunks=%d, graph=%d, conf=%.2f)",
            latency_ms, self._llm.provider_name,
            answer_type, evidence_quality,
            len(result.chunks), len(result.graph_evidence), confidence or 0.0,
        )

        # Reorder citations so that pages explicitly referenced in the answer
        # appear first; within each group, retain descending relevance order.
        citations = _rerank_citations_by_answer(citation_engine.to_schema(), answer)

        return AnswerResponse(
            query_id=query_id,
            question=question,
            answer=answer,
            answer_type=answer_type,
            evidence_quality=evidence_quality,
            confidence_note=confidence_note,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            graph_evidence=result.graph_evidence,
            confidence=confidence,
            limitations=limitations,
            latency_ms=round(latency_ms, 1),
            provider=self._llm.provider_name,
            model=self._llm.model_id,
        )


# ---------------------------------------------------------------------------
# Citation reranking
# ---------------------------------------------------------------------------

# Matches page references in LLM-generated answers: "p.3", "p. 3", "p3" (with
# optional space and optional period).  Used to promote explicitly cited pages.
_PAGE_REF_RE = re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE)


def _rerank_citations_by_answer(
    citations: list,
    answer: str,
) -> list:
    """Promote citations whose page numbers are explicitly referenced in the answer.

    After LLM generation, citations are in retrieval order (descending relevance
    score).  When the LLM explicitly names a page — e.g. "According to p.3…"
    or "(p.1, Section 2)" — those passages should appear first in the citation
    list so callers can surface the most directly referenced sources.

    Algorithm
    ---------
    1. Extract all page-number mentions from the answer text.
    2. Stable-sort citations into two groups:
       - Group 0: page_number mentioned in the answer  (explicitly cited)
       - Group 1: all others                           (implicitly used)
    3. Within each group, retain descending relevance_score order.

    If the answer contains no page references, the original order is unchanged.
    """
    referenced_pages = {
        int(m.group(1)) for m in _PAGE_REF_RE.finditer(answer)
    }
    if not referenced_pages:
        return citations

    return sorted(
        citations,
        key=lambda c: (
            0 if c.page_number in referenced_pages else 1,
            -c.relevance_score,
        ),
    )


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    chunks: list,
    graph_evidence: list,
) -> Optional[float]:
    """Heuristic confidence score in [0, 1].

    - Base: mean cosine similarity of the top-3 retrieved chunks (or fewer).
    - Graph boost: +0.05 per graph triple found, capped so total ≤ 1.0.
    - Returns None when no chunks were retrieved (no basis for confidence).
    """
    if not chunks:
        return None

    top = chunks[: min(3, len(chunks))]
    base = sum(c.score for c in top) / len(top)

    graph_boost = min(len(graph_evidence) * 0.05, 0.15)
    return round(min(base + graph_boost, 1.0), 3)


def _compute_confidence_note(
    chunks: list,
    graph_evidence: list,
    confidence: Optional[float],
) -> str:
    """Human-readable explanation of the confidence score and its components."""
    if not chunks:
        return (
            "No passages were retrieved from the corpus. "
            "Confidence cannot be assessed — the response is a refusal."
        )

    n = len(chunks)
    top = chunks[: min(3, n)]
    base = sum(c.score for c in top) / len(top)

    note = (
        f"Derived from {n} retrieved passage{'s' if n != 1 else ''} "
        f"(mean top-3 cosine similarity: {base:.2f})"
    )
    if graph_evidence:
        boost = min(len(graph_evidence) * 0.05, 0.15)
        note += (
            f", with +{boost:.2f} boost from "
            f"{len(graph_evidence)} graph triple{'s' if len(graph_evidence) != 1 else ''}"
        )
    else:
        note += ", without graph evidence"

    if confidence is not None:
        if confidence >= 0.7:
            note += ". Evidence quality is strong."
        elif confidence >= 0.5:
            note += ". Evidence quality is moderate — verify key claims against the source."
        elif confidence >= _LOW_CONFIDENCE_THRESHOLD:
            note += ". Evidence quality is weak — treat conclusions as preliminary."
        else:
            note += (
                f". Evidence quality is insufficient (conf={confidence:.2f}) "
                "— the answer may not be reliable."
            )

    return note


# ---------------------------------------------------------------------------
# Trust-layer classifiers
# ---------------------------------------------------------------------------

def _classify_evidence_quality(
    chunks: list,
    confidence: Optional[float],
) -> str:
    """Categorical evidence quality label derived from confidence and chunk count."""
    if not chunks or confidence is None:
        return "insufficient"
    if confidence >= 0.7 and len(chunks) >= 3:
        return "strong"
    if confidence >= 0.5 and len(chunks) >= 2:
        return "moderate"
    if confidence >= _LOW_CONFIDENCE_THRESHOLD:
        return "weak"
    return "insufficient"


def _classify_answer_type(
    answer: str,
    chunks: list,
    confidence: Optional[float],
) -> str:
    """Classify the answer as cited / partial / refused / no_corpus."""
    if not chunks:
        return "no_corpus"
    if _REFUSAL_PHRASE in answer:
        return "refused"
    if _PARTIAL_PHRASE in answer or (
        confidence is not None and confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        return "partial"
    return "cited"


# ---------------------------------------------------------------------------
# Limitations inference
# ---------------------------------------------------------------------------

def _infer_limitations(
    provider: str,
    chunks: list,
    graph_evidence: list,
    graph_nodes: int,
    confidence: Optional[float],
) -> list[str]:
    """Build a list of human-readable limitation strings for this response."""
    lims: list[str] = []

    if provider == "mock":
        lims.append(
            "Mock LLM provider active — responses are deterministic stubs. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai for real answers."
        )

    if not chunks:
        lims.append(
            "No relevant passages were found in the corpus. "
            "Ingest more documents or broaden the question."
        )
    elif len(chunks) == 1:
        lims.append(
            "Only one supporting passage was retrieved. "
            "Answers based on a single chunk may miss broader context."
        )

    if graph_nodes == 0:
        lims.append(
            "Knowledge graph is empty — entity extraction requires spaCy "
            "(python -m spacy download en_core_web_sm). "
            "Graph evidence and score fusion are unavailable."
        )
    elif not graph_evidence:
        lims.append(
            "No graph evidence matched this query. "
            "The retrieved passages may not contain recognised named entities."
        )

    if confidence is not None and confidence < 0.4:
        lims.append(
            f"Retrieval confidence is low ({confidence:.2f}). "
            "The corpus may not contain sufficient information to answer this question reliably."
        )

    return lims
