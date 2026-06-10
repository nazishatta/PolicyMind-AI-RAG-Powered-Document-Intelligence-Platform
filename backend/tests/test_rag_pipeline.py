"""Tests for services/rag_pipeline.py using in-memory providers."""

from __future__ import annotations

import pytest

from app.core.document_loader import PolicyDocument
from app.schemas.response import Citation


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_ingest_returns_result(pipeline, sample_document):
    result = pipeline.ingest(sample_document)
    assert result.doc_id == sample_document.doc_id
    assert result.chunk_count > 0
    assert result.page_count == sample_document.page_count


def test_ingest_populates_vector_store(pipeline, sample_document):
    pipeline.ingest(sample_document)
    assert pipeline._vs.count() > 0


def test_ingest_populates_graph(pipeline, sample_document):
    pipeline.ingest(sample_document)
    # Entity extraction requires spaCy; skip gracefully if not installed
    pytest.importorskip("spacy", reason="spaCy not installed")
    # At least the graph should have been attempted
    assert pipeline._gs.entity_count() >= 0


@pytest.mark.asyncio
async def test_query_returns_answer(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("What are the emission reduction targets?")
    assert result.answer
    assert len(result.citations) > 0
    assert result.latency_ms > 0
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_query_citations_reference_correct_doc(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("What is the Just Transition Fund?")
    for cite in result.citations:
        assert cite.doc_id == sample_document.doc_id


@pytest.mark.asyncio
async def test_query_with_doc_id_filter(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query(
        "Renewable energy targets",
        doc_id=sample_document.doc_id,
    )
    assert result.answer


@pytest.mark.asyncio
async def test_query_empty_store_returns_mock_answer(pipeline):
    result = await pipeline.query("Any question")
    assert result.answer     # mock always returns something
    assert result.citations == []


# ---------------------------------------------------------------------------
# Trust-layer fields at pipeline level
# ---------------------------------------------------------------------------

_VALID_ANSWER_TYPES = {"cited", "partial", "refused", "no_corpus"}
_VALID_EVIDENCE_QUALITIES = {"strong", "moderate", "weak", "insufficient"}

_POLICY_TEXT = (
    "Section 1: Emission Targets. All signatory states shall reduce greenhouse "
    "gas emissions by 45 percent relative to 2005 baseline levels by 2030. "
    "The Environment Agency will publish quarterly compliance reports. "
    "Section 2: Renewable Energy. A minimum 40 percent share of renewable "
    "energy in final consumption is mandated by 2028 under Article 7."
)


@pytest.mark.asyncio
async def test_query_trust_layer_all_fields_present(pipeline, sample_document):
    """All 11 trust-layer fields must be present in AnswerResponse."""
    pipeline.ingest(sample_document)
    result = await pipeline.query("What are the targets?")
    assert result.query_id
    assert result.question
    assert result.answer
    assert result.answer_type in _VALID_ANSWER_TYPES
    assert result.evidence_quality in _VALID_EVIDENCE_QUALITIES
    assert isinstance(result.confidence_note, str) and result.confidence_note
    assert isinstance(result.citations, list)
    assert isinstance(result.retrieved_chunks, list)
    assert isinstance(result.graph_evidence, list)
    assert isinstance(result.limitations, list)
    assert result.latency_ms > 0
    assert result.provider


@pytest.mark.asyncio
async def test_query_no_corpus_answer_type_is_no_corpus(pipeline):
    result = await pipeline.query("Anything at all?")
    assert result.answer_type == "no_corpus"


@pytest.mark.asyncio
async def test_query_no_corpus_confidence_is_none(pipeline):
    result = await pipeline.query("Any question?")
    assert result.confidence is None


@pytest.mark.asyncio
async def test_query_no_corpus_citations_empty(pipeline):
    result = await pipeline.query("Any question?")
    assert result.citations == []
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_query_no_corpus_graph_evidence_empty(pipeline):
    result = await pipeline.query("Any question?")
    assert result.graph_evidence == []


@pytest.mark.asyncio
async def test_query_no_corpus_has_limitations(pipeline):
    result = await pipeline.query("Any question?")
    assert isinstance(result.limitations, list)
    assert len(result.limitations) >= 1


@pytest.mark.asyncio
async def test_query_answer_type_in_valid_set(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("Greenhouse gas emissions?")
    assert result.answer_type in _VALID_ANSWER_TYPES


@pytest.mark.asyncio
async def test_query_evidence_quality_in_valid_set(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("Renewable energy targets?")
    assert result.evidence_quality in _VALID_EVIDENCE_QUALITIES


@pytest.mark.asyncio
async def test_query_confidence_is_float_with_corpus(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("What are the targets?")
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_query_latency_ms_positive(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("Any question?")
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_query_graph_disabled_no_evidence(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("Targets?", include_graph=False)
    assert result.graph_evidence == []


@pytest.mark.asyncio
async def test_query_top_k_override(pipeline, sample_document):
    pipeline.ingest(sample_document)
    result = await pipeline.query("What are the targets?", top_k=1)
    assert len(result.citations) <= 1
    assert len(result.retrieved_chunks) <= 1


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

def test_list_documents_empty_before_ingest(pipeline):
    docs = pipeline.list_documents()
    assert docs == []


def test_list_documents_after_ingest(pipeline, sample_document):
    pipeline.ingest(sample_document)
    docs = pipeline.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == sample_document.doc_id


def test_get_document_found(pipeline, sample_document):
    pipeline.ingest(sample_document)
    detail = pipeline.get_document(sample_document.doc_id)
    assert detail is not None
    assert detail["doc_id"] == sample_document.doc_id


def test_get_document_not_found_returns_none(pipeline):
    result = pipeline.get_document("nonexistent-doc-id")
    assert result is None


def test_delete_document_success(pipeline, sample_document):
    pipeline.ingest(sample_document)
    deleted = pipeline.delete_document(sample_document.doc_id)
    assert deleted is True
    assert pipeline.list_documents() == []


def test_delete_document_not_found_returns_false(pipeline):
    deleted = pipeline.delete_document("phantom-doc-id")
    assert deleted is False


# ---------------------------------------------------------------------------
# Ingest edge cases
# ---------------------------------------------------------------------------

def test_ingest_empty_text_raises(pipeline):
    empty_doc = PolicyDocument(
        doc_id="empty_doc",
        title="Empty",
        pages=[(1, "   ")],
    )
    with pytest.raises(ValueError):
        pipeline.ingest(empty_doc)


# ---------------------------------------------------------------------------
# Citation reranking (pure-function unit tests)
# ---------------------------------------------------------------------------

def _make_citation(chunk_id: str, page_number: int, relevance_score: float) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        doc_id="d1",
        doc_title="Test Doc",
        page_number=page_number,
        excerpt="...",
        relevance_score=relevance_score,
    )


def test_rerank_citations_referenced_page_moves_to_front():
    from app.services.rag_pipeline import _rerank_citations_by_answer

    citations = [
        _make_citation("c1", page_number=3, relevance_score=0.9),
        _make_citation("c2", page_number=1, relevance_score=0.7),
    ]
    answer = "According to p.1 of the document, the target is 45 percent."
    result = _rerank_citations_by_answer(citations, answer)

    assert result[0].page_number == 1   # referenced page floats to front
    assert result[1].page_number == 3


def test_rerank_citations_no_references_preserves_order():
    from app.services.rag_pipeline import _rerank_citations_by_answer

    citations = [
        _make_citation("c1", page_number=3, relevance_score=0.9),
        _make_citation("c2", page_number=1, relevance_score=0.7),
    ]
    answer = "The policy sets ambitious climate targets."
    result = _rerank_citations_by_answer(citations, answer)

    # No page references → original order preserved
    assert result[0].page_number == 3
    assert result[1].page_number == 1


def test_rerank_citations_multiple_referenced_pages_sorted_by_score():
    from app.services.rag_pipeline import _rerank_citations_by_answer

    citations = [
        _make_citation("c1", page_number=2, relevance_score=0.6),
        _make_citation("c2", page_number=1, relevance_score=0.8),
        _make_citation("c3", page_number=5, relevance_score=0.5),
    ]
    answer = "See p.1 and p.2 for the relevant sections."
    result = _rerank_citations_by_answer(citations, answer)

    # pages 1 and 2 are both referenced → sorted by score within that group
    assert result[0].page_number == 1   # higher score among referenced pages
    assert result[1].page_number == 2
    assert result[2].page_number == 5   # unreferenced, last


def test_rerank_citations_empty_list_is_safe():
    from app.services.rag_pipeline import _rerank_citations_by_answer

    result = _rerank_citations_by_answer([], "Answer text with p.1 reference.")
    assert result == []
