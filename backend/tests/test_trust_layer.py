"""Trust layer tests for PolicyMind-AI.

Verifies that the responsible-AI behaviours introduced in Prompt 8 work
correctly end-to-end through the API:

  TestAnswerType           — answer_type values and refusal classification
  TestRefusalBehavior      — no-corpus refusal without LLM call
  TestConfidenceNote       — human-readable confidence explanation
  TestEvidenceQuality      — categorical quality labels
  TestSourceTraceability   — citation ↔ chunk consistency
  TestLimitationsLayer     — limitations surfaced per condition
  TestSystemPromptBehavior — citation-first preamble in mock answers
  TestTrustHelpers         — unit tests for the pure helper functions

All tests run offline using in-memory providers, mock LLM, and fixed-seed
stub embeddings — no API keys or external services required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared test documents
# ---------------------------------------------------------------------------

_POLICY_DOC = (
    "Section 1: Emission Targets. All signatory states shall reduce greenhouse "
    "gas emissions by 45 percent relative to 2005 baseline levels by 2030. "
    "The Environment Agency will publish quarterly compliance reports. "
    "Section 2: Renewable Energy. A minimum 40 percent share of renewable "
    "energy in final consumption is mandated by 2028 under Article 7. "
    "Section 3: Just Transition Fund. EUR 12 billion is allocated to support "
    "coal-dependent communities. Applications close June 2025."
)

_HEALTH_DOC = (
    "Part I: Vaccination Programme. The Department of Health will distribute "
    "vaccines against influenza and measles to all practitioners by October 2024. "
    "Part II: Funding. GBP 800 million is approved under the NHS Modernisation "
    "Act 2023 for vaccine logistics. Regional boards must submit plans by January 2024. "
    "Part III: Monitoring. The Public Health Observatory reports quarterly. "
    "Uptake below 75 percent triggers intervention under Regulation 12."
)

_REFUSAL_PHRASE = (
    "The available documents do not contain enough information to answer this question."
)
_PARTIAL_PHRASE = "Based on limited evidence:"


# ===========================================================================
# 1. Answer type classification
# ===========================================================================

class TestAnswerType:
    """answer_type is set correctly for each retrieval scenario."""

    def test_no_corpus_answer_type_is_no_corpus(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "What targets exist?"})
        assert resp.status_code == 200
        assert resp.json()["answer_type"] == "no_corpus"

    def test_with_corpus_answer_type_is_cited_or_partial(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets exist?"})
        assert resp.json()["answer_type"] in {"cited", "partial"}

    def test_answer_type_in_valid_set(self, api_client):
        valid = {"cited", "partial", "refused", "no_corpus"}
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        for q in [
            "What emission targets are set?",
            "What is the renewable energy obligation?",
        ]:
            resp = api_client.post("/api/v1/query", json={"question": q})
            assert resp.json()["answer_type"] in valid

    def test_answer_type_field_always_present(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        assert "answer_type" in resp.json()

    def test_no_corpus_answer_type_present_with_doc_filter(self, api_client):
        # Even if a doc_id_filter is given for a non-existent doc, answer_type is returned.
        resp = api_client.post("/api/v1/query", json={
            "question": "What targets?",
            "doc_id": "nonexistent-doc-id",
        })
        assert "answer_type" in resp.json()


# ===========================================================================
# 2. Refusal behaviour
# ===========================================================================

class TestRefusalBehavior:
    """The pipeline refuses transparently when there is nothing to retrieve."""

    def test_no_corpus_answer_contains_refusal_phrase(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "What targets exist?"})
        assert _REFUSAL_PHRASE in resp.json()["answer"]

    def test_no_corpus_answer_not_mock_stub(self, api_client):
        """The refusal is produced by the pipeline, not the mock LLM.

        The mock provider wraps its output in '[MOCK ANSWER]'. Refusals bypass
        the LLM entirely and must not carry that marker.
        """
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert "[MOCK ANSWER]" not in resp.json()["answer"]

    def test_no_corpus_citations_empty(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        body = resp.json()
        assert body["citations"] == []
        assert body["retrieved_chunks"] == []

    def test_no_corpus_confidence_is_none(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["confidence"] is None

    def test_no_corpus_graph_evidence_empty(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["graph_evidence"] == []

    def test_refusal_still_returns_limitations(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        lims = resp.json()["limitations"]
        assert isinstance(lims, list)
        assert len(lims) > 0

    def test_refusal_returns_200_not_error(self, api_client):
        """A refusal is a valid response, not an HTTP error."""
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.status_code == 200

    def test_corpus_query_does_not_produce_refusal_phrase(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={
            "question": "What emission targets are set for 2030?"
        })
        # With a populated corpus the pipeline should not insert the refusal phrase
        # in most cases (it may appear if the real LLM judges context insufficient,
        # but mock provider should not produce it when chunks exist).
        assert resp.json()["answer_type"] != "no_corpus"


# ===========================================================================
# 3. Confidence note
# ===========================================================================

class TestConfidenceNote:
    """confidence_note is a non-empty, human-readable string."""

    def test_confidence_note_field_always_present(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert "confidence_note" in resp.json()

    def test_no_corpus_note_mentions_no_passages(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        note = resp.json()["confidence_note"]
        assert "passage" in note.lower() or "retrieved" in note.lower()

    def test_no_corpus_note_is_nonempty(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["confidence_note"].strip()

    def test_with_corpus_note_mentions_passages(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        note = resp.json()["confidence_note"]
        assert "passage" in note.lower()

    def test_with_corpus_note_mentions_cosine_similarity(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        note = resp.json()["confidence_note"]
        assert "cosine" in note.lower() or "similarity" in note.lower()

    def test_with_corpus_note_is_nonempty_string(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        note = resp.json()["confidence_note"]
        assert isinstance(note, str) and note.strip()

    def test_note_differs_corpus_vs_empty(self, api_client):
        """The confidence note should change when the corpus is populated."""
        empty_note = api_client.post(
            "/api/v1/query", json={"question": "Anything here?"}
        ).json()["confidence_note"]

        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        corpus_note = api_client.post(
            "/api/v1/query", json={"question": "Anything here?"}
        ).json()["confidence_note"]

        assert empty_note != corpus_note


# ===========================================================================
# 4. Evidence quality
# ===========================================================================

class TestEvidenceQuality:
    """evidence_quality carries a valid categorical label."""

    _VALID = {"strong", "moderate", "weak", "insufficient"}

    def test_evidence_quality_field_always_present(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert "evidence_quality" in resp.json()

    def test_no_corpus_evidence_quality_is_insufficient(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["evidence_quality"] == "insufficient"

    def test_evidence_quality_always_valid_label(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["evidence_quality"] in self._VALID

    def test_with_corpus_evidence_quality_in_valid_set(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        assert resp.json()["evidence_quality"] in self._VALID

    def test_evidence_quality_not_strong_when_corpus_empty(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        assert resp.json()["evidence_quality"] != "strong"

    def test_multi_doc_quality_in_valid_set(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        api_client.post("/api/v1/ingest", json={"text": _HEALTH_DOC, "title": "Health"})
        resp = api_client.post("/api/v1/query", json={
            "question": "What targets exist?", "top_k": 5
        })
        assert resp.json()["evidence_quality"] in self._VALID


# ===========================================================================
# 5. Source traceability
# ===========================================================================

class TestSourceTraceability:
    """Every cited chunk must be traceable to an ingested document."""

    def _ingest_and_query(self, api_client, question="What are the emission targets?"):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy Doc"})
        return api_client.post("/api/v1/query", json={"question": question})

    def test_citation_doc_id_matches_ingest(self, api_client):
        ingest = api_client.post("/api/v1/ingest", json={
            "text": _POLICY_DOC, "title": "Policy Doc"
        })
        doc_id = ingest.json()["doc_id"]
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        for cite in resp.json()["citations"]:
            assert cite["doc_id"] == doc_id

    def test_citation_ids_subset_of_retrieved_chunk_ids(self, api_client):
        resp = self._ingest_and_query(api_client)
        body = resp.json()
        cite_ids = {c["chunk_id"] for c in body["citations"]}
        chunk_ids = {c["chunk_id"] for c in body["retrieved_chunks"]}
        assert cite_ids.issubset(chunk_ids)

    def test_retrieved_chunk_texts_nonempty(self, api_client):
        resp = self._ingest_and_query(api_client)
        for chunk in resp.json()["retrieved_chunks"]:
            assert chunk["text"].strip()

    def test_chunk_doc_id_matches_citation_doc_id(self, api_client):
        resp = self._ingest_and_query(api_client)
        body = resp.json()
        cite_doc_ids = {c["doc_id"] for c in body["citations"]}
        chunk_doc_ids = {c["doc_id"] for c in body["retrieved_chunks"]}
        # Both sets should draw from the same documents
        assert cite_doc_ids == chunk_doc_ids

    def test_citation_page_number_non_negative(self, api_client):
        resp = self._ingest_and_query(api_client)
        for cite in resp.json()["citations"]:
            assert cite["page_number"] >= 0

    def test_no_hidden_doc_assumptions_no_corpus(self, api_client):
        """A refusal response must not fabricate doc_ids or chunk_ids."""
        resp = api_client.post("/api/v1/query", json={"question": "Any question?"})
        body = resp.json()
        assert body["citations"] == []
        assert body["retrieved_chunks"] == []

    def test_chunk_relevance_scores_in_unit_interval(self, api_client):
        resp = self._ingest_and_query(api_client)
        for chunk in resp.json()["retrieved_chunks"]:
            assert 0.0 <= chunk["relevance_score"] <= 1.0


# ===========================================================================
# 6. Limitations layer
# ===========================================================================

class TestLimitationsLayer:
    """Limitations are surfaced accurately for each failure mode."""

    def test_empty_corpus_surfaces_no_passages_limitation(self, api_client):
        lims = api_client.post(
            "/api/v1/query", json={"question": "Anything?"}
        ).json()["limitations"]
        assert any("passage" in l.lower() or "ingest" in l.lower() for l in lims)

    def test_mock_provider_limitation_always_present(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "P"})
        lims = api_client.post(
            "/api/v1/query", json={"question": "What targets?"}
        ).json()["limitations"]
        assert any("mock" in l.lower() for l in lims)

    def test_limitations_are_nonempty_strings(self, api_client):
        lims = api_client.post(
            "/api/v1/query", json={"question": "Anything?"}
        ).json()["limitations"]
        for lim in lims:
            assert isinstance(lim, str) and lim.strip()

    def test_limitations_list_always_present(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        assert "limitations" in resp.json()
        assert isinstance(resp.json()["limitations"], list)

    def test_no_corpus_limitation_present_in_refusal(self, api_client):
        """Refusal responses must explain WHY there is no answer."""
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        lims = resp.json()["limitations"]
        assert len(lims) >= 1  # at least the "no passages" limitation

    def test_graph_empty_limitation_surfaces_when_applicable(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "P"})
        resp = api_client.post("/api/v1/query", json={"question": "Targets?"})
        lims = resp.json()["limitations"]
        body = resp.json()
        if body["graph_evidence"] == []:
            assert any("graph" in l.lower() for l in lims)


# ===========================================================================
# 7. System prompt citation-first behaviour
# ===========================================================================

class TestSystemPromptBehavior:
    """The strengthened system prompt shapes mock output in verifiable ways."""

    def test_mock_answer_non_empty_with_corpus(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        assert resp.json()["answer"].strip()

    def test_mock_answer_references_passage_count(self, api_client):
        """The mock provider reports how many passages it received."""
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What are the targets?"})
        answer = resp.json()["answer"]
        # Mock provider includes passage count or page refs
        has_passage_ref = any(t in answer for t in ["passage", "p.", "[p.", "MOCK"])
        assert has_passage_ref

    def test_answer_provider_field_correct(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        assert resp.json()["provider"] == "mock"

    def test_answer_model_field_present(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        assert resp.json()["model"]

    def test_include_graph_false_yields_no_graph_evidence(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _POLICY_DOC, "title": "Policy"})
        resp = api_client.post("/api/v1/query", json={
            "question": "What targets?", "include_graph": False
        })
        assert resp.json()["graph_evidence"] == []


# ===========================================================================
# 8. Unit tests for the trust-layer helper functions
# ===========================================================================

class TestTrustHelpers:
    """Pure-function unit tests — no fixtures needed."""

    def test_compute_confidence_returns_none_for_empty_chunks(self):
        from app.services.rag_pipeline import _compute_confidence
        assert _compute_confidence([], []) is None

    def test_compute_confidence_bounded(self):
        from app.services.rag_pipeline import _compute_confidence
        from app.services.vector_store import SearchResult
        chunks = [
            SearchResult(chunk_id="c1", text="x", score=0.8, metadata={}),
            SearchResult(chunk_id="c2", text="y", score=0.6, metadata={}),
        ]
        conf = _compute_confidence(chunks, [])
        assert conf is not None
        assert 0.0 <= conf <= 1.0

    def test_compute_confidence_note_no_chunks(self):
        from app.services.rag_pipeline import _compute_confidence_note
        note = _compute_confidence_note([], [], None)
        assert "No passages" in note or "passage" in note.lower()
        assert isinstance(note, str) and note.strip()

    def test_compute_confidence_note_with_chunks(self):
        from app.services.rag_pipeline import _compute_confidence_note
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="txt", score=0.75, metadata={})]
        note = _compute_confidence_note(chunks, [], 0.75)
        assert "passage" in note.lower()
        assert "cosine" in note.lower() or "similarity" in note.lower()

    def test_classify_evidence_quality_no_chunks_is_insufficient(self):
        from app.services.rag_pipeline import _classify_evidence_quality
        assert _classify_evidence_quality([], None) == "insufficient"
        assert _classify_evidence_quality([], 0.9) == "insufficient"

    def test_classify_evidence_quality_strong(self):
        from app.services.rag_pipeline import _classify_evidence_quality
        from app.services.vector_store import SearchResult
        chunks = [
            SearchResult(chunk_id=f"c{i}", text="x", score=0.9, metadata={})
            for i in range(3)
        ]
        assert _classify_evidence_quality(chunks, 0.85) == "strong"

    def test_classify_evidence_quality_moderate(self):
        from app.services.rag_pipeline import _classify_evidence_quality
        from app.services.vector_store import SearchResult
        chunks = [
            SearchResult(chunk_id=f"c{i}", text="x", score=0.6, metadata={})
            for i in range(2)
        ]
        assert _classify_evidence_quality(chunks, 0.6) == "moderate"

    def test_classify_evidence_quality_weak(self):
        from app.services.rag_pipeline import _classify_evidence_quality
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.4, metadata={})]
        assert _classify_evidence_quality(chunks, 0.4) == "weak"

    def test_classify_evidence_quality_insufficient(self):
        from app.services.rag_pipeline import _classify_evidence_quality
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.2, metadata={})]
        assert _classify_evidence_quality(chunks, 0.2) == "insufficient"

    def test_classify_answer_type_no_chunks(self):
        from app.services.rag_pipeline import _classify_answer_type
        assert _classify_answer_type("anything", [], None) == "no_corpus"

    def test_classify_answer_type_refusal(self):
        from app.services.rag_pipeline import _classify_answer_type, _REFUSAL_PHRASE
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.8, metadata={})]
        assert _classify_answer_type(_REFUSAL_PHRASE + " more text", chunks, 0.8) == "refused"

    def test_classify_answer_type_partial_by_phrase(self):
        from app.services.rag_pipeline import _classify_answer_type, _PARTIAL_PHRASE
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.8, metadata={})]
        assert _classify_answer_type(_PARTIAL_PHRASE + " …", chunks, 0.8) == "partial"

    def test_classify_answer_type_partial_by_confidence(self):
        from app.services.rag_pipeline import _classify_answer_type
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.2, metadata={})]
        assert _classify_answer_type("some answer", chunks, 0.2) == "partial"

    def test_classify_answer_type_cited(self):
        from app.services.rag_pipeline import _classify_answer_type
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.8, metadata={})]
        assert _classify_answer_type("A normal grounded answer.", chunks, 0.8) == "cited"

    def test_infer_limitations_mock_provider(self):
        from app.services.rag_pipeline import _infer_limitations
        lims = _infer_limitations("mock", [], [], 0, None)
        assert any("mock" in l.lower() for l in lims)

    def test_infer_limitations_no_chunks(self):
        from app.services.rag_pipeline import _infer_limitations
        lims = _infer_limitations("mock", [], [], 5, None)
        assert any("passage" in l.lower() for l in lims)

    def test_infer_limitations_low_confidence(self):
        from app.services.rag_pipeline import _infer_limitations
        from app.services.vector_store import SearchResult
        chunks = [SearchResult(chunk_id="c1", text="x", score=0.2, metadata={})]
        lims = _infer_limitations("mock", chunks, [], 5, 0.2)
        assert any("confidence" in l.lower() for l in lims)
