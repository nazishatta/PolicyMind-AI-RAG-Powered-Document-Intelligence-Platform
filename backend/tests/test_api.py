"""Integration tests for all FastAPI endpoints.

The test suite runs fully offline:
  - InMemoryVectorStore (no ChromaDB)
  - InMemoryGraphService (no Neo4j)
  - MockProvider (no LLM API key)
  - StubEmbedding in conftest (no model download)

Coverage:
  GET  /                              — root index
  GET  /health                        — enriched health check
  POST /api/v1/ingest                 — text + URL-connector paths
  GET  /api/v1/documents              — list
  GET  /api/v1/documents/{doc_id}     — single-doc metadata
  DELETE /api/v1/documents/{doc_id}   — removal
  POST /api/v1/query                  — answer + retrieved_chunks
  GET  /api/v1/graph/stats            — graph statistics
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Shared text that reliably produces ≥1 chunk (min_chars=50)
# ---------------------------------------------------------------------------

_LONG_TEXT = (
    "The National Climate Strategy 2030 commits Member States to reduce greenhouse "
    "gas emissions by 55 percent relative to 1990 levels. The European Commission "
    "will oversee compliance and allocate EUR 15 billion from the Just Transition Fund. "
    "Article 12 establishes binding renewable energy targets of 30 percent by 2025."
)


# ============================================================
# GET /
# ============================================================

class TestRoot:
    def test_root_returns_200(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200

    def test_root_contains_name(self, api_client):
        body = api_client.get("/").json()
        assert "name" in body
        assert "PolicyMind" in body["name"]

    def test_root_contains_version(self, api_client):
        body = api_client.get("/").json()
        assert "version" in body
        assert body["version"]

    def test_root_status_is_running(self, api_client):
        body = api_client.get("/").json()
        assert body["status"] == "running"

    def test_root_contains_endpoints_directory(self, api_client):
        body = api_client.get("/").json()
        assert "endpoints" in body
        eps = body["endpoints"]
        assert isinstance(eps, dict)
        assert len(eps) >= 5

    def test_root_endpoints_includes_ingest(self, api_client):
        body = api_client.get("/").json()
        values = " ".join(body["endpoints"].values())
        assert "ingest" in values.lower()

    def test_root_endpoints_includes_query(self, api_client):
        body = api_client.get("/").json()
        values = " ".join(body["endpoints"].values())
        assert "query" in values.lower()

    def test_root_contains_providers(self, api_client):
        body = api_client.get("/").json()
        assert "providers" in body
        assert body["providers"]["llm"] == "mock"

    def test_root_contains_docs_url(self, api_client):
        body = api_client.get("/").json()
        assert "docs_url" in body
        assert body["docs_url"]

    def test_root_description_non_empty(self, api_client):
        body = api_client.get("/").json()
        assert len(body.get("description", "")) > 10


# ============================================================
# GET /health
# ============================================================

class TestHealth:
    def test_health_returns_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, api_client):
        body = api_client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_contains_version(self, api_client):
        body = api_client.get("/health").json()
        assert "version" in body
        assert body["version"]

    def test_health_llm_provider_is_mock(self, api_client):
        body = api_client.get("/health").json()
        assert body["llm_provider"] == "mock"

    def test_health_llm_model_present(self, api_client):
        body = api_client.get("/health").json()
        assert "llm_model" in body
        assert body["llm_model"]

    def test_health_vector_store_present(self, api_client):
        body = api_client.get("/health").json()
        assert "vector_store" in body

    def test_health_vector_store_chunks_is_int(self, api_client):
        body = api_client.get("/health").json()
        assert isinstance(body["vector_store_chunks"], int)
        assert body["vector_store_chunks"] >= 0

    def test_health_graph_provider_present(self, api_client):
        body = api_client.get("/health").json()
        assert "graph_provider" in body

    def test_health_graph_entities_is_int(self, api_client):
        body = api_client.get("/health").json()
        assert isinstance(body["graph_entities"], int)
        assert body["graph_entities"] >= 0

    def test_health_graph_relations_is_int(self, api_client):
        body = api_client.get("/health").json()
        assert isinstance(body["graph_relations"], int)

    def test_health_chunk_count_increases_after_ingest(self, api_client):
        before = api_client.get("/health").json()["vector_store_chunks"]
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Health Test"})
        after = api_client.get("/health").json()["vector_store_chunks"]
        assert after > before


# ============================================================
# POST /api/v1/ingest
# ============================================================

class TestIngest:
    def test_ingest_text_returns_201(self, api_client):
        r = api_client.post(
            "/api/v1/ingest",
            json={"text": _LONG_TEXT, "title": "Ingest Test"},
        )
        assert r.status_code == 201

    def test_ingest_response_has_doc_id(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert "doc_id" in r.json()

    def test_ingest_response_has_chunk_count(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert r.json()["chunk_count"] > 0

    def test_ingest_response_has_entities_extracted(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert "entities_extracted" in r.json()

    def test_ingest_response_has_processing_status(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert r.json()["processing_status"] == "completed"

    def test_ingest_response_has_message(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert r.json()["message"]

    def test_ingest_source_url_none_for_text(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        assert r.json()["source_url"] is None

    def test_ingest_rejects_missing_source(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"title": "No source"})
        assert r.status_code == 422

    def test_ingest_rejects_both_url_and_text(self, api_client):
        r = api_client.post(
            "/api/v1/ingest",
            json={"url": "https://example.com/doc.pdf", "text": "Also text"},
        )
        assert r.status_code == 422

    def test_ingest_is_idempotent(self, api_client):
        """Re-ingesting the same content returns the same doc_id."""
        r1 = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Idem"})
        r2 = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Idem"})
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["doc_id"] == r2.json()["doc_id"]


# ============================================================
# GET /api/v1/documents
# ============================================================

class TestListDocuments:
    def test_list_returns_200(self, api_client):
        r = api_client.get("/api/v1/documents")
        assert r.status_code == 200

    def test_list_empty_store(self, api_client):
        body = api_client.get("/api/v1/documents").json()
        assert body["documents"] == []
        assert body["total"] == 0

    def test_list_after_ingest(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Listed"})
        body = api_client.get("/api/v1/documents").json()
        assert body["total"] >= 1

    def test_list_document_has_required_fields(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Fields"})
        docs = api_client.get("/api/v1/documents").json()["documents"]
        assert docs
        for doc in docs:
            assert "doc_id" in doc
            assert "doc_title" in doc
            assert "chunk_count" in doc
            assert doc["chunk_count"] >= 0

    def test_list_total_matches_documents_length(self, api_client):
        body = api_client.get("/api/v1/documents").json()
        assert body["total"] == len(body["documents"])


# ============================================================
# GET /api/v1/documents/{doc_id}
# ============================================================

class TestDocumentDetail:
    def _ingest(self, api_client, title: str = "Detail Test") -> str:
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": title})
        assert r.status_code == 201
        return r.json()["doc_id"]

    def test_get_document_returns_200(self, api_client):
        doc_id = self._ingest(api_client)
        r = api_client.get(f"/api/v1/documents/{doc_id}")
        assert r.status_code == 200

    def test_get_document_unknown_returns_404(self, api_client):
        r = api_client.get("/api/v1/documents/nonexistent_abc123")
        assert r.status_code == 404

    def test_get_document_has_doc_id(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert body["doc_id"] == doc_id

    def test_get_document_has_doc_title(self, api_client):
        doc_id = self._ingest(api_client, title="Specific Title")
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert body["doc_title"] == "Specific Title"

    def test_get_document_has_chunk_count(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert "chunk_count" in body
        assert body["chunk_count"] > 0

    def test_get_document_has_page_count(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert "page_count" in body
        assert body["page_count"] >= 1

    def test_get_document_has_sections_list(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert "sections" in body
        assert isinstance(body["sections"], list)

    def test_get_document_has_status(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert body["status"] == "indexed"

    def test_get_document_source_url_none_for_text(self, api_client):
        doc_id = self._ingest(api_client)
        body = api_client.get(f"/api/v1/documents/{doc_id}").json()
        assert body["source_url"] is None

    def test_get_document_gone_after_delete(self, api_client):
        doc_id = self._ingest(api_client)
        api_client.delete(f"/api/v1/documents/{doc_id}")
        r = api_client.get(f"/api/v1/documents/{doc_id}")
        assert r.status_code == 404


# ============================================================
# DELETE /api/v1/documents/{doc_id}
# ============================================================

class TestDeleteDocument:
    def test_delete_returns_200(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Del"})
        doc_id = r.json()["doc_id"]
        assert api_client.delete(f"/api/v1/documents/{doc_id}").status_code == 200

    def test_delete_unknown_returns_404(self, api_client):
        r = api_client.delete("/api/v1/documents/no_such_id")
        assert r.status_code == 404

    def test_delete_response_has_deleted_true(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        doc_id = r.json()["doc_id"]
        body = api_client.delete(f"/api/v1/documents/{doc_id}").json()
        assert body["deleted"] is True
        assert body["doc_id"] == doc_id

    def test_delete_removes_from_list(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Remove"})
        doc_id = r.json()["doc_id"]
        api_client.delete(f"/api/v1/documents/{doc_id}")
        ids = [d["doc_id"] for d in api_client.get("/api/v1/documents").json()["documents"]]
        assert doc_id not in ids


# ============================================================
# POST /api/v1/query
# ============================================================

class TestQuery:
    def _ingest_and_query(self, api_client, question: str) -> dict:
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Q-Doc"})
        r = api_client.post("/api/v1/query", json={"question": question})
        assert r.status_code == 200
        return r.json()

    def test_query_returns_200(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        r = api_client.post("/api/v1/query", json={"question": "What are the emission targets?"})
        assert r.status_code == 200

    def test_query_has_answer(self, api_client):
        body = self._ingest_and_query(api_client, "What emission reductions are required?")
        assert "answer" in body and body["answer"]

    def test_query_has_query_id(self, api_client):
        body = self._ingest_and_query(api_client, "What is Article 12 about?")
        assert "query_id" in body and body["query_id"]

    def test_query_has_citations_list(self, api_client):
        body = self._ingest_and_query(api_client, "What is the Just Transition Fund?")
        assert isinstance(body["citations"], list)

    def test_query_has_retrieved_chunks(self, api_client):
        body = self._ingest_and_query(api_client, "What is the renewable energy target?")
        assert "retrieved_chunks" in body
        assert isinstance(body["retrieved_chunks"], list)

    def test_query_retrieved_chunks_have_required_fields(self, api_client):
        body = self._ingest_and_query(api_client, "What are the reduction targets?")
        for chunk in body["retrieved_chunks"]:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "text" in chunk
            assert "page_number" in chunk
            assert "relevance_score" in chunk
            assert 0.0 <= chunk["relevance_score"] <= 1.0

    def test_query_has_graph_evidence_list(self, api_client):
        body = self._ingest_and_query(api_client, "Who oversees compliance?")
        assert isinstance(body["graph_evidence"], list)

    def test_query_has_confidence_field(self, api_client):
        body = self._ingest_and_query(api_client, "What is the Just Transition Fund?")
        assert "confidence" in body

    def test_query_has_limitations_list(self, api_client):
        body = self._ingest_and_query(api_client, "What are the targets?")
        assert isinstance(body["limitations"], list)

    def test_query_has_latency_ms(self, api_client):
        body = self._ingest_and_query(api_client, "What is the energy target?")
        assert body["latency_ms"] > 0

    def test_query_has_provider(self, api_client):
        body = self._ingest_and_query(api_client, "What are the emission targets?")
        assert body["provider"] == "mock"

    def test_query_has_model(self, api_client):
        body = self._ingest_and_query(api_client, "What are the emission targets?")
        assert body["model"]

    def test_query_rejects_short_question(self, api_client):
        r = api_client.post("/api/v1/query", json={"question": "Hi"})
        assert r.status_code == 422

    def test_query_with_doc_id_filter(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Filtered"})
        doc_id = r.json()["doc_id"]
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission reduction targets?", "doc_id": doc_id},
        )
        assert r.status_code == 200

    def test_query_empty_store_returns_mock_answer(self, api_client):
        r = api_client.post("/api/v1/query", json={"question": "Any policy question here?"})
        assert r.status_code == 200
        assert r.json()["answer"]

    def test_query_limitations_include_mock_warning(self, api_client):
        body = self._ingest_and_query(api_client, "What are the emission targets?")
        lims = " ".join(body["limitations"]).lower()
        assert "mock" in lims

    def test_query_citations_relevance_scores_in_range(self, api_client):
        body = self._ingest_and_query(api_client, "What is Article 12?")
        for c in body["citations"]:
            assert 0.0 <= c["relevance_score"] <= 1.0

    def test_query_include_graph_false(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT})
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "include_graph_evidence": False},
        )
        assert r.status_code == 200
        assert r.json()["graph_evidence"] == []


# ============================================================
# POST /api/v1/query — graph_depth validation
# ============================================================

class TestGraphDepth:
    """Validate the optional graph_depth field on QueryRequest.

    Valid range is 1–3 (inclusive). Values outside that range must be
    rejected with 422 before the pipeline is ever called.
    """

    def _ingest(self, api_client) -> None:
        api_client.post("/api/v1/ingest", json={"text": _LONG_TEXT, "title": "Depth-Doc"})

    def test_query_accepts_graph_depth_1(self, api_client):
        self._ingest(api_client)
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "graph_depth": 1},
        )
        assert r.status_code == 200

    def test_query_accepts_graph_depth_2(self, api_client):
        self._ingest(api_client)
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "graph_depth": 2},
        )
        assert r.status_code == 200

    def test_query_accepts_graph_depth_3(self, api_client):
        self._ingest(api_client)
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "graph_depth": 3},
        )
        assert r.status_code == 200

    def test_query_rejects_graph_depth_0(self, api_client):
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "graph_depth": 0},
        )
        assert r.status_code == 422

    def test_query_rejects_graph_depth_4(self, api_client):
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?", "graph_depth": 4},
        )
        assert r.status_code == 422

    def test_query_graph_depth_omitted_uses_default(self, api_client):
        self._ingest(api_client)
        r = api_client.post(
            "/api/v1/query",
            json={"question": "What are the emission targets?"},
        )
        assert r.status_code == 200

    def test_query_graph_depth_2_returns_full_response_shape(self, api_client):
        self._ingest(api_client)
        body = api_client.post(
            "/api/v1/query",
            json={
                "question": "What are the emission targets?",
                "graph_depth": 2,
                "include_graph_evidence": True,
            },
        ).json()
        for field in (
            "query_id", "question", "answer", "answer_type", "evidence_quality",
            "confidence_note", "citations", "retrieved_chunks", "graph_evidence",
            "confidence", "limitations", "latency_ms", "provider", "model",
        ):
            assert field in body, f"Missing field: {field}"

    def test_query_graph_depth_3_response_shape_valid(self, api_client):
        self._ingest(api_client)
        body = api_client.post(
            "/api/v1/query",
            json={
                "question": "What are the emission targets?",
                "graph_depth": 3,
                "include_graph_evidence": True,
            },
        ).json()
        assert isinstance(body["citations"], list)
        assert isinstance(body["graph_evidence"], list)
        assert isinstance(body["limitations"], list)


# ============================================================
# GET /api/v1/graph/stats
# ============================================================

class TestGraphStats:
    def test_graph_stats_returns_200(self, api_client):
        r = api_client.get("/api/v1/graph/stats")
        assert r.status_code == 200

    def test_graph_stats_has_total_entities(self, api_client):
        body = api_client.get("/api/v1/graph/stats").json()
        assert "total_entities" in body
        assert isinstance(body["total_entities"], int)

    def test_graph_stats_has_total_relations(self, api_client):
        body = api_client.get("/api/v1/graph/stats").json()
        assert "total_relations" in body

    def test_graph_stats_has_graph_provider(self, api_client):
        body = api_client.get("/api/v1/graph/stats").json()
        assert "graph_provider" in body

    def test_graph_stats_has_graph_enabled(self, api_client):
        body = api_client.get("/api/v1/graph/stats").json()
        assert "graph_enabled" in body
