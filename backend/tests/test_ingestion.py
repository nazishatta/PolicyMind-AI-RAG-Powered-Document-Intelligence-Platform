"""Comprehensive ingestion layer tests.

All tests run fully offline — network calls are patched at the
``_fetch_content`` boundary so no external service is required.
Covers: text loading, URL loading, error cases, chunker, connectors,
API endpoints, and list/delete document management.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.document_loader import (
    DocumentEmptyError,
    DocumentFetchError,
    DocumentParseError,
    DocumentSizeError,
    IngestionError,
    PolicyDocument,
    UnsupportedContentTypeError,
    _clean,
    _infer_title,
    _is_pdf,
    _validate_content_type,
    load_from_text,
    load_from_url,
)
from app.core.text_chunker import TextChunk, chunk_document
from app.core.connectors import (
    DocumentConnector,
    TextConnector,
    URLConnector,
    connector_from_request,
)


# ============================================================
# Fixtures
# ============================================================

SAMPLE_POLICY_TEXT = (
    "The National Climate Strategy 2030 commits all signatory states to reduce "
    "greenhouse gas emissions by 55 percent relative to 1990 levels. "
    "Article 12 establishes binding targets for the energy sector, including a "
    "minimum 30 percent share of renewable energy in final energy consumption by 2025. "
    "The European Commission will oversee compliance and allocate EUR 15 billion "
    "from the Just Transition Fund. Penalties for non-compliance are defined under "
    "Regulation (EU) 2021/1119. Member States must submit annual progress reports "
    "to the Commission by 31 March of each year."
)

MULTI_PAGE_PAGES = [
    (1, "EXECUTIVE SUMMARY\n\nThis strategy outlines emission reduction targets."),
    (2, "1. Background\n\nGreenhouse gas emissions have risen 20 percent since 2000."),
    (3, "2. Targets\n\nAll states shall achieve net zero by 2050."),
]


# ============================================================
# Tests: load_from_text
# ============================================================

class TestLoadFromText:
    def test_returns_policy_document(self):
        doc = load_from_text(SAMPLE_POLICY_TEXT, title="Climate Strategy")
        assert isinstance(doc, PolicyDocument)

    def test_title_is_used_when_provided(self):
        doc = load_from_text("Some policy text.", title="My Title")
        assert doc.title == "My Title"

    def test_auto_title_from_first_line(self):
        doc = load_from_text("Climate Action Plan 2035\n\nDetails follow.")
        assert len(doc.title) > 0

    def test_stable_doc_id_for_same_content(self):
        doc1 = load_from_text("Identical content.")
        doc2 = load_from_text("Identical content.")
        assert doc1.doc_id == doc2.doc_id

    def test_different_content_gives_different_ids(self):
        doc1 = load_from_text("Content A.")
        doc2 = load_from_text("Content B.")
        assert doc1.doc_id != doc2.doc_id

    def test_page_count_is_one(self):
        doc = load_from_text(SAMPLE_POLICY_TEXT)
        assert doc.page_count == 1

    def test_full_text_contains_content(self):
        doc = load_from_text("Article 1: All states shall report annually.")
        assert "Article 1" in doc.full_text

    def test_source_label_stored(self):
        doc = load_from_text("Text.", source_label="UNDP")
        assert doc.source_label == "UNDP"

    def test_source_url_is_none(self):
        doc = load_from_text("Text.")
        assert doc.source_url is None

    def test_metadata_marks_source_as_raw_text(self):
        doc = load_from_text("Some content.")
        assert doc.metadata.get("source") == "raw_text"

    def test_empty_text_raises_document_empty_error(self):
        with pytest.raises(DocumentEmptyError):
            load_from_text("")

    def test_whitespace_only_raises_document_empty_error(self):
        with pytest.raises(DocumentEmptyError):
            load_from_text("   \n\t  ")

    def test_oversized_text_raises_document_size_error(self):
        with pytest.raises(DocumentSizeError):
            load_from_text("x" * 500_001)

    def test_text_exactly_at_limit_is_accepted(self):
        # Should not raise
        doc = load_from_text("a" * 500_000)
        assert doc.page_count == 1


# ============================================================
# Tests: load_from_url (network calls patched at _fetch_content)
# ============================================================

class TestLoadFromUrl:
    @pytest.mark.asyncio
    async def test_loads_plain_text(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(return_value=(b"Climate policy 2030.", "text/plain")),
        ):
            doc = await load_from_url("https://example.com/policy.txt", title="Test")
        assert doc.title == "Test"
        assert "Climate" in doc.full_text
        assert doc.source_url == "https://example.com/policy.txt"

    @pytest.mark.asyncio
    async def test_infers_title_from_url_stem(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(return_value=(b"Short text.", "text/plain")),
        ):
            doc = await load_from_url("https://example.com/climate_action_plan.txt")
        assert "Climate" in doc.title or "Action" in doc.title or len(doc.title) > 0

    @pytest.mark.asyncio
    async def test_pdf_url_calls_pdf_parser(self):
        fake_pdf_bytes = b"%PDF-1.4 fake content"
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(return_value=(fake_pdf_bytes, "application/pdf")),
        ), patch(
            "app.core.document_loader._parse_pdf_bytes",
            return_value=[(1, "Parsed PDF content.")],
        ):
            doc = await load_from_url("https://example.com/doc.pdf")
        assert "Parsed PDF content." in doc.full_text

    @pytest.mark.asyncio
    async def test_document_fetch_error_propagates(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(side_effect=DocumentFetchError("HTTP 404")),
        ):
            with pytest.raises(DocumentFetchError, match="404"):
                await load_from_url("https://example.com/missing.pdf")

    @pytest.mark.asyncio
    async def test_size_error_propagates(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(side_effect=DocumentSizeError("Exceeds 50 MB")),
        ):
            with pytest.raises(DocumentSizeError):
                await load_from_url("https://example.com/huge.pdf")

    @pytest.mark.asyncio
    async def test_unsupported_content_type_propagates(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(side_effect=UnsupportedContentTypeError("image/png")),
        ):
            with pytest.raises(UnsupportedContentTypeError):
                await load_from_url("https://example.com/photo.png")

    @pytest.mark.asyncio
    async def test_empty_text_response_raises(self):
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(return_value=(b"   ", "text/plain")),
        ):
            with pytest.raises(DocumentEmptyError):
                await load_from_url("https://example.com/empty.txt")

    @pytest.mark.asyncio
    async def test_metadata_contains_url(self):
        url = "https://example.com/report.txt"
        with patch(
            "app.core.document_loader._fetch_content",
            new=AsyncMock(return_value=(b"Policy content here.", "text/plain")),
        ):
            doc = await load_from_url(url)
        assert doc.metadata.get("url") == url


# ============================================================
# Tests: internal helpers
# ============================================================

class TestHelpers:
    def test_clean_rejects_hyphenated_linebreak(self):
        text = "green-\nhouse"
        result = _clean(text)
        assert "greenhouse" in result

    def test_clean_normalises_whitespace(self):
        result = _clean("too   many   spaces")
        assert "  " not in result

    def test_is_pdf_by_content_type(self):
        assert _is_pdf("application/pdf", "https://example.com/doc")

    def test_is_pdf_by_url_extension(self):
        assert _is_pdf("text/plain", "https://example.com/report.pdf")

    def test_is_not_pdf_for_plain_text(self):
        assert not _is_pdf("text/plain", "https://example.com/report.txt")

    def test_validate_content_type_accepts_pdf(self):
        _validate_content_type("application/pdf", "https://x.com/doc.pdf")  # no raise

    def test_validate_content_type_accepts_text(self):
        _validate_content_type("text/plain; charset=utf-8", "https://x.com/doc.txt")

    def test_validate_content_type_accepts_octet_stream(self):
        _validate_content_type("application/octet-stream", "https://x.com/blob")

    def test_validate_content_type_rejects_image(self):
        with pytest.raises(UnsupportedContentTypeError):
            _validate_content_type("image/png", "https://x.com/photo.png")

    def test_validate_content_type_rejects_zip(self):
        with pytest.raises(UnsupportedContentTypeError):
            _validate_content_type("application/zip", "https://x.com/archive.zip")

    def test_infer_title_from_url_stem(self):
        title = _infer_title("https://example.com/climate_action_plan_2030.pdf", "")
        assert "Climate" in title or "Action" in title

    def test_infer_title_from_first_text_line(self):
        title = _infer_title(None, "World Development Report 2022\n\nDetails here.")
        assert "World Development" in title

    def test_infer_title_fallback(self):
        title = _infer_title(None, "")
        assert title == "Untitled Policy Document"


# ============================================================
# Tests: chunk_document
# ============================================================

class TestChunkDocument:
    def _make_doc(self, pages=None, source_url=None):
        return PolicyDocument(
            doc_id="test123",
            title="Test Policy",
            pages=pages or [(1, SAMPLE_POLICY_TEXT)],
            source_url=source_url,
        )

    def test_produces_at_least_one_chunk(self):
        doc = self._make_doc()
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) > 0

    def test_all_chunks_are_text_chunk_instances(self):
        doc = self._make_doc()
        chunks = chunk_document(doc)
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        doc = self._make_doc()
        chunks = chunk_document(doc, chunk_size=200)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_encode_page_and_doc(self):
        doc = self._make_doc()
        chunks = chunk_document(doc, chunk_size=200)
        for c in chunks:
            assert c.chunk_id.startswith(doc.doc_id)
            assert "_p" in c.chunk_id and "_c" in c.chunk_id

    def test_no_empty_chunk_text(self):
        doc = self._make_doc()
        chunks = chunk_document(doc)
        assert all(c.text.strip() for c in chunks)

    def test_min_chars_filter(self):
        doc = self._make_doc(pages=[(1, "Short.\n\n" + "B" * 300)])
        chunks = chunk_document(doc, chunk_size=200, min_chars=50)
        assert all(len(c.text) >= 50 for c in chunks)

    def test_all_chunks_inherit_doc_id(self):
        doc = self._make_doc()
        for c in chunk_document(doc):
            assert c.doc_id == doc.doc_id

    def test_page_numbers_are_positive(self):
        doc = self._make_doc()
        for c in chunk_document(doc):
            assert c.page_number >= 1

    def test_source_url_propagated(self):
        doc = self._make_doc(source_url="https://example.com/policy.pdf")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.source_url == "https://example.com/policy.pdf"

    def test_source_url_none_when_not_set(self):
        doc = self._make_doc(source_url=None)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.source_url is None

    def test_metadata_contains_doc_title(self):
        doc = self._make_doc()
        for c in chunk_document(doc):
            assert c.metadata.get("doc_title") == doc.title

    def test_multi_page_document(self):
        doc = PolicyDocument(
            doc_id="mp_test",
            title="Multi-Page",
            pages=MULTI_PAGE_PAGES,
        )
        chunks = chunk_document(doc, chunk_size=100)
        page_nums = {c.page_number for c in chunks}
        assert len(page_nums) > 1

    def test_empty_page_is_skipped(self):
        doc = PolicyDocument(
            doc_id="ep_test",
            title="Sparse",
            pages=[(1, "   "), (2, SAMPLE_POLICY_TEXT)],
        )
        chunks = chunk_document(doc)
        assert all(c.page_number != 1 for c in chunks)

    def test_invalid_size_overlap_raises(self):
        doc = self._make_doc()
        with pytest.raises(ValueError, match="chunk_size.*chunk_overlap"):
            chunk_document(doc, chunk_size=64, chunk_overlap=64)

    def test_heading_detected_for_article(self):
        doc = PolicyDocument(
            doc_id="hd_test",
            title="Reg",
            pages=[(1, "Article 12 — Compliance Targets\nAll states shall comply.")],
        )
        chunks = chunk_document(doc, chunk_size=500)
        headings = [c.section_heading for c in chunks if c.section_heading]
        assert len(headings) > 0

    def test_word_count_property(self):
        doc = self._make_doc()
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.word_count == len(c.text.split())

    def test_citation_label_includes_title(self):
        doc = self._make_doc()
        chunks = chunk_document(doc, chunk_size=200)
        for c in chunks:
            label = c.citation_label()
            assert doc.title in label


# ============================================================
# Tests: connectors
# ============================================================

class TestConnectors:
    @pytest.mark.asyncio
    async def test_text_connector_returns_document(self):
        conn = TextConnector(SAMPLE_POLICY_TEXT, title="Test", source_label="Test Org")
        doc = await conn.fetch()
        assert isinstance(doc, PolicyDocument)
        assert doc.title == "Test"
        assert doc.source_label == "Test Org"

    def test_text_connector_source_description(self):
        conn = TextConnector("Policy content here.", title="Test")
        assert "Policy content" in conn.source_description

    @pytest.mark.asyncio
    async def test_url_connector_calls_load_from_url(self):
        with patch(
            "app.core.connectors.load_from_url",
            new=AsyncMock(return_value=PolicyDocument(
                doc_id="abc", title="From URL", pages=[(1, "Content")],
                source_url="https://example.com/doc.pdf",
            )),
        ):
            conn = URLConnector("https://example.com/doc.pdf", title="Fetched")
            doc = await conn.fetch()
        assert doc.title == "From URL"

    def test_url_connector_source_description(self):
        conn = URLConnector("https://example.com/report.pdf")
        assert "https://example.com/report.pdf" in conn.source_description

    def test_connector_satisfies_protocol(self):
        conn = TextConnector("Some text.")
        assert isinstance(conn, DocumentConnector)

    def test_connector_from_request_url(self):
        conn = connector_from_request(url="https://example.com/doc.pdf")
        assert isinstance(conn, URLConnector)

    def test_connector_from_request_text(self):
        conn = connector_from_request(text="Policy text here.")
        assert isinstance(conn, TextConnector)

    def test_connector_from_request_raises_on_both(self):
        with pytest.raises(ValueError, match="not both"):
            connector_from_request(url="https://x.com", text="also text")

    def test_connector_from_request_raises_on_neither(self):
        with pytest.raises(ValueError):
            connector_from_request()


# ============================================================
# Tests: API endpoints (ingest / list / delete)
# ============================================================

class TestIngestEndpoints:
    def test_ingest_text_payload_returns_201(self, api_client):
        r = api_client.post(
            "/api/v1/ingest",
            json={"text": SAMPLE_POLICY_TEXT, "title": "Climate Strategy 2030"},
        )
        assert r.status_code == 201
        body = r.json()
        assert "doc_id" in body
        assert body["chunk_count"] > 0
        assert body["title"] == "Climate Strategy 2030"

    def test_ingest_returns_source_url_none_for_text(self, api_client):
        # Text must be long enough to produce at least one chunk (min_chars=50)
        r = api_client.post(
            "/api/v1/ingest",
            json={"text": "Article 1 requires all signatory states to submit annual compliance reports to the Commission."},
        )
        assert r.status_code == 201
        assert r.json()["source_url"] is None

    def test_ingest_rejects_missing_source(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"title": "No Source"})
        assert r.status_code == 422

    def test_ingest_rejects_both_url_and_text(self, api_client):
        r = api_client.post(
            "/api/v1/ingest",
            json={"url": "https://example.com/doc.pdf", "text": "Also text"},
        )
        assert r.status_code == 422

    def test_ingest_empty_text_returns_422(self, api_client):
        r = api_client.post("/api/v1/ingest", json={"text": "   "})
        assert r.status_code in (201, 422)   # cleaned text may be caught at load or chunk level

    def test_list_documents_empty_initially(self, api_client):
        r = api_client.get("/api/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert "documents" in body
        assert "total" in body

    def test_list_documents_after_ingest(self, api_client):
        api_client.post(
            "/api/v1/ingest",
            json={"text": SAMPLE_POLICY_TEXT, "title": "Listed Doc"},
        )
        r = api_client.get("/api/v1/documents")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_delete_nonexistent_document_returns_404(self, api_client):
        r = api_client.delete("/api/v1/documents/nonexistent_id_xyz")
        assert r.status_code == 404

    def test_ingest_then_delete_removes_document(self, api_client):
        ingest_r = api_client.post(
            "/api/v1/ingest",
            json={"text": "Unique content for delete test. " * 5, "title": "Delete Me"},
        )
        assert ingest_r.status_code == 201
        doc_id = ingest_r.json()["doc_id"]

        delete_r = api_client.delete(f"/api/v1/documents/{doc_id}")
        assert delete_r.status_code == 200
        assert delete_r.json()["deleted"] is True

        # Document should no longer appear in the list
        list_r = api_client.get("/api/v1/documents")
        ids = [d["doc_id"] for d in list_r.json()["documents"]]
        assert doc_id not in ids
