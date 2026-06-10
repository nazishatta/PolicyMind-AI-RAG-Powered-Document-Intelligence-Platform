"""Unit tests for core/connectors.py.

All tests use in-memory fixtures only — no network calls are made.
URLConnector network behaviour is covered by integration tests that
use httpx's built-in test transport when needed.
"""

from __future__ import annotations

import pytest

from app.core.connectors import (
    DocumentConnector,
    TextConnector,
    URLConnector,
    connector_from_request,
)
from app.core.document_loader import (
    DocumentEmptyError,
    DocumentSizeError,
    PolicyDocument,
)


# ---------------------------------------------------------------------------
# TextConnector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_connector_returns_document():
    conn = TextConnector(text="Sustainable development requires inclusive governance.", title="Test")
    doc = await conn.fetch()
    assert isinstance(doc, PolicyDocument)
    assert doc.title == "Test"
    assert "inclusive governance" in doc.full_text


@pytest.mark.asyncio
async def test_text_connector_auto_title():
    conn = TextConnector(text="The World Bank provides development finance to member countries.")
    doc = await conn.fetch()
    assert len(doc.title) > 0


@pytest.mark.asyncio
async def test_text_connector_empty_raises():
    conn = TextConnector(text="")
    with pytest.raises(DocumentEmptyError):
        await conn.fetch()


@pytest.mark.asyncio
async def test_text_connector_whitespace_only_raises():
    conn = TextConnector(text="   \n\n   ")
    with pytest.raises(DocumentEmptyError):
        await conn.fetch()


@pytest.mark.asyncio
async def test_text_connector_oversized_raises():
    conn = TextConnector(text="x" * 500_001)
    with pytest.raises(DocumentSizeError):
        await conn.fetch()


@pytest.mark.asyncio
async def test_text_connector_source_label_stored():
    conn = TextConnector(
        text="The UN Sustainable Development Goals guide global development.",
        source_label="UNDP",
    )
    doc = await conn.fetch()
    assert doc.source_label == "UNDP"


@pytest.mark.asyncio
async def test_text_connector_source_url_is_none():
    conn = TextConnector(text="Some policy content here.")
    doc = await conn.fetch()
    assert doc.source_url is None


# ---------------------------------------------------------------------------
# URLConnector construction (no network)
# ---------------------------------------------------------------------------

def test_url_connector_stores_url():
    conn = URLConnector(url="https://example.org/policy.pdf")
    assert conn.url == "https://example.org/policy.pdf"


def test_url_connector_stores_max_bytes():
    conn = URLConnector(url="https://example.org/doc.pdf", max_bytes=10 * 1024 * 1024)
    assert conn.max_bytes == 10 * 1024 * 1024


def test_url_connector_source_description():
    conn = URLConnector(url="https://example.org/policy.pdf")
    assert "https://example.org/policy.pdf" in conn.source_description


# ---------------------------------------------------------------------------
# connector_from_request factory
# ---------------------------------------------------------------------------

def test_factory_url_returns_url_connector():
    conn = connector_from_request(url="https://example.org/doc.pdf")
    assert isinstance(conn, URLConnector)


def test_factory_text_returns_text_connector():
    conn = connector_from_request(text="Policy content here.")
    assert isinstance(conn, TextConnector)


def test_factory_both_raises():
    with pytest.raises(ValueError):
        connector_from_request(url="https://example.org/doc.pdf", text="Some text")


def test_factory_neither_raises():
    with pytest.raises(ValueError):
        connector_from_request()


def test_factory_url_passes_max_bytes():
    conn = connector_from_request(url="https://example.org/doc.pdf", max_bytes=5 * 1024 * 1024)
    assert isinstance(conn, URLConnector)
    assert conn.max_bytes == 5 * 1024 * 1024


def test_factory_passes_title_and_label_to_url():
    conn = connector_from_request(
        url="https://example.org/doc.pdf",
        title="Custom Title",
        source_label="EUR-Lex",
    )
    assert isinstance(conn, URLConnector)
    assert conn.title == "Custom Title"
    assert conn.source_label == "EUR-Lex"


def test_factory_passes_title_and_label_to_text():
    conn = connector_from_request(
        text="Some policy content.",
        title="Custom Title",
        source_label="World Bank",
    )
    assert isinstance(conn, TextConnector)
    assert conn.title == "Custom Title"
    assert conn.source_label == "World Bank"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_text_connector_satisfies_protocol():
    conn = TextConnector(text="Test")
    assert isinstance(conn, DocumentConnector)


def test_url_connector_satisfies_protocol():
    conn = URLConnector(url="https://example.org/doc.pdf")
    assert isinstance(conn, DocumentConnector)
