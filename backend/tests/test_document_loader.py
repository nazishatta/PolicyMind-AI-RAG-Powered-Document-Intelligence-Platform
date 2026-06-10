"""Tests for core/document_loader.py — basic happy-path coverage.

Extended error-case coverage is in test_ingestion.py.
"""

import pytest

from app.core.document_loader import (
    DocumentEmptyError,
    DocumentSizeError,
    PolicyDocument,
    load_from_text,
)


def test_load_from_text_returns_document():
    doc = load_from_text("Climate targets include 55% by 2030.", title="Test Doc")
    assert isinstance(doc, PolicyDocument)
    assert doc.title == "Test Doc"
    assert doc.page_count == 1
    assert "55%" in doc.full_text


def test_load_from_text_auto_title():
    doc = load_from_text("The World Bank provides development finance.", title=None)
    assert len(doc.title) > 0


def test_load_from_text_assigns_stable_doc_id():
    text = "Consistent text for hashing."
    doc1 = load_from_text(text)
    doc2 = load_from_text(text)
    assert doc1.doc_id == doc2.doc_id


def test_load_from_text_different_content_different_ids():
    doc1 = load_from_text("First document text.")
    doc2 = load_from_text("Second document text.")
    assert doc1.doc_id != doc2.doc_id


def test_full_text_property(sample_document):
    assert len(sample_document.full_text) > 0
    assert "Climate" in sample_document.full_text


def test_page_count(sample_document):
    assert sample_document.page_count == 1


def test_total_chars(sample_document):
    assert sample_document.total_chars > 0


def test_empty_text_raises():
    with pytest.raises(DocumentEmptyError):
        load_from_text("")


def test_oversized_text_raises():
    with pytest.raises(DocumentSizeError):
        load_from_text("x" * 500_001)


def test_source_label_stored():
    doc = load_from_text("Some policy content.", source_label="World Bank")
    assert doc.source_label == "World Bank"


def test_source_url_none_for_text():
    doc = load_from_text("Some policy content.")
    assert doc.source_url is None


def test_metadata_source_field():
    doc = load_from_text("Some content.")
    assert doc.metadata.get("source") == "raw_text"
