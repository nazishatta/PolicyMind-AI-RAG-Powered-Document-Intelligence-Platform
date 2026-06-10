"""Tests for core/text_chunker.py — basic coverage.

Extended chunker tests (multi-page, heading detection, source_url) are in test_ingestion.py.
"""

import pytest

from app.core.document_loader import PolicyDocument
from app.core.text_chunker import TextChunk, chunk_document


def test_chunk_document_returns_chunks(sample_document):
    chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_ids_are_unique(sample_document):
    chunks = chunk_document(sample_document, chunk_size=200)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_text_is_non_empty(sample_document):
    chunks = chunk_document(sample_document)
    assert all(len(c.text.strip()) > 0 for c in chunks)


def test_chunk_inherits_doc_id(sample_document):
    chunks = chunk_document(sample_document)
    for c in chunks:
        assert c.doc_id == sample_document.doc_id


def test_chunk_page_numbers_valid(sample_document):
    chunks = chunk_document(sample_document)
    for c in chunks:
        assert c.page_number >= 1


def test_min_chars_filters_short_chunks():
    doc = PolicyDocument(
        doc_id="tiny",
        title="Tiny",
        pages=[(1, "A" * 10 + "\n\n" + "B" * 300)],
    )
    chunks = chunk_document(doc, chunk_size=200, min_chars=50)
    assert all(len(c.text) >= 50 for c in chunks)


def test_chunk_metadata_contains_title(sample_document):
    chunks = chunk_document(sample_document)
    for c in chunks:
        assert c.metadata.get("doc_title") == sample_document.title


def test_source_url_propagated_to_chunks():
    doc = PolicyDocument(
        doc_id="url_test",
        title="URL Doc",
        pages=[(1, "Policy content for URL test. " * 20)],
        source_url="https://example.org/policy.pdf",
    )
    chunks = chunk_document(doc)
    for c in chunks:
        assert c.source_url == "https://example.org/policy.pdf"


def test_invalid_overlap_raises():
    doc = PolicyDocument(doc_id="x", title="X", pages=[(1, "Some text.")])
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=64, chunk_overlap=64)


def test_word_count_property(sample_document):
    chunks = chunk_document(sample_document)
    for c in chunks:
        assert c.word_count == len(c.text.split())


def test_char_count_property(sample_document):
    chunks = chunk_document(sample_document)
    for c in chunks:
        assert c.char_count == len(c.text)
