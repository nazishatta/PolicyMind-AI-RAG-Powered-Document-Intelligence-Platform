"""Tests for core/citation_engine.py — CitationEngine.

Covers: record accumulation, schema export, APA formatting,
excerpt truncation, score rounding, and clear().
"""

from __future__ import annotations

import pytest

from app.core.citation_engine import CitationEngine, CitationRecord
from app.schemas.response import Citation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _engine_with_two_records() -> CitationEngine:
    engine = CitationEngine()
    engine.record(
        chunk_id="doc1_p0001_c000000",
        doc_id="doc1abc",
        doc_title="National Climate Strategy 2030",
        page_number=1,
        excerpt="Member States shall reduce emissions by 55 percent.",
        relevance_score=0.91234,
        section_heading="Article 12",
    )
    engine.record(
        chunk_id="doc1_p0002_c000001",
        doc_id="doc1abc",
        doc_title="National Climate Strategy 2030",
        page_number=2,
        excerpt="The Just Transition Fund allocates EUR 17.5 billion.",
        relevance_score=0.87,
    )
    return engine


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------

def test_record_accumulates():
    engine = CitationEngine()
    assert engine.to_schema() == []
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc A",
        page_number=1, excerpt="Some text.", relevance_score=0.8,
    )
    assert len(engine.to_schema()) == 1


def test_record_multiple():
    engine = _engine_with_two_records()
    assert len(engine.to_schema()) == 2


def test_record_excerpt_truncated_at_300():
    engine = CitationEngine()
    long_text = "x" * 500
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc",
        page_number=1, excerpt=long_text, relevance_score=0.5,
    )
    schema = engine.to_schema()
    assert len(schema[0].excerpt) == 300


def test_record_excerpt_under_300_kept_fully():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc",
        page_number=1, excerpt="Short text.", relevance_score=0.5,
    )
    assert engine.to_schema()[0].excerpt == "Short text."


def test_record_score_rounded_to_4dp():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc",
        page_number=1, excerpt="Text.", relevance_score=0.912345678,
    )
    assert engine.to_schema()[0].relevance_score == round(0.912345678, 4)


def test_record_section_heading_stored():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc",
        page_number=1, excerpt="Text.", relevance_score=0.5,
        section_heading="Article 5",
    )
    assert engine.to_schema()[0].section_heading == "Article 5"


def test_record_section_heading_defaults_none():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Doc",
        page_number=1, excerpt="Text.", relevance_score=0.5,
    )
    assert engine.to_schema()[0].section_heading is None


# ---------------------------------------------------------------------------
# to_schema()
# ---------------------------------------------------------------------------

def test_to_schema_returns_citation_objects():
    engine = _engine_with_two_records()
    schema = engine.to_schema()
    assert all(isinstance(c, Citation) for c in schema)


def test_to_schema_maps_all_fields():
    engine = CitationEngine()
    engine.record(
        chunk_id="chunk_xyz",
        doc_id="doc_abc",
        doc_title="World Development Report",
        page_number=7,
        excerpt="Poverty rates declined in 2020.",
        relevance_score=0.75,
        section_heading="Chapter 3",
    )
    c = engine.to_schema()[0]
    assert c.chunk_id == "chunk_xyz"
    assert c.doc_id == "doc_abc"
    assert c.doc_title == "World Development Report"
    assert c.page_number == 7
    assert c.excerpt == "Poverty rates declined in 2020."
    assert c.relevance_score == 0.75
    assert c.section_heading == "Chapter 3"


def test_to_schema_empty_engine():
    assert CitationEngine().to_schema() == []


def test_to_schema_preserves_order():
    engine = CitationEngine()
    for i in range(5):
        engine.record(
            chunk_id=f"c{i}", doc_id="d", doc_title="D",
            page_number=i + 1, excerpt=f"Text {i}", relevance_score=float(i) / 10,
        )
    schema = engine.to_schema()
    pages = [c.page_number for c in schema]
    assert pages == list(range(1, 6))


# ---------------------------------------------------------------------------
# to_apa()
# ---------------------------------------------------------------------------

def test_to_apa_returns_strings():
    engine = _engine_with_two_records()
    apa = engine.to_apa()
    assert len(apa) == 2
    assert all(isinstance(s, str) for s in apa)


def test_to_apa_contains_doc_title():
    engine = _engine_with_two_records()
    for entry in engine.to_apa():
        assert "National Climate Strategy 2030" in entry


def test_to_apa_contains_page_number():
    engine = _engine_with_two_records()
    apa = engine.to_apa()
    assert "p. 1" in apa[0]
    assert "p. 2" in apa[1]


def test_to_apa_includes_section_heading():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Strategy",
        page_number=3, excerpt="Text.", relevance_score=0.6,
        section_heading="Article 5",
    )
    assert "Article 5" in engine.to_apa()[0]


def test_to_apa_no_heading_omits_comma():
    engine = CitationEngine()
    engine.record(
        chunk_id="c1", doc_id="d1", doc_title="Strategy",
        page_number=3, excerpt="Text.", relevance_score=0.6,
    )
    apa = engine.to_apa()[0]
    # Should not have a leading comma between title and page ref
    assert ", p. 3" in apa


def test_to_apa_empty_engine():
    assert CitationEngine().to_apa() == []


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_empties_records():
    engine = _engine_with_two_records()
    engine.clear()
    assert engine.to_schema() == []


def test_clear_allows_new_records():
    engine = _engine_with_two_records()
    engine.clear()
    engine.record(
        chunk_id="c_new", doc_id="d_new", doc_title="New Doc",
        page_number=1, excerpt="Fresh content.", relevance_score=0.9,
    )
    assert len(engine.to_schema()) == 1
