"""Tests for services/graph_service.py — InMemoryGraphService.

Covers: entity CRUD, relation CRUD, neighbour traversal,
edge retrieval (in + out), duplicate handling, and count methods.
"""

from __future__ import annotations

import pytest

from app.core.entity_extraction import Entity, Relation
from app.services.graph_service import (
    GraphEdge,
    GraphNode,
    InMemoryGraphService,
    build_graph_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entity(name: str, label: str = "ORG", doc_id: str = "doc1") -> Entity:
    return Entity(
        text=name, label=label, doc_id=doc_id, page_number=1,
        chunk_id="chunk1", char_start=0, char_end=len(name),
    )


def _relation(src: str, tgt: str, rtype: str = "RELATED_TO", doc_id: str = "doc1") -> Relation:
    return Relation(source=src, relation_type=rtype, target=tgt, doc_id=doc_id, chunk_id="chunk1")


@pytest.fixture()
def graph() -> InMemoryGraphService:
    return InMemoryGraphService()


# ---------------------------------------------------------------------------
# Initialisation / factory
# ---------------------------------------------------------------------------

def test_build_graph_service_memory():
    gs = build_graph_service(provider="memory")
    assert isinstance(gs, InMemoryGraphService)


def test_build_graph_service_unknown_raises():
    with pytest.raises(ValueError, match="Unknown graph provider"):
        build_graph_service(provider="unknown_db")


def test_initial_counts_are_zero(graph):
    assert graph.entity_count() == 0
    assert graph.relation_count() == 0


# ---------------------------------------------------------------------------
# add_entity
# ---------------------------------------------------------------------------

def test_add_entity_increments_count(graph):
    graph.add_entity(_entity("European Commission"))
    assert graph.entity_count() == 1


def test_add_multiple_entities(graph):
    graph.add_entity(_entity("European Commission"))
    graph.add_entity(_entity("Germany", label="GPE"))
    assert graph.entity_count() == 2


def test_add_duplicate_entity_does_not_duplicate(graph):
    ent = _entity("European Commission")
    graph.add_entity(ent)
    graph.add_entity(ent)
    assert graph.entity_count() == 1


def test_add_entity_with_different_doc_ids_still_one_node(graph):
    graph.add_entity(_entity("European Commission", doc_id="doc1"))
    graph.add_entity(_entity("European Commission", doc_id="doc2"))
    assert graph.entity_count() == 1


# ---------------------------------------------------------------------------
# add_relation
# ---------------------------------------------------------------------------

def test_add_relation_increments_count(graph):
    graph.add_entity(_entity("European Commission"))
    graph.add_entity(_entity("Germany", label="GPE"))
    graph.add_relation(_relation("European Commission", "Germany", "OPERATES_IN"))
    assert graph.relation_count() == 1


def test_add_relation_auto_creates_missing_nodes(graph):
    graph.add_relation(_relation("European Commission", "Germany"))
    assert graph.entity_count() == 2


def test_add_multiple_relations(graph):
    graph.add_relation(_relation("A", "B", "REL1"))
    graph.add_relation(_relation("B", "C", "REL2"))
    assert graph.relation_count() == 2


def test_add_relation_with_zero_confidence(graph):
    rel = Relation(source="A", relation_type="WEAK", target="B",
                   doc_id="doc1", chunk_id="c1", confidence=0.0)
    graph.add_relation(rel)
    edges = graph.get_edges("A")
    assert any(e.confidence == 0.0 for e in edges)


# ---------------------------------------------------------------------------
# get_edges
# ---------------------------------------------------------------------------

def test_get_edges_out_edges(graph):
    graph.add_relation(_relation("European Commission", "Germany", "OPERATES_IN"))
    edges = graph.get_edges("European Commission")
    assert any(e.source == "European Commission" and e.target == "Germany" for e in edges)


def test_get_edges_in_edges(graph):
    graph.add_relation(_relation("European Commission", "Germany", "OPERATES_IN"))
    edges = graph.get_edges("Germany")
    # Germany should appear as target in an in-edge
    assert any(e.target == "Germany" for e in edges)


def test_get_edges_unknown_entity_returns_empty(graph):
    assert graph.get_edges("Nonexistent Entity") == []


def test_get_edges_returns_graph_edge_objects(graph):
    graph.add_relation(_relation("A", "B"))
    edges = graph.get_edges("A")
    assert all(isinstance(e, GraphEdge) for e in edges)


def test_get_edges_relation_type_preserved(graph):
    graph.add_relation(_relation("European Commission", "Germany", "OPERATES_IN"))
    edges = graph.get_edges("European Commission")
    out_edges = [e for e in edges if e.source == "European Commission"]
    assert any(e.relation == "OPERATES_IN" for e in out_edges)


def test_get_edges_doc_id_preserved(graph):
    graph.add_relation(_relation("A", "B", doc_id="my_doc"))
    edges = graph.get_edges("A")
    assert any(e.doc_id == "my_doc" for e in edges)


# ---------------------------------------------------------------------------
# get_neighbours
# ---------------------------------------------------------------------------

def test_get_neighbours_depth_1(graph):
    graph.add_relation(_relation("A", "B"))
    graph.add_relation(_relation("B", "C"))
    neighbours = graph.get_neighbours("A", depth=1)
    names = {n.name for n in neighbours}
    assert "B" in names
    assert "C" not in names   # depth-1 should not reach C


def test_get_neighbours_depth_2(graph):
    graph.add_relation(_relation("A", "B"))
    graph.add_relation(_relation("B", "C"))
    neighbours = graph.get_neighbours("A", depth=2)
    names = {n.name for n in neighbours}
    assert "B" in names
    assert "C" in names


def test_get_neighbours_unknown_returns_empty(graph):
    result = graph.get_neighbours("UnknownEntity", depth=1)
    assert result == []


def test_get_neighbours_returns_graph_node_objects(graph):
    graph.add_relation(_relation("A", "B"))
    neighbours = graph.get_neighbours("A", depth=1)
    assert all(isinstance(n, GraphNode) for n in neighbours)


def test_get_neighbours_excludes_self(graph):
    graph.add_relation(_relation("A", "B"))
    neighbours = graph.get_neighbours("A", depth=1)
    names = {n.name for n in neighbours}
    assert "A" not in names


# ---------------------------------------------------------------------------
# entity_count / relation_count
# ---------------------------------------------------------------------------

def test_entity_count_matches_added(graph):
    for i in range(5):
        graph.add_entity(_entity(f"Entity {i}"))
    assert graph.entity_count() == 5


def test_relation_count_matches_added(graph):
    for i in range(4):
        graph.add_relation(_relation(f"E{i}", f"E{i + 1}"))
    assert graph.relation_count() == 4


# ---------------------------------------------------------------------------
# Integration: ingest → query graph
# ---------------------------------------------------------------------------

def test_full_entity_relation_roundtrip(graph):
    ec = _entity("European Commission", label="ORG")
    de = _entity("Germany", label="GPE")
    rel = _relation("European Commission", "Germany", "OPERATES_IN")

    graph.add_entity(ec)
    graph.add_entity(de)
    graph.add_relation(rel)

    assert graph.entity_count() == 2
    assert graph.relation_count() == 1

    edges = graph.get_edges("European Commission")
    assert any(
        e.source == "European Commission"
        and e.target == "Germany"
        and e.relation == "OPERATES_IN"
        for e in edges
    )

    neighbours = graph.get_neighbours("European Commission", depth=1)
    assert any(n.name == "Germany" for n in neighbours)
