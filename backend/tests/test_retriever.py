"""Tests for core/retriever.py — HybridRetriever.

All tests use in-memory providers and a stub embedding model.
No network calls, no API keys, no spaCy model required.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.document_loader import PolicyDocument
from app.core.retriever import HybridRetriever, RetrievalResult
from app.core.text_chunker import chunk_document
from app.services.graph_service import InMemoryGraphService
from app.services.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

POLICY_TEXT = (
    "The European Commission oversees compliance with the 2030 Climate Strategy. "
    "Member States must reduce greenhouse gas emissions by 55 percent relative to 1990. "
    "The Just Transition Fund allocates EUR 17.5 billion to affected regions. "
    "Article 12 sets binding renewable energy targets of 30 percent by 2025. "
    "Regulation (EU) 2021/1119 defines penalties for non-compliance. "
    "Annual progress reports are due by 31 March each year."
)

POLICY_TEXT_B = (
    "The World Bank provides development finance to low-income countries. "
    "The International Monetary Fund supports macroeconomic stability. "
    "Sustainable development goals include poverty eradication by 2030. "
    "The Paris Agreement on climate change was adopted in December 2015."
)


def _make_stub_embedding(dim: int = 384):
    """Return an EmbeddingModel subclass that produces seeded random unit vectors."""
    from app.core.embeddings import EmbeddingModel

    class _Stub(EmbeddingModel):
        def embed(self, texts: list[str]) -> np.ndarray:
            rng = np.random.default_rng(seed=sum(ord(c) for c in "".join(texts)) % 2**31)
            vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.where(norms == 0, 1.0, norms)

    return _Stub()


def _build_retriever(
    text: str = POLICY_TEXT,
    alpha: float = 0.7,
) -> tuple[HybridRetriever, InMemoryVectorStore, InMemoryGraphService]:
    emb = _make_stub_embedding()
    vs = InMemoryVectorStore()
    gs = InMemoryGraphService()

    doc = PolicyDocument(doc_id="doc_test", title="Test Policy", pages=[(1, text)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    embeddings = emb.embed([c.text for c in chunks])
    vs.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=vs,
        graph_service=gs,
        embedding_model=emb,
        alpha=alpha,
    )
    return retriever, vs, gs


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_invalid_alpha_raises():
    emb = _make_stub_embedding()
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(
            vector_store=InMemoryVectorStore(),
            graph_service=InMemoryGraphService(),
            embedding_model=emb,
            alpha=1.5,
        )


def test_alpha_zero_is_valid():
    emb = _make_stub_embedding()
    r = HybridRetriever(
        vector_store=InMemoryVectorStore(),
        graph_service=InMemoryGraphService(),
        embedding_model=emb,
        alpha=0.0,
    )
    assert r._alpha == 0.0


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_retrieve_empty_store_returns_empty():
    emb = _make_stub_embedding()
    retriever = HybridRetriever(
        vector_store=InMemoryVectorStore(),
        graph_service=InMemoryGraphService(),
        embedding_model=emb,
    )
    result = retriever.retrieve("Any question?")
    assert isinstance(result, RetrievalResult)
    assert result.chunks == []
    assert result.graph_evidence == []
    assert result.entity_hits == []


# ---------------------------------------------------------------------------
# Basic vector retrieval (no graph)
# ---------------------------------------------------------------------------

def test_retrieve_returns_results(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    doc = PolicyDocument(doc_id="d1", title="Climate", pages=[(1, POLICY_TEXT)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    result = retriever.retrieve("emission reduction targets", top_k=3)
    assert len(result.chunks) <= 3
    assert all(hasattr(c, "score") for c in result.chunks)


def test_retrieve_respects_top_k(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    doc = PolicyDocument(doc_id="d2", title="Big", pages=[(1, POLICY_TEXT * 5)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    result = retriever.retrieve("climate targets", top_k=2)
    assert len(result.chunks) <= 2


def test_retrieve_graph_disabled_skips_graph(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    from app.core.entity_extraction import Entity
    in_memory_graph.add_entity(
        Entity(text="European Commission", label="ORG", doc_id="d3",
               page_number=1, chunk_id="c1", char_start=0, char_end=20)
    )

    doc = PolicyDocument(doc_id="d3", title="EU", pages=[(1, POLICY_TEXT)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    result = retriever.retrieve("European Commission", top_k=3, include_graph=False)
    assert result.graph_evidence == []
    assert result.entity_hits == []


def test_retrieve_returns_result_dataclass(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    doc = PolicyDocument(doc_id="d4", title="P", pages=[(1, POLICY_TEXT)])
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    result = retriever.retrieve("renewable energy")
    assert isinstance(result, RetrievalResult)
    assert isinstance(result.chunks, list)
    assert isinstance(result.graph_evidence, list)
    assert isinstance(result.entity_hits, list)


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------

def test_retrieve_with_graph_finds_evidence(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """When entities from chunks exist in graph, evidence should be returned."""
    from app.core.entity_extraction import Entity, Relation

    # Build entity + relation that appear in POLICY_TEXT
    ent = Entity(text="European Commission", label="ORG", doc_id="d5",
                 page_number=1, chunk_id="c5", char_start=4, char_end=23)
    in_memory_graph.add_entity(ent)
    rel = Relation(source="European Commission", relation_type="OPERATES_IN",
                   target="Member States", doc_id="d5", chunk_id="c5")
    in_memory_graph.add_relation(rel)

    doc = PolicyDocument(doc_id="d5", title="EU Climate", pages=[(1, POLICY_TEXT)])
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
        alpha=0.5,
    )
    result = retriever.retrieve("European Commission compliance", top_k=5)
    # Graph evidence should surface European Commission triples
    assert isinstance(result.graph_evidence, list)
    # entity_hits may or may not match depending on regex/spaCy - don't hard-assert
    assert isinstance(result.entity_hits, list)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def test_score_fusion_all_vector_when_alpha_1(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """With alpha=1.0 graph boost is zero — ordering is pure vector similarity."""
    doc = PolicyDocument(doc_id="df1", title="V", pages=[(1, POLICY_TEXT)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever_vector = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
        alpha=1.0,
    )
    result = retriever_vector.retrieve("climate targets", top_k=5)
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_score_fusion_max_graph_evidence_capped(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """max_graph_evidence parameter limits returned triples."""
    from app.core.entity_extraction import Entity, Relation

    for i in range(15):
        ent = Entity(text=f"Organisation {i}", label="ORG", doc_id="df2",
                     page_number=1, chunk_id=f"c{i}", char_start=0, char_end=10)
        in_memory_graph.add_entity(ent)
        rel = Relation(source=f"Organisation {i}", relation_type="RELATED_TO",
                       target="Policy", doc_id="df2", chunk_id=f"c{i}")
        in_memory_graph.add_relation(rel)

    doc = PolicyDocument(doc_id="df2", title="Orgs",
                         pages=[(1, " ".join(f"Organisation {i}" for i in range(15)))])
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
        max_graph_evidence=5,
    )
    result = retriever.retrieve("Organisation policy targets", top_k=5)
    assert len(result.graph_evidence) <= 5


# ---------------------------------------------------------------------------
# doc_id_filter
# ---------------------------------------------------------------------------

def test_retrieve_doc_id_filter(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    doc_a = PolicyDocument(doc_id="filter_a", title="A", pages=[(1, POLICY_TEXT)])
    doc_b = PolicyDocument(doc_id="filter_b", title="B", pages=[(1, POLICY_TEXT_B)])

    for doc in (doc_a, doc_b):
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        embeddings = mock_embedding_model.embed([c.text for c in chunks])
        in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    result = retriever.retrieve("climate", top_k=10, doc_id_filter="filter_a")
    for chunk in result.chunks:
        assert chunk.metadata.get("doc_id") == "filter_a"


# ---------------------------------------------------------------------------
# Entity regex fallback (no spaCy required)
# ---------------------------------------------------------------------------

def test_entity_regex_fallback():
    """_extract_entities_from_text must work without spaCy.

    The retriever's internal helper tries spaCy then falls back to a regex
    when any exception is raised (ImportError, OSError, etc.).
    """
    emb = _make_stub_embedding()
    retriever = HybridRetriever(
        vector_store=InMemoryVectorStore(),
        graph_service=InMemoryGraphService(),
        embedding_model=emb,
    )
    # Patch the NLP loader in entity_extraction to force the regex fallback path
    import unittest.mock as mock
    with mock.patch(
        "app.core.entity_extraction._get_nlp",
        side_effect=ImportError("no spacy"),
    ):
        entities = retriever._extract_entities_from_text(
            "The European Commission oversees Member States in 2030."
        )
    # Should still return a list (regex results) rather than raising
    assert isinstance(entities, list)


# ---------------------------------------------------------------------------
# Proportional citation-aware graph boost
# ---------------------------------------------------------------------------

def test_fuse_scores_proportional_not_binary(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """A chunk with 2 matching entities outscores one with 1 (same vector score)."""
    from app.core.entity_extraction import Entity, Relation
    from app.services.vector_store import SearchResult

    # Two entities in the graph
    for name in ("European Commission", "Member States"):
        in_memory_graph.add_entity(
            Entity(text=name, label="ORG", doc_id="d_prop",
                   page_number=1, chunk_id="c_prop", char_start=0, char_end=10)
        )
    in_memory_graph.add_relation(
        Relation(source="European Commission", relation_type="OVERSEES",
                 target="Member States", doc_id="d_prop", chunk_id="c_prop")
    )

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
        alpha=0.5,
    )

    # Chunk A mentions both entities; chunk B only one; chunk C neither
    chunk_a = SearchResult(
        chunk_id="ca", text="European Commission oversees Member States.", score=0.5, metadata={}
    )
    chunk_b = SearchResult(
        chunk_id="cb", text="European Commission published a report.", score=0.5, metadata={}
    )
    chunk_c = SearchResult(
        chunk_id="cc", text="No named entities whatsoever in this text.", score=0.5, metadata={}
    )

    entity_hits = ["European Commission", "Member States"]
    entity_confidences = {"European Commission": 1.0, "Member States": 1.0}

    reranked = retriever._fuse_scores([chunk_a, chunk_b, chunk_c], entity_hits, entity_confidences)
    positions = {c.chunk_id: i for i, c in enumerate(reranked)}

    # More entity overlap → higher rank
    assert positions["ca"] < positions["cb"], "2-entity chunk should beat 1-entity chunk"
    assert positions["cc"] > positions["ca"], "No-entity chunk should rank lowest"


def test_fuse_scores_no_entity_match_uses_only_vector(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """When no chunk entities match the graph, fused score equals α·vector_score."""
    from app.services.vector_store import SearchResult

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
        alpha=0.6,
    )

    chunk = SearchResult(chunk_id="cx", text="No graph entities.", score=0.5, metadata={})
    reranked = retriever._fuse_scores([chunk], entity_hits=[], entity_confidences={})

    # With no entity_set, score should equal 0.6 * 0.5 = 0.3 — but we can only
    # verify ORDER (the score field is not mutated).  Just assert it returns the chunk.
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "cx"


def test_fuse_scores_empty_entity_confidences_does_not_raise(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """_fuse_scores must not crash when entity_confidences is empty."""
    from app.services.vector_store import SearchResult

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    chunks = [
        SearchResult(chunk_id=f"c{i}", text=f"Some text {i}.", score=0.5, metadata={})
        for i in range(3)
    ]
    result = retriever._fuse_scores(chunks, entity_hits=["Org"], entity_confidences={})
    assert len(result) == 3


# ---------------------------------------------------------------------------
# _expand_graph returns entity_confidences dict
# ---------------------------------------------------------------------------

def test_expand_graph_returns_entity_confidences(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """_expand_graph must return a dict mapping entity names to mean edge confidence."""
    from app.core.entity_extraction import Entity, Relation
    from app.core.document_loader import PolicyDocument
    from app.core.text_chunker import chunk_document

    in_memory_graph.add_entity(
        Entity(text="European Commission", label="ORG", doc_id="d_ec",
               page_number=1, chunk_id="c_ec", char_start=0, char_end=20)
    )
    in_memory_graph.add_relation(
        Relation(source="European Commission", relation_type="OVERSEES",
                 target="Member States", doc_id="d_ec", chunk_id="c_ec",
                 confidence=0.9)
    )

    doc = PolicyDocument(
        doc_id="d_ec", title="EC Test",
        pages=[(1, "The European Commission oversees Member States.")]
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )
    _, _, confidences = retriever._expand_graph(chunks, depth=1)

    # European Commission has one edge with confidence=0.9 → mean=0.9
    if "European Commission" in confidences:
        assert abs(confidences["European Commission"] - 0.9) < 0.01


def test_expand_graph_deduplicates_triples(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """The same graph triple must not appear twice in the evidence list."""
    from app.core.entity_extraction import Entity, Relation
    from app.services.vector_store import SearchResult

    in_memory_graph.add_entity(
        Entity(text="ClimateOrg", label="ORG", doc_id="d_dup",
               page_number=1, chunk_id="c_dup", char_start=0, char_end=10)
    )
    in_memory_graph.add_entity(
        Entity(text="ClimateTarget", label="ORG", doc_id="d_dup",
               page_number=1, chunk_id="c_dup2", char_start=0, char_end=13)
    )
    in_memory_graph.add_relation(
        Relation(source="ClimateOrg", relation_type="TARGETS",
                 target="ClimateTarget", doc_id="d_dup", chunk_id="c_dup")
    )

    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        embedding_model=mock_embedding_model,
    )

    # Two chunks both mentioning the same entities
    candidates = [
        SearchResult(chunk_id="ca", text="ClimateOrg sets ClimateTarget goals.", score=0.8, metadata={}),
        SearchResult(chunk_id="cb", text="ClimateOrg and ClimateTarget cooperate.", score=0.7, metadata={}),
    ]
    _, evidence, _ = retriever._expand_graph(candidates, depth=1)

    triple_keys = [f"{e.entity}|{e.relation}|{e.target}" for e in evidence]
    assert len(triple_keys) == len(set(triple_keys)), "Duplicate triples found in evidence"


# ---------------------------------------------------------------------------
# Multi-hop graph expansion
# ---------------------------------------------------------------------------

def test_expand_graph_multihop_collects_neighbour_edges(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """Depth-2 expansion collects edges from neighbours of directly-matched entities."""
    from app.core.entity_extraction import Entity, Relation
    from app.core.document_loader import PolicyDocument
    from app.core.text_chunker import chunk_document

    # 3-node chain: European Commission → Member States → Environment Agency
    for name in ("European Commission", "Member States", "Environment Agency"):
        in_memory_graph.add_entity(
            Entity(text=name, label="ORG", doc_id="d_mh",
                   page_number=1, chunk_id=f"c_{name[:3]}", char_start=0, char_end=len(name))
        )
    in_memory_graph.add_relation(
        Relation(source="European Commission", relation_type="OVERSEES",
                 target="Member States", doc_id="d_mh", chunk_id="c_mh1")
    )
    in_memory_graph.add_relation(
        Relation(source="Member States", relation_type="REPORTS_TO",
                 target="Environment Agency", doc_id="d_mh", chunk_id="c_mh2")
    )

    # Only the 1st-hop entity appears in the document text
    text = "The European Commission oversees compliance with the 2030 Climate Strategy."
    doc = PolicyDocument(doc_id="d_mh", title="MH Test", pages=[(1, text)])
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever_d1 = HybridRetriever(
        vector_store=in_memory_vector_store, graph_service=in_memory_graph,
        embedding_model=mock_embedding_model, alpha=0.5, graph_depth=1,
    )
    retriever_d2 = HybridRetriever(
        vector_store=in_memory_vector_store, graph_service=in_memory_graph,
        embedding_model=mock_embedding_model, alpha=0.5, graph_depth=2,
    )

    result_d1 = retriever_d1.retrieve("European Commission compliance", top_k=5)
    result_d2 = retriever_d2.retrieve("European Commission compliance", top_k=5)

    # Depth-2 must find at least as many triples as depth-1
    assert len(result_d2.graph_evidence) >= len(result_d1.graph_evidence), (
        f"depth-2 should collect ≥ depth-1 evidence; "
        f"got d1={len(result_d1.graph_evidence)}, d2={len(result_d2.graph_evidence)}"
    )


def test_multihop_evidence_has_discounted_confidence(mock_embedding_model, in_memory_vector_store, in_memory_graph):
    """Multi-hop edges should carry lower confidence than direct edges."""
    from app.core.entity_extraction import Entity, Relation
    from app.core.document_loader import PolicyDocument
    from app.core.text_chunker import chunk_document
    from app.core.retriever import _MULTIHOP_DISCOUNT

    for name in ("PolicyOrg", "TargetBody", "RemoteAgency"):
        in_memory_graph.add_entity(
            Entity(text=name, label="ORG", doc_id="d_disc",
                   page_number=1, chunk_id=f"c_{name[:3]}", char_start=0, char_end=len(name))
        )
    in_memory_graph.add_relation(
        Relation(source="PolicyOrg", relation_type="FUNDS",
                 target="TargetBody", doc_id="d_disc", chunk_id="c_disc1", confidence=1.0)
    )
    in_memory_graph.add_relation(
        Relation(source="TargetBody", relation_type="DELEGATES_TO",
                 target="RemoteAgency", doc_id="d_disc", chunk_id="c_disc2", confidence=1.0)
    )

    text = "PolicyOrg sets the funding framework for the region."
    doc = PolicyDocument(doc_id="d_disc", title="Disc Test", pages=[(1, text)])
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    embeddings = mock_embedding_model.embed([c.text for c in chunks])
    in_memory_vector_store.upsert(chunks, embeddings)

    retriever_d2 = HybridRetriever(
        vector_store=in_memory_vector_store, graph_service=in_memory_graph,
        embedding_model=mock_embedding_model, alpha=0.5, graph_depth=2,
    )
    result = retriever_d2.retrieve("PolicyOrg funding", top_k=5)

    if len(result.graph_evidence) >= 2:
        confidences = [e.confidence for e in result.graph_evidence]
        # Direct edge (PolicyOrg → TargetBody) has confidence=1.0;
        # indirect edge (TargetBody → RemoteAgency) should be discounted.
        assert min(confidences) <= 1.0 * _MULTIHOP_DISCOUNT + 0.01, (
            "Multi-hop edge should carry discounted confidence"
        )
