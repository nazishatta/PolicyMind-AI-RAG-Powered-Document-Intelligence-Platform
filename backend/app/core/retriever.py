"""Hybrid GraphRAG retriever.

Retrieval pipeline
------------------
1. Embed the question → query vector.
2. Vector search → top-k candidate chunks (fetches 2x for reranking headroom).
3. Extract named entities from candidate chunk texts (spaCy → regex fallback).
4. Graph neighbourhood expansion: collect edges for matched entities up to
   ``graph_depth`` hops.  Depth-1 returns direct edges; depth-2 also walks
   to each matched entity's neighbours and collects their edges (discounted).
5. Score fusion — citation-aware, proportional graph boost:

       fused = α · vector_score + (1-α) · graph_boost

   where graph_boost reflects *how many* of the query's relevant entities
   appear in this chunk AND *how confident* those graph edges are:

       graph_boost = Σ entity_confidence(e) / |entity_set|
                     for e in (chunk_entities ∩ entity_set)

   This replaces the old binary (0 or 1) boost with a score that rewards
   chunks covering more of the query's key entities at higher edge confidence.

6. Return top-k reranked chunks + deduplicated graph evidence sorted by
   confidence (direct edges before discounted multi-hop edges).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.embeddings import EmbeddingModel
from app.schemas.response import GraphEvidence
from app.services.graph_service import BaseGraphService
from app.services.vector_store import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)

# Regex fallback: capitalised phrases (1–5 words) or 4-digit years.
_ENTITY_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,4}|\d{4})\b"
)

# spaCy entity types relevant to policy documents
_POLICY_ENTITY_TYPES = {
    "ORG", "GPE", "LAW", "MONEY", "PERCENT", "DATE",
    "PRODUCT", "EVENT", "NORP",
}

# Confidence multiplier applied to evidence collected via multi-hop expansion
# (indirect graph neighbours) vs direct edges.
_MULTIHOP_DISCOUNT = 0.8


@dataclass
class RetrievalResult:
    """Output of a single hybrid retrieval pass."""

    chunks: list[SearchResult]
    graph_evidence: list[GraphEvidence]
    entity_hits: list[str] = field(default_factory=list)


class HybridRetriever:
    """Fuses semantic vector search with knowledge-graph neighbourhood expansion.

    Parameters
    ----------
    vector_store:       Chunked document index.
    graph_service:      Knowledge graph (in-memory or Neo4j).
    embedding_model:    Model used to embed the query vector.
    alpha:              Weight on vector score (0.0–1.0). Default 0.7.
    graph_depth:        Neighbourhood traversal depth.  1 = direct edges only;
                        2 = also collects edges from 1-hop neighbours
                        (with a confidence discount).
    max_graph_evidence: Cap on GraphEvidence triples returned. Default 10.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_service: BaseGraphService,
        embedding_model: EmbeddingModel,
        alpha: float = 0.7,
        graph_depth: int = 1,
        max_graph_evidence: int = 10,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self._vs = vector_store
        self._gs = graph_service
        self._emb = embedding_model
        self._alpha = alpha
        self._graph_depth = graph_depth
        self._max_graph_evidence = max_graph_evidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
        include_graph: bool = True,
        graph_depth: Optional[int] = None,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        graph_depth:
            Overrides ``self._graph_depth`` for this call only.
            1 = direct edges; 2 = also collect neighbour edges (discounted);
            3 = two layers of indirect evidence.  ``None`` → use instance default.

        Returns a :class:`RetrievalResult` containing reranked chunks and
        graph evidence.  If the graph is empty or include_graph is False,
        vector-only results are returned without reranking.
        """
        q_vec = self._emb.embed_one(question)

        # Fetch extra candidates so reranking has headroom
        candidate_k = min(top_k * 2, top_k + 10)
        candidates = self._vs.search(q_vec, top_k=candidate_k, doc_id_filter=doc_id_filter)

        if not candidates:
            return RetrievalResult(chunks=[], graph_evidence=[])

        # Per-call depth override: use caller-supplied value, else instance default.
        effective_depth = graph_depth if graph_depth is not None else self._graph_depth

        graph_has_data = self._gs.entity_count() > 0
        if include_graph and graph_has_data:
            entity_hits, graph_evidence, entity_confidences = self._expand_graph(
                candidates, effective_depth
            )
            reranked = self._fuse_scores(candidates, entity_hits, entity_confidences)
        else:
            entity_hits, graph_evidence, entity_confidences = [], [], {}
            reranked = candidates

        final = reranked[:top_k]
        logger.info(
            "Retrieval: top_k=%d, candidates=%d, entity_hits=%d, graph_triples=%d, depth=%d",
            len(final), len(candidates), len(entity_hits), len(graph_evidence), effective_depth,
        )
        return RetrievalResult(
            chunks=final,
            graph_evidence=graph_evidence[: self._max_graph_evidence],
            entity_hits=entity_hits,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract named entities from text using spaCy or regex fallback."""
        try:
            import spacy  # type: ignore
            from app.core.entity_extraction import _get_nlp
            nlp = _get_nlp()
            doc = nlp(text)  # type: ignore[operator]
            return [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in _POLICY_ENTITY_TYPES
            ]
        except Exception:
            # Regex fallback — no spaCy model required
            return _ENTITY_RE.findall(text)

    def _expand_graph(
        self,
        candidates: list[SearchResult],
        depth: int,
    ) -> tuple[list[str], list[GraphEvidence], dict[str, float]]:
        """Collect graph edges for entities found in candidate chunk texts.

        Parameters
        ----------
        depth:
            Traversal depth.  1 = direct edges only.  >1 = also walk each
            matched entity's graph neighbours and collect *their* edges,
            applying a confidence discount of ``_MULTIHOP_DISCOUNT`` per hop
            to signal indirect evidence.

        Returns
        -------
        found : list[str]
            Entity names that had at least one edge or graph neighbour.
        evidence : list[GraphEvidence]
            Deduplicated, confidence-sorted triples (direct edges first).
        entity_confidences : dict[str, float]
            Maps entity name → mean confidence of its direct edges.
            Used by ``_fuse_scores`` to weight the graph boost proportionally.
        """
        entity_names: set[str] = set()
        for hit in candidates:
            for name in self._extract_entities_from_text(hit.text):
                entity_names.add(name)

        evidence: list[GraphEvidence] = []
        found: list[str] = []
        entity_confidences: dict[str, float] = {}
        seen_triples: set[str] = set()

        for name in entity_names:
            edges = self._gs.get_edges(name)
            if edges:
                found.append(name)
                # Mean confidence of this entity's direct edges (used for boost weighting)
                entity_confidences[name] = sum(e.confidence for e in edges) / len(edges)

                for edge in edges:
                    key = f"{edge.source}|{edge.relation}|{edge.target}"
                    if key not in seen_triples:
                        seen_triples.add(key)
                        evidence.append(
                            GraphEvidence(
                                entity=edge.source,
                                relation=edge.relation,
                                target=edge.target,
                                source_doc_id=edge.doc_id,
                                confidence=edge.confidence,
                            )
                        )

                # Multi-hop expansion: when depth > 1, collect edges from neighbours
                # of directly-matched entities (indirect graph context).
                if depth > 1:
                    neighbours = self._gs.get_neighbours(name, depth=depth)
                    for neighbour in neighbours:
                        for n_edge in self._gs.get_edges(neighbour.name):
                            key = f"{n_edge.source}|{n_edge.relation}|{n_edge.target}"
                            if key not in seen_triples:
                                seen_triples.add(key)
                                evidence.append(
                                    GraphEvidence(
                                        entity=n_edge.source,
                                        relation=n_edge.relation,
                                        target=n_edge.target,
                                        source_doc_id=n_edge.doc_id,
                                        # Discount indirect evidence confidence
                                        confidence=round(
                                            n_edge.confidence * _MULTIHOP_DISCOUNT, 4
                                        ),
                                    )
                                )
            else:
                # No direct edges: check whether the node exists (isolated node)
                neighbours = self._gs.get_neighbours(name, depth=depth)
                if neighbours:
                    found.append(name)

        evidence.sort(key=lambda e: e.confidence, reverse=True)
        return found, evidence, entity_confidences

    def _fuse_scores(
        self,
        candidates: list[SearchResult],
        entity_hits: list[str],
        entity_confidences: dict[str, float],
    ) -> list[SearchResult]:
        """Rerank candidates by fusing vector score with a citation-aware graph boost.

        The graph boost is proportional rather than binary:

            graph_boost = Σ entity_confidences[e] / |entity_set|
                          for e in (chunk_entities ∩ entity_set)

        This means a chunk that mentions *more* of the query's relevant
        entities — and whose graph edges have *higher* confidence — receives
        a larger boost.  A chunk with no matching entities scores
        ``α · vector_score`` exactly (the graph term is zero).

        The maximum possible graph_boost approaches 1.0 only when all of the
        query's relevant entities appear in the chunk and all their edges have
        confidence=1.0.
        """
        entity_set = set(entity_hits)
        n_query_entities = max(len(entity_set), 1)
        alpha = self._alpha

        def _score(hit: SearchResult) -> float:
            if not entity_set:
                return alpha * hit.score

            chunk_entities = set(self._extract_entities_from_text(hit.text))
            matched = chunk_entities & entity_set
            if not matched:
                return alpha * hit.score

            # Sum confidences of matched entities, normalised by total query-relevant
            # entities so the boost is always in [0, 1].
            graph_boost = (
                sum(entity_confidences.get(e, 0.5) for e in matched)
                / n_query_entities
            )
            return alpha * hit.score + (1.0 - alpha) * graph_boost

        return sorted(candidates, key=_score, reverse=True)
