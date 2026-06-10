"""Self-contained GraphRAG module for PolicyMind AI.

Uses spaCy (NER) + NetworkX (in-memory graph) + the existing ChromaDB vector
store.  Has zero dependency on backend/ or app/services/.

Degrades gracefully at every level:
  - spaCy not installed           → entity extraction returns []
  - en_core_web_sm not downloaded → entity extraction returns []
  - networkx not installed        → PolicyKnowledgeGraph raises ImportError
    (caught by GraphRAGPipeline, which sets is_ready=False)
  - graph_pipeline is None/not ready → answer_question_with_graph_rag falls
    back to the standard answer_question_with_rag()
"""

from __future__ import annotations

import os
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Entity type filter — policy-relevant spaCy labels
# ---------------------------------------------------------------------------

_POLICY_ENTITY_TYPES: frozenset[str] = frozenset(
    {"ORG", "PERSON", "GPE", "LAW", "NORP", "FAC", "EVENT", "PRODUCT"}
)

# ---------------------------------------------------------------------------
# Lazy spaCy loader
# ---------------------------------------------------------------------------

_nlp: Any = None
_spacy_available: bool | None = None  # None = not yet checked


def _check_spacy() -> bool:
    """Return True if spaCy and en_core_web_sm are both usable."""
    global _spacy_available
    if _spacy_available is not None:
        return _spacy_available
    try:
        import spacy  # type: ignore  # noqa: F401
        spacy.load("en_core_web_sm")
        _spacy_available = True
    except Exception:
        _spacy_available = False
    return _spacy_available


def _get_nlp() -> Any:
    """Return cached spaCy nlp object, or None if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # type: ignore
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' loaded.")
    except Exception as exc:
        logger.debug("spaCy unavailable: %s", exc)
        _nlp = None
    return _nlp


# ===========================================================================
# PART 1 — Entity Extraction
# ===========================================================================

def extract_entities_from_text(
    text: str,
    doc_name: str,
    page_number: int,
) -> list[dict[str, Any]]:
    """Extract policy-relevant named entities from a text chunk using spaCy NER.

    Falls back silently to an empty list when spaCy or its English model is
    not installed.  The caller never needs to handle the spaCy-absent case.

    Args:
        text:        Raw text of the chunk.
        doc_name:    Name of the source document (stored on each entity).
        page_number: Page number within the source document.

    Returns:
        List of entity dicts:
        ``{"text": str, "label": str, "doc_name": str, "page": int}``
        Empty list when spaCy is unavailable or NER finds no policy entities.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    try:
        doc = nlp(text[:100_000])  # spaCy has a max-length guard
    except Exception as exc:
        logger.warning("spaCy NER failed on chunk from '%s' p%d: %s", doc_name, page_number, exc)
        return []

    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in _POLICY_ENTITY_TYPES:
            continue
        name = ent.text.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entities.append(
            {
                "text": name,
                "label": ent.label_,
                "doc_name": doc_name,
                "page": page_number,
            }
        )

    return entities


# ===========================================================================
# PART 2 — Knowledge Graph
# ===========================================================================

class PolicyKnowledgeGraph:
    """In-memory directed knowledge graph backed by NetworkX.

    Nodes represent named entities.  Edges represent CO_OCCURS_WITH
    relationships — two entities co-occur when they both appear in the same
    document chunk.  Edge weight counts the number of co-occurrence events.
    """

    def __init__(self) -> None:
        try:
            import networkx as nx  # type: ignore
            self.graph: Any = nx.DiGraph()
        except ImportError as exc:
            raise ImportError(
                "networkx is required for GraphRAG. pip install networkx"
            ) from exc

        # entity_name → list of document names that mention it
        self.entity_docs: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_entities_from_chunk(
        self,
        entities: list[dict[str, Any]],
        chunk_text: str,  # kept in signature for future use (e.g. LLM relations)
    ) -> None:
        """Add entities and pairwise co-occurrence edges from a single chunk.

        Nodes carry ``label``, ``docs``, and ``pages`` attributes.
        Edges carry ``relation`` (always "CO_OCCURS_WITH") and ``weight``
        (number of times the pair has co-occurred across all chunks).
        """
        for ent in entities:
            name = ent["text"]
            label = ent.get("label", "UNKNOWN")
            doc_name = ent["doc_name"]
            page = ent["page"]

            if self.graph.has_node(name):
                node = self.graph.nodes[name]
                if doc_name not in node.get("docs", []):
                    node["docs"] = node.get("docs", []) + [doc_name]
                if page not in node.get("pages", []):
                    node["pages"] = node.get("pages", []) + [page]
            else:
                self.graph.add_node(name, label=label, docs=[doc_name], pages=[page])

            # entity_docs mapping
            if name not in self.entity_docs:
                self.entity_docs[name] = []
            if doc_name not in self.entity_docs[name]:
                self.entity_docs[name].append(doc_name)

        # Pairwise co-occurrence edges (undirected semantics, stored as directed)
        names = [e["text"] for e in entities]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a == b:
                    continue
                if self.graph.has_edge(a, b):
                    self.graph[a][b]["weight"] = self.graph[a][b].get("weight", 1) + 1
                else:
                    self.graph.add_edge(a, b, relation="CO_OCCURS_WITH", weight=1)
                # Also add reverse direction (symmetric)
                if self.graph.has_edge(b, a):
                    self.graph[b][a]["weight"] = self.graph[b][a].get("weight", 1) + 1
                else:
                    self.graph.add_edge(b, a, relation="CO_OCCURS_WITH", weight=1)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Return entities reachable within ``depth`` hops from ``entity_name``.

        Args:
            entity_name: Source entity to expand from.
            depth:       Maximum graph hops (1 = direct neighbours only).

        Returns:
            List of neighbour dicts:
            ``{"entity": str, "relation": str, "confidence": float}``
            Empty when entity not in graph or networkx fails.
        """
        if not self.graph.has_node(entity_name):
            return []

        try:
            import networkx as nx  # type: ignore

            ego = nx.ego_graph(self.graph, entity_name, radius=depth)
            neighbours: list[dict[str, Any]] = []

            for node in ego.nodes():
                if node == entity_name:
                    continue

                relation = "CO_OCCURS_WITH"
                confidence = 0.3  # default for multi-hop neighbours

                if self.graph.has_edge(entity_name, node):
                    edge = self.graph[entity_name][node]
                    weight = max(1, edge.get("weight", 1))
                    # Normalise weight: asymptote at 1.0, reaches 0.9 at weight=9
                    confidence = round(1.0 - (1.0 / weight), 3)
                    confidence = max(0.1, min(1.0, confidence))
                    relation = edge.get("relation", "CO_OCCURS_WITH")

                neighbours.append(
                    {"entity": node, "relation": relation, "confidence": confidence}
                )

            # Sort by confidence descending
            neighbours.sort(key=lambda x: x["confidence"], reverse=True)
            return neighbours

        except Exception as exc:
            logger.error("get_entity_neighbors failed for '%s': %s", entity_name, exc)
            return []

    def get_graph_boost(
        self,
        query_entities: list[str],
        chunk_entities: list[str],
    ) -> float:
        """Calculate a graph-based boost score in [0.0, 1.0].

        Boost = |query_entities ∩ chunk_entities| / |query_entities|.
        Returns 0.0 when query_entities is empty (no penalty to the caller).

        Args:
            query_entities: Entity names extracted from the user's question.
            chunk_entities: Entity names extracted from a candidate chunk.
        """
        if not query_entities:
            return 0.0
        q_set = set(query_entities)
        c_set = set(chunk_entities)
        overlap = len(q_set & c_set)
        return round(overlap / len(q_set), 4)

    def get_stats(self) -> dict[str, Any]:
        """Return entity count, relation count, and top-5 most connected entities."""
        try:
            n_entities = self.graph.number_of_nodes()
            n_relations = self.graph.number_of_edges()

            if n_entities == 0:
                return {"entity_count": 0, "relation_count": 0, "top_entities": []}

            degrees = dict(self.graph.degree())
            top = sorted(degrees, key=lambda x: degrees[x], reverse=True)[:5]
            return {
                "entity_count": n_entities,
                "relation_count": n_relations,
                "top_entities": top,
            }
        except Exception as exc:
            logger.error("get_stats failed: %s", exc)
            return {"entity_count": 0, "relation_count": 0, "top_entities": []}

    def extract_query_entities(self, query_text: str) -> list[str]:
        """Extract entity name strings from a query using the same spaCy model.

        Returns empty list when spaCy is unavailable — no exception raised.
        """
        entities = extract_entities_from_text(query_text, "_query_", 0)
        return [e["text"] for e in entities]


# ===========================================================================
# PART 3 — GraphRAG Pipeline
# ===========================================================================

class GraphRAGPipeline:
    """Wraps PolicyKnowledgeGraph with ingestion and hybrid-search logic.

    Designed to sit on top of the existing ChromaDB vector store without
    replacing it — the graph adds entity-aware re-ranking on top of vector
    search results.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.spacy_available: bool = _check_spacy()

        try:
            self.knowledge_graph = PolicyKnowledgeGraph()
        except ImportError as exc:
            logger.error("NetworkX unavailable — GraphRAGPipeline inactive: %s", exc)
            self.knowledge_graph = None  # type: ignore
            return

        logger.info(
            "GraphRAGPipeline created (spaCy available: %s)", self.spacy_available
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_chunks_to_graph(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract entities from chunks and build the knowledge graph.

        Args:
            chunks: List of chunk dicts from ``src.text_splitter``
                    (keys: chunk_id, document_name, page_number, text).

        Returns:
            Summary dict:
            ``{"entities_extracted": int, "relations_created": int,
               "spacy_used": bool}``
        """
        if self.knowledge_graph is None:
            return {"entities_extracted": 0, "relations_created": 0, "spacy_used": False}

        total_entities = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            doc_name = chunk.get("document_name", "unknown")
            page_number = int(chunk.get("page_number", 0))

            try:
                entities = extract_entities_from_text(text, doc_name, page_number)
                if entities:
                    self.knowledge_graph.add_entities_from_chunk(entities, text)
                    total_entities += len(entities)
            except Exception as exc:
                logger.warning(
                    "Skipping chunk from '%s' p%d: %s", doc_name, page_number, exc
                )

        stats = self.knowledge_graph.get_stats()
        self.is_ready = True

        logger.info(
            "Graph ingestion done: %d entities extracted, %d graph nodes, %d relations",
            total_entities, stats["entity_count"], stats["relation_count"],
        )

        return {
            "entities_extracted": total_entities,
            "relations_created": stats["relation_count"],
            "spacy_used": self.spacy_available,
        }

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        vector_store: Any,
        top_k: int = 5,
        document_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval: vector search + graph-entity re-ranking.

        Step 1: Retrieve ``top_k * 2`` candidates via ChromaDB cosine search.
        Step 2: Extract entities from the query.
        Step 3: For each candidate, extract chunk entities and compute hybrid score.
        Step 4: Re-rank by hybrid score, return top ``top_k`` results.
        Step 5: Annotate each result with ``graph_boost`` and ``hybrid_score``.

        Falls back to pure vector ranking when the graph is empty or spaCy is
        unavailable (graph_boost = 0.0 for all chunks, so ranking is unchanged).

        Args:
            query:           User question.
            vector_store:    Loaded LangChain Chroma instance.
            top_k:           Number of results to return.
            document_filter: Optional filename filter passed to semantic_search.

        Returns:
            Re-ranked list of chunk dicts with added ``graph_boost`` and
            ``hybrid_score`` fields.
        """
        from src.retriever import semantic_search  # noqa: PLC0415

        # Step 1 — vector candidates (2x for re-ranking headroom)
        try:
            candidates = semantic_search(
                query, vector_store, top_k=top_k * 2, document_filter=document_filter
            )
        except Exception as exc:
            logger.error("hybrid_search: semantic_search failed: %s", exc)
            return []

        if not candidates:
            return []

        # Step 2 — query entities
        query_entities: list[str] = []
        if self.knowledge_graph is not None and self.spacy_available:
            try:
                query_entities = self.knowledge_graph.extract_query_entities(query)
            except Exception as exc:
                logger.debug("Query entity extraction failed: %s", exc)

        # Steps 3 & 4 — score each candidate and re-rank
        scored: list[dict[str, Any]] = []
        for chunk in candidates:
            vector_score = float(chunk.get("score", 0.0))

            graph_boost = 0.0
            if query_entities and self.knowledge_graph is not None:
                try:
                    chunk_entities = extract_entities_from_text(
                        chunk.get("text", ""),
                        chunk.get("document_name", ""),
                        int(chunk.get("page_number", 0)),
                    )
                    chunk_entity_names = [e["text"] for e in chunk_entities]
                    graph_boost = self.knowledge_graph.get_graph_boost(
                        query_entities, chunk_entity_names
                    )
                except Exception:
                    pass

            hybrid_score = round(0.7 * vector_score + 0.3 * graph_boost, 4)

            entry = dict(chunk)
            entry["graph_boost"] = round(graph_boost, 4)
            entry["hybrid_score"] = hybrid_score
            scored.append(entry)

        # Re-rank by hybrid_score descending, return top_k
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Graph evidence
    # ------------------------------------------------------------------

    def get_graph_evidence_for_query(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Return up to 5 high-confidence graph triples related to the query.

        Args:
            query: User question string.

        Returns:
            List of triple dicts:
            ``{"entity": str, "relation": str, "target": str,
               "confidence": float, "source": str}``
        """
        if self.knowledge_graph is None or not self.spacy_available:
            return []

        try:
            query_entities = self.knowledge_graph.extract_query_entities(query)
        except Exception:
            return []

        if not query_entities:
            return []

        triples: list[dict[str, Any]] = []
        seen: set[str] = set()

        for entity_name in query_entities:
            neighbours = self.knowledge_graph.get_entity_neighbors(entity_name, depth=1)
            for nb in neighbours:
                key = f"{entity_name}|{nb['entity']}"
                if key in seen:
                    continue
                seen.add(key)

                # Find a source document for this relationship
                source_docs = self.knowledge_graph.entity_docs.get(entity_name, [])
                source = source_docs[0] if source_docs else "unknown"

                triples.append(
                    {
                        "entity": entity_name,
                        "relation": nb.get("relation", "CO_OCCURS_WITH"),
                        "target": nb["entity"],
                        "confidence": nb["confidence"],
                        "source": source,
                    }
                )

        # Sort by confidence, keep top 5
        triples.sort(key=lambda x: x["confidence"], reverse=True)
        return triples[:5]


# ===========================================================================
# PART 4 — Integration function
# ===========================================================================

_GRAPH_SYSTEM_PROMPT = (
    "You are PolicyMind AI with knowledge graph capabilities. "
    "Answer using the document evidence and knowledge graph context provided. "
    "The knowledge graph shows entity relationships extracted from the documents. "
    "Structure your answer with:\n"
    "- ANSWER: (direct response)\n"
    "- KEY ENTITIES: (important organizations, policies, or concepts mentioned)\n"
    "- EVIDENCE: (specific citations with source and page)\n"
    "If graph context is available, mention the relationships between key entities."
)


def answer_question_with_graph_rag(
    question: str,
    vector_store: Any,
    graph_pipeline: GraphRAGPipeline | None,
    document_filter: str | None = None,
) -> dict[str, Any]:
    """End-to-end GraphRAG query: hybrid search + graph evidence + LLM answer.

    Falls back silently to ``answer_question_with_rag()`` when:
      - graph_pipeline is None
      - graph_pipeline.is_ready is False
      - any step of the graph path raises an exception

    Args:
        question:        User's natural language question.
        vector_store:    Loaded LangChain Chroma instance.
        graph_pipeline:  Initialised GraphRAGPipeline (or None).
        document_filter: Optional filename to restrict retrieval scope.

    Returns:
        Standardised result dict compatible with ``_render_rag_result()``
        in ``app/components/chat_ui.py``:

        ``answer``          (str)
        ``answer_type``     (str)  — "graph_rag" or existing types on fallback
        ``provider``        (str)
        ``confidence``      (float)
        ``sources_used``    (list[str])
        ``graph_evidence``  (list[dict])
        ``graph_boost_used`` (bool)
        ``entities_found``  (list[str])
        ``fallback_used``   (bool)
        ``results``         (list[dict])
        ``sources``         (list[str])
        ``query_type``      (str)
    """
    from src.rag_chain import (  # noqa: PLC0415
        answer_question_with_rag,
        build_context_from_results,
        detect_query_type,
        generate_rag_answer,
    )
    from src.citation_utils import format_sources  # noqa: PLC0415

    # ── Fallback guard ───────────────────────────────────────────────────────
    if graph_pipeline is None or not graph_pipeline.is_ready:
        logger.info("GraphRAG not ready — falling back to standard RAG.")
        result = answer_question_with_rag(
            question, vector_store, document_filter=document_filter
        )
        result.setdefault("graph_evidence", [])
        result.setdefault("graph_boost_used", False)
        result.setdefault("entities_found", [])
        return result

    try:
        # Step 1 — Hybrid search (vector + graph re-ranking)
        results = graph_pipeline.hybrid_search(
            question, vector_store, top_k=5, document_filter=document_filter
        )

        # Step 2 — Graph evidence triples
        graph_evidence = graph_pipeline.get_graph_evidence_for_query(question)

        # Step 3 — Build enhanced context
        doc_context = build_context_from_results(results)
        context = "DOCUMENT EVIDENCE:\n" + doc_context

        if graph_evidence:
            context += "\n\nKNOWLEDGE GRAPH CONTEXT:\n"
            for triple in graph_evidence:
                context += (
                    f"- {triple['entity']} → {triple['relation']} "
                    f"→ {triple['target']} "
                    f"(confidence: {triple['confidence']:.0%})\n"
                )

        # Step 4 — LLM answer with graph-enhanced system prompt
        query_type = detect_query_type(question)
        sources = format_sources(results)
        avg_score = (
            sum(r.get("hybrid_score", r.get("score", 0.0)) for r in results) / len(results)
            if results
            else 0.0
        )

        answer_data = generate_rag_answer(
            question,
            context,
            sources,
            results,
            avg_score,
            query_type,
            system_prompt=_GRAPH_SYSTEM_PROMPT,
        )

        # Step 5 — Collect query entities for display
        entities_found: list[str] = []
        if graph_pipeline.knowledge_graph is not None:
            try:
                entities_found = graph_pipeline.knowledge_graph.extract_query_entities(question)
            except Exception:
                pass

        graph_boost_used = any(r.get("graph_boost", 0.0) > 0.0 for r in results)

        logger.info(
            "GraphRAG answer: provider=%s, type=%s, conf=%.3f, triples=%d, boost_used=%s",
            answer_data["provider"],
            answer_data["answer_type"],
            avg_score,
            len(graph_evidence),
            graph_boost_used,
        )

        return {
            "answer": answer_data["answer"],
            "answer_type": "graph_rag",
            "provider": answer_data["provider"],
            "confidence": round(avg_score, 4),
            "sources_used": sources,
            "graph_evidence": graph_evidence,
            "graph_boost_used": graph_boost_used,
            "entities_found": entities_found,
            "fallback_used": answer_data["fallback_used"],
            "results": results,
            "sources": sources,
            "query_type": query_type,
        }

    except Exception as exc:
        logger.error("GraphRAG pipeline failed (%s) — falling back to standard RAG.", exc)
        result = answer_question_with_rag(
            question, vector_store, document_filter=document_filter
        )
        result.setdefault("graph_evidence", [])
        result.setdefault("graph_boost_used", False)
        result.setdefault("entities_found", [])
        return result
