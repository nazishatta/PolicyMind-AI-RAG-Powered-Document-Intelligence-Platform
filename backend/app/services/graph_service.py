"""Knowledge graph service abstraction.

NetworkX provides a zero-dependency in-memory graph (default and demo mode).
Neo4j is activated when GRAPH_PROVIDER=neo4j and NEO4J_PASSWORD is set.
Both implementations satisfy the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.entity_extraction import Entity, Relation

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    name: str
    label: str
    doc_ids: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    doc_id: str
    confidence: float = 1.0


class BaseGraphService(ABC):
    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def add_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def get_neighbours(self, entity_name: str, depth: int = 1) -> list[GraphNode]: ...

    @abstractmethod
    def get_edges(self, entity_name: str) -> list[GraphEdge]: ...

    @abstractmethod
    def entity_count(self) -> int: ...

    @abstractmethod
    def relation_count(self) -> int: ...


# ---------------------------------------------------------------------------
# NetworkX (in-memory)
# ---------------------------------------------------------------------------

class InMemoryGraphService(BaseGraphService):
    """Directed graph backed by NetworkX — no external services required."""

    def __init__(self) -> None:
        try:
            import networkx as nx  # type: ignore
            self._g: Any = nx.DiGraph()
        except ImportError as exc:
            raise ImportError("pip install networkx") from exc

    def add_entity(self, entity: Entity) -> None:
        name = entity.text
        if self._g.has_node(name):
            self._g.nodes[name].setdefault("doc_ids", []).append(entity.doc_id)
        else:
            self._g.add_node(name, label=entity.label, doc_ids=[entity.doc_id])

    def add_relation(self, relation: Relation) -> None:
        if not self._g.has_node(relation.source):
            self._g.add_node(relation.source, label="UNKNOWN", doc_ids=[relation.doc_id])
        if not self._g.has_node(relation.target):
            self._g.add_node(relation.target, label="UNKNOWN", doc_ids=[relation.doc_id])
        self._g.add_edge(
            relation.source,
            relation.target,
            relation=relation.relation_type,
            doc_id=relation.doc_id,
            confidence=relation.confidence,
        )

    def get_neighbours(self, entity_name: str, depth: int = 1) -> list[GraphNode]:
        if not self._g.has_node(entity_name):
            return []
        import networkx as nx  # type: ignore

        reachable = nx.single_source_shortest_path_length(self._g, entity_name, cutoff=depth)
        nodes = []
        for name, dist in reachable.items():
            if name == entity_name:
                continue
            attrs = self._g.nodes[name]
            nodes.append(
                GraphNode(
                    name=name,
                    label=attrs.get("label", "UNKNOWN"),
                    doc_ids=attrs.get("doc_ids", []),
                    properties={"distance": dist},
                )
            )
        return nodes

    def get_edges(self, entity_name: str) -> list[GraphEdge]:
        edges = []
        for src, tgt, data in self._g.out_edges(entity_name, data=True):
            edges.append(
                GraphEdge(
                    source=src,
                    relation=data.get("relation", "RELATED_TO"),
                    target=tgt,
                    doc_id=data.get("doc_id", ""),
                    confidence=data.get("confidence", 1.0),
                )
            )
        for src, tgt, data in self._g.in_edges(entity_name, data=True):
            edges.append(
                GraphEdge(
                    source=src,
                    relation=data.get("relation", "RELATED_TO"),
                    target=tgt,
                    doc_id=data.get("doc_id", ""),
                    confidence=data.get("confidence", 1.0),
                )
            )
        return edges

    def entity_count(self) -> int:
        return self._g.number_of_nodes()

    def relation_count(self) -> int:
        return self._g.number_of_edges()


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

class Neo4jGraphService(BaseGraphService):
    """Neo4j bolt-driver implementation.

    Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to be set.
    Install: pip install neo4j
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._db = database
        except ImportError as exc:
            raise ImportError("pip install neo4j  (or use GRAPH_PROVIDER=memory)") from exc
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        cypher = (
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        with self._driver.session(database=self._db) as s:
            s.run(cypher)

    def _run(self, cypher: str, **params: Any) -> Any:
        with self._driver.session(database=self._db) as s:
            return s.run(cypher, **params).data()

    def add_entity(self, entity: Entity) -> None:
        self._run(
            "MERGE (e:Entity {name: $name}) "
            "SET e.label = $label "
            "WITH e "
            "CALL apoc.create.addLabels(e, [$label]) YIELD node RETURN node",
            name=entity.text, label=entity.label,
        )

    def add_relation(self, relation: Relation) -> None:
        cypher = (
            "MERGE (a:Entity {name: $src}) "
            "MERGE (b:Entity {name: $tgt}) "
            f"MERGE (a)-[r:{relation.relation_type}]->(b) "
            "SET r.doc_id = $doc_id, r.confidence = $conf"
        )
        self._run(cypher, src=relation.source, tgt=relation.target,
                  doc_id=relation.doc_id, conf=relation.confidence)

    def get_neighbours(self, entity_name: str, depth: int = 1) -> list[GraphNode]:
        rows = self._run(
            f"MATCH (a:Entity {{name: $name}})-[*1..{depth}]-(b:Entity) "
            "RETURN DISTINCT b.name AS name, b.label AS label",
            name=entity_name,
        )
        return [GraphNode(name=r["name"], label=r["label"]) for r in rows]

    def get_edges(self, entity_name: str) -> list[GraphEdge]:
        rows = self._run(
            "MATCH (a:Entity {name: $name})-[r]->(b:Entity) "
            "RETURN a.name AS src, type(r) AS rel, b.name AS tgt, "
            "r.doc_id AS doc_id, r.confidence AS conf",
            name=entity_name,
        )
        return [
            GraphEdge(r["src"], r["rel"], r["tgt"], r.get("doc_id", ""), r.get("conf", 1.0))
            for r in rows
        ]

    def entity_count(self) -> int:
        return self._run("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]

    def relation_count(self) -> int:
        return self._run("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_graph_service(
    provider: str = "memory",
    neo4j_uri: str = "",
    neo4j_user: str = "",
    neo4j_password: str = "",
    neo4j_database: str = "neo4j",
) -> BaseGraphService:
    if provider == "neo4j":
        return Neo4jGraphService(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    if provider == "memory":
        return InMemoryGraphService()
    raise ValueError(f"Unknown graph provider: {provider!r}")
