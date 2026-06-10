"""Vector store abstraction.

ChromaDB (persistent, recommended) and an InMemoryVectorStore (no deps,
good for tests) both implement the same interface so services are
store-agnostic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.core.text_chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float           # cosine similarity [0, 1]
    metadata: dict


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[dict]: ...

    @abstractmethod
    def get_document_detail(self, doc_id: str) -> Optional[dict]: ...

    @abstractmethod
    def count(self) -> int: ...


# ---------------------------------------------------------------------------
# ChromaDB implementation
# ---------------------------------------------------------------------------

class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB-backed store with cosine similarity."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "policy_chunks",
    ) -> None:
        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._collection: Optional[Any] = None

    def _init(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install chromadb") from exc

        client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", self._collection_name)
        return self._collection

    def upsert(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        col = self._init()
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "section_heading": c.section_heading or "",
                    **c.metadata,
                }
                for c in chunks
            ],
        )
        logger.info("Upserted %d chunks", len(chunks))

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        col = self._init()
        n = min(top_k, col.count() or top_k)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec.tolist()],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_id_filter:
            kwargs["where"] = {"doc_id": doc_id_filter}
        res = col.query(**kwargs)
        return [
            SearchResult(
                chunk_id=cid,
                text=doc,
                score=round(1.0 - dist, 4),
                metadata=meta,
            )
            for cid, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]

    def delete_document(self, doc_id: str) -> None:
        self._init().delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[dict]:
        col = self._init()
        result = col.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        chunk_counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            did = meta.get("doc_id", "")
            if not did:
                continue
            chunk_counts[did] = chunk_counts.get(did, 0) + 1
            if did not in seen:
                raw_url = meta.get("source_url", "")
                seen[did] = {
                    "doc_id": did,
                    "doc_title": meta.get("doc_title", ""),
                    "source_url": raw_url if raw_url else None,
                }
        return [
            {**doc, "chunk_count": chunk_counts.get(doc["doc_id"], 0)}
            for doc in seen.values()
        ]

    def get_document_detail(self, doc_id: str) -> Optional[dict]:
        col = self._init()
        result = col.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = result.get("metadatas") or []
        if not metas:
            return None
        pages = {int(m.get("page_number", 1)) for m in metas}
        sections = sorted({m.get("section_heading", "") for m in metas if m.get("section_heading")})
        sample = metas[0]
        return {
            "doc_id": doc_id,
            "doc_title": sample.get("doc_title", ""),
            "source_url": sample.get("source_url") or None,
            "source_label": sample.get("source_label") or None,
            "chunk_count": len(metas),
            "page_count": max(pages),
            "sections": sections,
            "status": "indexed",
        }

    def count(self) -> int:
        return self._init().count()


# ---------------------------------------------------------------------------
# In-memory implementation (numpy, zero external deps)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(BaseVectorStore):
    """Ephemeral numpy-backed store — useful for tests and quick demos."""

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._matrix: Optional[np.ndarray] = None

    def upsert(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        # Simple append; re-ingesting the same chunk_id creates a duplicate
        # (acceptable for demo purposes; production would deduplicate)
        self._chunks.extend(chunks)
        self._matrix = (
            np.vstack([self._matrix, embeddings])
            if self._matrix is not None
            else embeddings.copy()
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        pool = [
            (i, c)
            for i, c in enumerate(self._chunks)
            if doc_id_filter is None or c.doc_id == doc_id_filter
        ]
        if not pool:
            return []
        idxs, pool_chunks = zip(*pool)
        sub = self._matrix[list(idxs)]
        scores = sub @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk_id=pool_chunks[i].chunk_id,
                text=pool_chunks[i].text,
                score=round(float(scores[i]), 4),
                metadata=pool_chunks[i].metadata,
            )
            for i in order
        ]

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if self._matrix is not None and keep else None

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        chunk_counts: dict[str, int] = {}
        for c in self._chunks:
            did = c.doc_id
            chunk_counts[did] = chunk_counts.get(did, 0) + 1
            if did not in seen:
                raw_url = c.metadata.get("source_url", "")
                seen[did] = {
                    "doc_id": did,
                    "doc_title": c.metadata.get("doc_title", ""),
                    "source_url": raw_url if raw_url else None,
                }
        return [
            {**doc, "chunk_count": chunk_counts.get(doc["doc_id"], 0)}
            for doc in seen.values()
        ]

    def get_document_detail(self, doc_id: str) -> Optional[dict]:
        chunks = [c for c in self._chunks if c.doc_id == doc_id]
        if not chunks:
            return None
        pages = {c.page_number for c in chunks}
        sections = sorted({c.section_heading for c in chunks if c.section_heading})
        sample = chunks[0]
        return {
            "doc_id": doc_id,
            "doc_title": sample.metadata.get("doc_title", ""),
            "source_url": sample.metadata.get("source_url") or None,
            "source_label": sample.metadata.get("source_label") or None,
            "chunk_count": len(chunks),
            "page_count": max(pages),
            "sections": sections,
            "status": "indexed",
        }

    def count(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_vector_store(
    provider: str = "chroma",
    chroma_persist_dir: str = "./data/chroma",
    chroma_collection: str = "policy_chunks",
) -> BaseVectorStore:
    if provider == "chroma":
        return ChromaVectorStore(chroma_persist_dir, chroma_collection)
    if provider == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"Unknown vector store provider: {provider!r}")
