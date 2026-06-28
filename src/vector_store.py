# ChromaDB cosine distance ranges from 0 (identical) to 1 (orthogonal) for
# unit-normalised embeddings (sentence-transformers outputs are always normalised).
# We convert to similarity with:  similarity = 1 - distance
# This gives 0.0 (no match) to 1.0 (perfect match), which is what every
# threshold in the rest of the codebase expects.

"""ChromaDB vector store management — cosine-metric collection, create, load, reset, extend."""

from __future__ import annotations

from typing import Any

import chromadb
from langchain_community.vectorstores import Chroma

from src.config import CHROMA_DB_PATH
from src.embeddings import get_embedding_model
from src.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "policymind_docs"
_COSINE_META = {"hnsw:space": "cosine"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client() -> Any:
    """Return a persistent chromadb client pointing at CHROMA_DB_PATH."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _ensure_cosine_collection(client: Any) -> Any:
    """Get or create the collection with cosine distance metric.

    If the collection already exists the metadata is NOT changed (ChromaDB does
    not allow changing the index metric after creation).  Use reset_vector_store()
    to rebuild an existing wrong-metric collection.
    """
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata=_COSINE_META,
    )


def _chunks_to_langchain_docs(
    chunks: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Convert chunk dicts to the parallel lists expected by Chroma.add_texts."""
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "document_name": c.get("document_name", ""),
            "page_number": c.get("page_number", 0),
        }
        for c in chunks
    ]
    ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
    return texts, metadatas, ids


def _langchain_wrapper(client: Any) -> Chroma:
    """Return a LangChain Chroma wrapper around the already-configured chromadb client."""
    return Chroma(
        client=client,
        collection_name=_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset_vector_store() -> None:
    """Delete the existing collection (whatever metric it used) and recreate it
    with the cosine distance metric.

    Call this whenever relevance scores seem wrong (typically caused by an
    existing L2-metric collection created before this fix was applied).
    After calling reset_vector_store(), call create_vector_store(chunks) to
    re-embed all your documents into the new cosine collection.
    """
    try:
        client = _get_client()
        try:
            client.delete_collection(_COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", _COLLECTION_NAME)
        except Exception as del_exc:
            logger.warning(
                "Could not delete collection '%s' (may not exist): %s",
                _COLLECTION_NAME, del_exc,
            )

        _ensure_cosine_collection(client)
        logger.info(
            "Recreated collection '%s' with cosine distance metric.", _COLLECTION_NAME
        )
    except Exception as exc:
        logger.error("reset_vector_store failed: %s", exc)
        raise


def create_vector_store(chunks: list[dict[str, Any]]) -> Chroma:
    """Embed all chunks and store them in the cosine-metric ChromaDB collection.

    If the collection does not exist it is created with cosine distance.
    If it already exists (e.g. after reset_vector_store()) the existing metric is kept.

    Args:
        chunks: List of chunk dicts (chunk_id, document_name, page_number, text).

    Returns:
        LangChain Chroma wrapper ready for similarity search.
    """
    try:
        logger.info(
            "Creating vector store with %d chunks at '%s'", len(chunks), CHROMA_DB_PATH
        )
        client = _get_client()
        _ensure_cosine_collection(client)

        texts, metadatas, ids = _chunks_to_langchain_docs(chunks)
        vector_store = _langchain_wrapper(client)
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        logger.info("Vector store created: %d chunks embedded.", len(chunks))
        return vector_store
    except Exception as exc:
        logger.error("Failed to create vector store: %s", exc)
        raise


def load_vector_store() -> Chroma:
    """Load the existing ChromaDB collection from disk.

    Ensures the collection is created with cosine metric if it does not yet exist.

    Returns:
        LangChain Chroma wrapper ready for similarity search.
    """
    try:
        logger.info("Loading vector store from '%s'", CHROMA_DB_PATH)
        client = _get_client()
        _ensure_cosine_collection(client)
        vector_store = _langchain_wrapper(client)
        logger.info("Vector store loaded.")
        return vector_store
    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc)
        raise


def get_collection_stats() -> dict[str, Any]:
    """Query the ChromaDB collection and return summary statistics.

    Returns:
        Dict with keys:
            total_chunks    (int)        — total number of embedded chunks
            documents       (list[str])  — sorted list of unique document names
            total_documents (int)        — number of unique documents
    """
    try:
        client = _get_client()
        collection = _ensure_cosine_collection(client)

        result = collection.get(include=["metadatas"])
        metadatas: list[dict[str, Any]] = result.get("metadatas") or []

        doc_names: list[str] = sorted(
            {m.get("document_name", "unknown") for m in metadatas if m}
        )

        stats = {
            "total_chunks": len(metadatas),
            "documents": doc_names,
            "total_documents": len(doc_names),
        }
        logger.info(
            "get_collection_stats: %d chunks across %d document(s).",
            stats["total_chunks"],
            stats["total_documents"],
        )
        return stats
    except Exception as exc:
        logger.error("get_collection_stats failed: %s", exc)
        return {"total_chunks": 0, "documents": [], "total_documents": 0}


def add_chunks_to_vector_store(
    chunks: list[dict[str, Any]], vector_store: Chroma
) -> Chroma:
    """Add new chunks to an already-loaded vector store.

    Args:
        chunks: List of chunk dicts to embed and store.
        vector_store: Existing LangChain Chroma instance to extend.

    Returns:
        The updated Chroma instance.
    """
    try:
        texts, metadatas, ids = _chunks_to_langchain_docs(chunks)
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("Added %d chunks to existing vector store.", len(chunks))
        return vector_store
    except Exception as exc:
        logger.error("Failed to add chunks to vector store: %s", exc)
        raise
