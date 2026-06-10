"""Semantic retrieval from ChromaDB — distance→similarity conversion, quality assessment,
smart search with summarization detection."""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import Chroma

from src.config import TOP_K_RESULTS
from src.logger import get_logger

logger = get_logger(__name__)

# Keywords that indicate the user wants a summary rather than a targeted answer.
_SUMMARIZATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "summarize", "summary", "summarise", "summarisation", "summarization",
        "overview", "key points", "main points", "extract", "outline",
        "brief me", "give me a summary", "what does this document",
        "what is this document", "what are the main",
    }
)


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    vector_store: Chroma,
    top_k: int = TOP_K_RESULTS,
    document_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search the vector store and return the top-k most relevant chunks.

    Uses similarity_search_with_score() to obtain raw cosine distances, then
    converts each distance to a similarity score:

        similarity = 1.0 - distance   (clamped to [0.0, 1.0])

    For unit-normalised embeddings (sentence-transformers) and a cosine-metric
    collection the raw distance is in [0, 1], so this conversion is exact.

    Args:
        query: Natural language query string.
        vector_store: Loaded Chroma vector store instance.
        top_k: Number of results to return.
        document_filter: If set (and not "All Documents"), restrict search to
            chunks whose document_name metadata exactly matches this value.
            Uses a ChromaDB $eq where-clause.

    Returns:
        List of dicts with keys: text, document_name, page_number, chunk_id, score.
        score is a cosine similarity in [0.0, 1.0] — higher is better.
    """
    try:
        # Build optional metadata filter for single-document search
        where_filter: dict | None = None
        if document_filter and document_filter != "All Documents":
            where_filter = {"document_name": {"$eq": document_filter}}

        # similarity_search_with_score returns (Document, raw_distance) pairs.
        # For chromadb cosine space, raw_distance = cosine_distance ∈ [0, 1].
        raw_results = vector_store.similarity_search_with_score(
            query, k=top_k, filter=where_filter
        )
        results: list[dict[str, Any]] = []

        for doc, distance in raw_results:
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            results.append(
                {
                    "text": doc.page_content,
                    "document_name": doc.metadata.get("document_name", "unknown"),
                    "page_number": doc.metadata.get("page_number", 0),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "score": round(similarity, 4),
                }
            )

        logger.info(
            "semantic_search: %d results, top score=%.4f, filter=%r, query: %.80s",
            len(results),
            results[0]["score"] if results else 0.0,
            document_filter,
            query,
        )
        return results
    except Exception as exc:
        logger.error("semantic_search failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Smart search (summarization-aware)
# ---------------------------------------------------------------------------

def smart_search(
    query: str,
    vector_store: Chroma,
    top_k: int = TOP_K_RESULTS,
    document_filter: str | None = None,
) -> dict[str, Any]:
    """Retrieve document chunks with query-type awareness.

    For summarization queries:
        - Fetches up to 30 chunks then selects the highest-scoring chunk from
          each unique page, returning up to 15 page-representative chunks.
          This gives broad document coverage rather than dense repetition of
          the most similar passages.

    For regular questions:
        - Fetches the top-k most similar chunks.

    Args:
        query: Natural language query string.
        vector_store: Loaded Chroma vector store instance.
        top_k: Maximum results for regular questions (default: 5).
        document_filter: If set (and not "All Documents"), restrict search to a
            single document.  Passed through to semantic_search().

    Returns:
        Dict with keys:
            results          (list[dict]) — chunk dicts with score
            is_summarization (bool)
            query_type       (str)        — "summarization" or "question"
    """
    query_lower = query.lower()
    is_summarization = any(kw in query_lower for kw in _SUMMARIZATION_KEYWORDS)

    if is_summarization:
        try:
            # Fetch a large pool, then spread across pages
            pool = semantic_search(
                query, vector_store, top_k=30, document_filter=document_filter
            )

            # Best chunk per page (page_number → chunk dict)
            page_best: dict[int, dict[str, Any]] = {}
            for r in pool:
                page = int(r.get("page_number", 0))
                if page not in page_best or r["score"] > page_best[page]["score"]:
                    page_best[page] = r

            # Sort by page number, cap at 15 representative pages
            results = [page_best[p] for p in sorted(page_best.keys())][:15]

            logger.info(
                "smart_search (summarization): %d page-representative chunks, filter=%r.",
                len(results),
                document_filter,
            )
        except Exception as exc:
            logger.error("smart_search summarization path failed: %s", exc)
            results = []

        return {"results": results, "is_summarization": True, "query_type": "summarization"}

    # Regular question path
    try:
        results = semantic_search(
            query, vector_store, top_k=top_k, document_filter=document_filter
        )
    except Exception as exc:
        logger.error("smart_search question path failed: %s", exc)
        results = []

    return {"results": results, "is_summarization": False, "query_type": "question"}


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

def assess_retrieval_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess the quality of retrieved results based on cosine similarity scores.

    Thresholds (calibrated for cosine similarity after the L2→cosine fix):
        strong       avg_score >= 0.65  → use_document
        moderate     avg_score >= 0.45  → use_hybrid
        weak         avg_score >= 0.30  → use_fallback
        insufficient avg_score <  0.30  → use_fallback

    Args:
        results: List of result dicts from semantic_search (score is 0–1 similarity).

    Returns:
        Dict with keys: avg_score, max_score, quality, recommendation.
    """
    if not results:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "quality": "insufficient",
            "recommendation": "use_fallback",
        }

    try:
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        if avg_score >= 0.65:
            quality, recommendation = "strong", "use_document"
        elif avg_score >= 0.45:
            quality, recommendation = "moderate", "use_hybrid"
        elif avg_score >= 0.30:
            quality, recommendation = "weak", "use_fallback"
        else:
            quality, recommendation = "insufficient", "use_fallback"

        assessment = {
            "avg_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "quality": quality,
            "recommendation": recommendation,
        }
        logger.info(
            "assess_retrieval_quality: %s (avg=%.4f, max=%.4f)",
            quality, avg_score, max_score,
        )
        return assessment
    except Exception as exc:
        logger.error("assess_retrieval_quality failed: %s", exc)
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "quality": "insufficient",
            "recommendation": "use_fallback",
        }


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results as a human-readable string.

    Args:
        results: List of result dicts from semantic_search.

    Returns:
        Formatted multi-line string.
    """
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r['document_name']} — Page {r['page_number']} "
            f"(score: {r['score']:.4f})\n{r['text'][:300]}..."
        )
    return "\n\n".join(lines)
