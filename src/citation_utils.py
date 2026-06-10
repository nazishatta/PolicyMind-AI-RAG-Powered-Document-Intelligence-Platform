"""Citation formatting utilities for PolicyMind AI search results."""

from __future__ import annotations

from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

_EXCERPT_LENGTH = 250


def format_citation(result: dict[str, Any]) -> str:
    """Format a single search result as a citation string.

    Args:
        result: Dict with keys: document_name, page_number, text, score.

    Returns:
        Formatted citation string.
    """
    doc = result.get("document_name", "Unknown Document")
    page = result.get("page_number", "?")
    score = result.get("score", 0.0)
    excerpt = result.get("text", "")[:_EXCERPT_LENGTH].replace("\n", " ")
    return f"📄 {doc} | Page {page} | Relevance: {score:.2%}\n   \"{excerpt}...\""


def format_sources(results: list[dict[str, Any]]) -> list[str]:
    """Format a list of search results as citation strings.

    Args:
        results: List of result dicts from semantic_search.

    Returns:
        List of formatted citation strings.
    """
    if not results:
        return []
    return [format_citation(r) for r in results]


def deduplicate_sources(sources: list[str]) -> list[str]:
    """Remove duplicate citation strings while preserving order.

    Args:
        sources: List of formatted citation strings.

    Returns:
        Deduplicated list in original order.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique.append(source)
    return unique
