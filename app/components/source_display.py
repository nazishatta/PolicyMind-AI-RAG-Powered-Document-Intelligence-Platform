"""Streamlit source display — colour-coded relevance bars and styled source cards.

Thresholds (calibrated for cosine similarity post-fix):
    STRONG   score >= 0.55  🟢
    MODERATE score >= 0.35  🟡
    WEAK     score <  0.35  🔴
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.ui_helpers import render_step_header


def render_source_card(source: dict[str, Any], index: int) -> None:
    """Render a single source as a styled expander with relevance badge."""
    score = max(0.0, min(1.0, float(source.get("score", 0.0))))
    score_pct = int(score * 100)
    doc_name = source.get("document_name", "Unknown")
    page_num = source.get("page_number", "?")
    text = source.get("text", "")
    chunk_id = source.get("chunk_id", "")

    if score >= 0.55:
        badge_text = f"STRONG {score_pct}%"
        dot = "🟢"
    elif score >= 0.35:
        badge_text = f"MODERATE {score_pct}%"
        dot = "🟡"
    else:
        badge_text = f"WEAK {score_pct}%"
        dot = "🔴"

    with st.expander(
        f"{dot} Source {index} | {doc_name} | Page {page_num} | {badge_text}"
    ):
        st.progress(score)
        st.markdown(f"**Document:** {doc_name}")
        st.markdown(f"**Page:** {page_num}")
        st.markdown(f"**Relevance:** {score_pct}%")
        st.markdown("**Excerpt:**")
        excerpt = text[:400]
        st.markdown(f"> {excerpt}{'...' if len(text) > 400 else ''}")
        if chunk_id:
            st.caption(f"Chunk ID: {chunk_id}")


def render_graph_panel(graph_stats: dict[str, Any]) -> None:
    """Display a compact knowledge-graph status panel."""
    entity_count = graph_stats.get("entity_count", 0)
    relation_count = graph_stats.get("relation_count", 0)
    graph_enabled = graph_stats.get("graph_enabled", False)
    docs_indexed = graph_stats.get("documents_indexed", 0)

    if entity_count == 0:
        st.info(
            "Knowledge Graph: empty — entity extraction requires spaCy "
            "(`python -m spacy download en_core_web_sm`)."
        )
    else:
        icon = "🟢" if graph_enabled else "🟡"
        st.markdown(
            f"{icon} **Knowledge Graph:** "
            f"{entity_count:,} entities | {relation_count:,} relations | "
            f"{docs_indexed} document(s)"
        )


def render_sources(results: list[dict[str, Any]]) -> None:
    """Display each retrieved chunk as a styled source card.

    Args:
        results: List of result dicts (text, document_name, page_number, chunk_id, score).
                 score must be a cosine similarity float in [0.0, 1.0].
    """
    if not results:
        st.info("No source chunks to display.")
        return

    render_step_header(
        5,
        "View Sources",
        "Explore retrieved evidence and confidence scores",
    )

    for i, result in enumerate(results, start=1):
        render_source_card(result, i)
