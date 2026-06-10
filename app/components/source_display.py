"""Streamlit source display — colour-coded relevance bars and code-block excerpts.

Thresholds (calibrated for cosine similarity post-fix):
    STRONG   score >= 0.65  🟢
    MODERATE score >= 0.45  🟡
    WEAK     score <  0.45  🔴
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# Cosine-similarity thresholds matching src/retriever.py and chat_ui.py
_STRONG: float = 0.65
_MODERATE: float = 0.45


def _relevance_label(score: float) -> str:
    """Return a colour-coded quality emoji + label for a cosine similarity score."""
    if score >= _STRONG:
        return "🟢 STRONG"
    if score >= _MODERATE:
        return "🟡 MODERATE"
    return "🔴 WEAK"


def render_graph_panel(graph_stats: dict[str, Any]) -> None:
    """Display a compact knowledge-graph status panel.

    Shows entity count, relation count, and whether the graph is active.
    Intended to give a visual indicator that GraphRAG was used for the
    last query.

    Args:
        graph_stats: Dict returned by ``src.graph_bridge.get_graph_stats()``.
    """
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
        colour = "green" if graph_enabled else "orange"
        icon = "🟢" if graph_enabled else "🟡"
        st.markdown(
            f"{icon} **Knowledge Graph:** "
            f"{entity_count:,} entities | {relation_count:,} relations | "
            f"{docs_indexed} document(s)"
        )


def render_sources(results: list[dict[str, Any]]) -> None:
    """Display each retrieved chunk with relevance bar, metadata, and excerpt.

    Layout per chunk (inside an st.expander):
        Document name       — plain text line
        Page number         — plain text line
        Relevance score     — st.progress(float 0–1) + percentage caption
        Text excerpt        — st.code(excerpt, language="") grey block

    Expander title:
        Source N  |  <doc>  |  Page <n>  |  🟢 STRONG (72.3%)
        Colour coding: ≥65% STRONG, 45–65% MODERATE, <45% WEAK

    Args:
        results: List of result dicts (text, document_name, page_number, chunk_id, score).
                 score must be a cosine similarity float in [0.0, 1.0].
    """
    if not results:
        st.info("No source chunks to display.")
        return

    st.subheader("Step 5 — View Sources")

    for i, result in enumerate(results, start=1):
        doc_name = result.get("document_name", "Unknown")
        page_num = result.get("page_number", "?")
        # Score arrives as cosine similarity (0.0–1.0); clamp defensively
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        text = result.get("text", "")
        chunk_id = result.get("chunk_id", "")

        quality = _relevance_label(score)
        label = (
            f"Source {i}  |  [{doc_name}]  |  Page {page_num}"
            f"  |  {quality} ({score:.1%})"
        )

        with st.expander(label):
            # Metadata lines — document name is bold and prominent
            st.markdown(f"**Document: {doc_name}**")
            st.markdown(f"**Page number:** {page_num}")

            # Relevance progress bar
            # st.progress() requires a float in [0.0, 1.0] — score is already correct.
            st.markdown("**Relevance score:**")
            st.progress(score)
            st.caption(f"{score:.1%} cosine similarity  ({quality})")

            # Excerpt in a monospace grey code block
            st.markdown("**Text excerpt:**")
            excerpt = text[:600] + ("…" if len(text) > 600 else "")
            st.code(excerpt, language="")

            if chunk_id:
                st.caption(f"Chunk ID: {chunk_id}")
