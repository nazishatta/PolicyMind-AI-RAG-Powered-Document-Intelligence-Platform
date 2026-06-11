"""Streamlit chat section — question input, query-type badge, confidence display,
and three-section answer layout (persisted across re-runs via session state)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from langchain_community.vectorstores import Chroma

from app.components.ui_helpers import render_step_header
from src.logger import get_logger
from src.rag_chain import answer_question_with_rag

logger = get_logger(__name__)

# Session-state key used to persist the last answer across Streamlit re-runs
_RESULT_KEY = "_chat_last_rag_result"


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_answer_type_badge(answer_type: str) -> None:
    """Render a colour-coded alert as the answer-type badge."""
    at_lower = answer_type.lower()

    if any(keyword in at_lower for keyword in ("summary", "map_reduce", "graph_rag", "document")):
        if answer_type == "single_document_summary":
            st.success("Answer Type: Single Document Summary  ✓  (grounded in uploaded document)")
        elif answer_type == "map_reduce":
            st.success("Answer Type: Map-Reduce Analysis  ✓  (multi-document synthesis)")
        elif answer_type == "graph_rag":
            st.success("Answer Type: GraphRAG Answer  ✓  (knowledge graph + document evidence)")
        elif answer_type == "Document Answer":
            st.success(f"Answer Type: {answer_type}  ✓  (grounded in uploaded documents)")
        else:
            st.success(f"Answer Type: {answer_type}  ✓  (document-grounded)")

    elif answer_type == "Partial Answer":
        st.warning(
            f"Answer Type: {answer_type}  ⚠  "
            "(document evidence supplemented with general knowledge)"
        )

    elif answer_type == "Evidence Only":
        st.info(
            f"Answer Type: {answer_type}  ℹ  "
            "(no API key configured; showing raw retrieved text)"
        )

    elif answer_type == "General Answer":
        st.info(
            f"Answer Type: {answer_type}  ℹ  "
            "(low document relevance — see note below)"
        )

    elif answer_type in ("Error", "No Corpus"):
        st.error(f"Answer Type: {answer_type}")

    else:
        st.error(f"Answer Type: {answer_type}")


_QUERY_TYPE_LABELS: dict[str, str] = {
    "summarization": "Summarization Request",
    "factual": "Factual Question",
    "analytical": "Analytical Question",
    "general": "General Question",
    "question": "Question",
}

_PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI GPT-4o-mini",
    "groq": "Groq LLaMA 3",
    "mock": "Mock (offline)",
    "anthropic": "Anthropic Claude",
    "none": "Evidence Only",
}

_EVIDENCE_QUALITY_COLOURS: dict[str, str] = {
    "strong": "green",
    "moderate": "orange",
    "weak": "orange",
    "insufficient": "red",
}

_ANSWER_TYPE_LABELS: dict[str, str] = {
    "cited": "Cited Answer — fully grounded in documents",
    "partial": "Partial Answer — limited evidence",
    "refused": "Refused — insufficient context",
    "no_corpus": "No Corpus — no documents indexed",
    "Error": "Error",
}


def _render_rag_result(rag_result: dict[str, Any]) -> None:
    """Render the full answer layout with styled card, progress bar, and metrics."""
    answer = rag_result.get("answer", "")
    answer_type = rag_result.get("answer_type", "General Answer")
    query_type = rag_result.get("query_type", "general")
    confidence = float(rag_result.get("confidence", 0.0))
    fallback_used = bool(rag_result.get("fallback_used", False))
    sources = rag_result.get("sources", [])

    # ------------------------------------------------------------------ #
    # Section A — Answer                                                   #
    # ------------------------------------------------------------------ #
    st.markdown("---")

    # Routing decision badge
    routing = rag_result.get("routing_decision", "standard_rag")
    if routing == "map_reduce":
        st.info("🗂️ **Routing:** Map-Reduce — analyzing each document independently")
    elif routing == "graph_rag":
        st.info("🧠 **Routing:** GraphRAG — using knowledge graph + vector search")
    else:
        st.info("🔍 **Routing:** Standard RAG — semantic vector search")

    # Query-type caption badge
    query_label = _QUERY_TYPE_LABELS.get(query_type, query_type.title())
    st.caption(f"Query detected as: **{query_label}**")

    _render_answer_type_badge(answer_type)

    # Answer wrapped in styled card
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Answer")
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

    provider = rag_result.get("provider", "none")
    provider_label = _PROVIDER_LABELS.get(provider, "Unknown")
    st.caption(f"Powered by: {provider_label}")

    if fallback_used:
        st.info(
            "This response supplements document evidence with general AI knowledge. "
            "Always verify important facts against authoritative sources."
        )

    # ------------------------------------------------------------------ #
    # Map-Reduce per-document analysis (when Map-Reduce was used)         #
    # ------------------------------------------------------------------ #
    per_doc_summaries = rag_result.get("per_doc_summaries", [])
    docs_successful = rag_result.get("docs_successful", 0)
    docs_failed = rag_result.get("docs_failed", 0)
    document_errors = rag_result.get("document_errors", [])

    if per_doc_summaries and answer_type in ("map_reduce", "single_document_summary"):
        successful_summaries = [s for s in per_doc_summaries if s.get("success", False)]
        if successful_summaries:
            with st.expander(
                f"Per-Document Analysis ({docs_successful} successful, "
                f"{docs_failed} failed)" if docs_failed > 0
                else f"Per-Document Analysis ({docs_successful} documents)",
                expanded=False,
            ):
                for summary in successful_summaries:
                    st.subheader(summary["doc_name"])
                    st.write(summary.get("summary", "(No analysis available)"))
                    st.caption(
                        f"Chunks analyzed: {summary['chunks_used']}  |  "
                        f"Avg relevance: {summary['avg_score']:.0%}"
                    )
                    st.divider()

                if docs_failed > 0:
                    st.warning(f"{docs_failed} document(s) failed to process")

        if document_errors:
            with st.expander("Debug: Document processing errors", expanded=False):
                for error in document_errors:
                    if error:
                        st.error(f"• {error}")

    # ------------------------------------------------------------------ #
    # Graph evidence (when GraphRAG was used)                              #
    # ------------------------------------------------------------------ #
    graph_evidence = rag_result.get("graph_evidence", [])
    graph_boost_used = rag_result.get("graph_boost_used", False)
    entities_found = rag_result.get("entities_found", [])

    if graph_evidence:
        with st.expander(
            f"Knowledge Graph Evidence ({len(graph_evidence)} entity relationships)",
            expanded=False,
        ):
            if entities_found:
                st.caption(f"Query entities detected: {', '.join(entities_found)}")
            for triple in graph_evidence:
                entity = triple.get("entity", "")
                relation = triple.get("relation", "CO_OCCURS_WITH")
                target = triple.get("target", "")
                conf = float(triple.get("confidence", 0.0))
                source = triple.get("source", "")
                st.markdown(f"**{entity}** → *{relation}* → **{target}**")
                st.progress(min(conf, 1.0))
                caption = f"Confidence: {conf:.0%}"
                if source and source != "unknown":
                    caption += f"  |  Source: {source}"
                st.caption(caption)
                st.divider()

    if graph_boost_used:
        st.caption("Retrieval enhanced by knowledge graph entity relationships")

    # ------------------------------------------------------------------ #
    # Section B — Confidence progress bar + metrics                        #
    # ------------------------------------------------------------------ #
    st.markdown("---")

    conf_pct = int(confidence * 100)
    if confidence >= 0.65:
        bar_color = "#28a745"
        conf_label = f"Strong Evidence ({conf_pct}%)"
    elif confidence >= 0.35:
        bar_color = "#ffc107"
        conf_label = f"Moderate Evidence ({conf_pct}%)"
    else:
        bar_color = "#dc3545"
        conf_label = f"Weak Evidence ({conf_pct}%)"

    st.markdown(f"""
<div style="margin:1rem 0;">
    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
        <span style="font-weight:bold; color:#444;">Retrieval Confidence</span>
        <span style="color:{bar_color}; font-weight:bold;">{conf_label}</span>
    </div>
    <div style="background:#e9ecef; border-radius:6px; height:10px;">
        <div style="background:{bar_color}; width:{conf_pct}%; height:10px;
                    border-radius:6px; transition:width 0.5s ease;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    sources_count = len(rag_result.get("sources_used", []))
    col1.metric("Sources Found", sources_count)
    col2.metric("Answer Type", answer_type.replace("_", " ").title())
    col3.metric(
        "Provider",
        (rag_result.get("provider") or "N/A").replace("_", " ").title(),
    )

    # ------------------------------------------------------------------ #
    # Section C — Sources expander                                         #
    # ------------------------------------------------------------------ #
    if sources:
        with st.expander(f"View Sources ({len(sources)})"):
            for i, src in enumerate(sources, start=1):
                st.markdown(f"**Source {i}:**")
                st.text(src)
                if i < len(sources):
                    st.markdown("---")


# ---------------------------------------------------------------------------
# GraphRAG answer renderer
# ---------------------------------------------------------------------------

_GRAPH_RESULT_KEY = "_graph_last_rag_result"


def render_graph_answer(result: dict[str, Any]) -> None:
    """Render a GraphRAG result with full three-section layout."""
    answer = result.get("answer", "")
    answer_type = result.get("answer_type", "cited")
    provider = result.get("provider", "none")
    confidence = result.get("confidence")
    graph_evidence = result.get("graph_evidence", [])
    citations = result.get("citations", [])
    evidence_quality = result.get("evidence_quality", "insufficient")
    limitations = result.get("limitations", [])
    latency_ms = result.get("latency_ms", 0.0)
    graph_used = result.get("graph_used", False)

    # ------------------------------------------------------------------ #
    # Section A — Answer                                                   #
    # ------------------------------------------------------------------ #
    st.markdown("---")

    at_label = _ANSWER_TYPE_LABELS.get(answer_type, answer_type.title())
    if answer_type == "cited":
        st.success(f"GraphRAG Answer Type: {at_label}  ✓")
    elif answer_type == "partial":
        st.warning(f"GraphRAG Answer Type: {at_label}  ⚠")
    elif answer_type in ("refused", "no_corpus"):
        st.error(f"GraphRAG Answer Type: {at_label}")
    else:
        st.info(f"GraphRAG Answer Type: {at_label}")

    st.markdown("#### Answer")
    st.markdown(answer)

    provider_label = _PROVIDER_LABELS.get(provider, provider.title())
    graph_label = "GraphRAG + " if graph_used else "GraphRAG (vector-only) + "
    st.caption(f"Powered by: {graph_label}{provider_label}")

    if limitations:
        with st.expander(f"Limitations / Caveats ({len(limitations)})", expanded=False):
            for lim in limitations:
                st.warning(lim)

    # ------------------------------------------------------------------ #
    # Section B — Metrics                                                  #
    # ------------------------------------------------------------------ #
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        if confidence is not None:
            conf_pct = f"{confidence:.1%}"
            colour = _EVIDENCE_QUALITY_COLOURS.get(evidence_quality, "red")
            if colour == "green":
                st.success(f"Confidence: **{conf_pct}** — {evidence_quality.upper()}")
            elif colour == "orange":
                st.warning(f"Confidence: **{conf_pct}** — {evidence_quality.upper()}")
            else:
                st.error(f"Confidence: **{conf_pct}** — {evidence_quality.upper()}")
        else:
            st.error("Confidence: **N/A** — no passages retrieved")

        st.metric("Sources Retrieved", len(citations))

    with col_right:
        st.markdown("**Knowledge Graph**")
        if graph_used:
            st.markdown(f":green[Active — {len(graph_evidence)} graph triple(s) used]")
        else:
            st.markdown(":blue[No graph triples matched this query]")

        st.markdown("**Latency**")
        st.markdown(f"{latency_ms:.0f} ms")

    # ------------------------------------------------------------------ #
    # Section C — Graph evidence + citations                               #
    # ------------------------------------------------------------------ #
    if graph_evidence:
        with st.expander(f"Knowledge Graph Evidence ({len(graph_evidence)} triples)"):
            for ev in graph_evidence:
                entity = ev.get("entity", "")
                relation = ev.get("relation", "")
                target = ev.get("target", "")
                conf = ev.get("confidence", 1.0)
                st.markdown(f"**{entity}** → *{relation}* → **{target}**")
                st.caption(f"Confidence: {conf:.0%}")
                st.divider()

    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            for i, cit in enumerate(citations, start=1):
                doc = cit.get("doc_title", "Unknown")
                page = cit.get("page_number", "?")
                score = cit.get("relevance_score", 0.0)
                excerpt = cit.get("excerpt", "")
                st.markdown(f"**[{i}] {doc}** — Page {page}  |  Relevance: {score:.1%}")
                if excerpt:
                    st.code(excerpt[:400], language="")
                if i < len(citations):
                    st.markdown("---")


# ---------------------------------------------------------------------------
# Main component
# ---------------------------------------------------------------------------

def render_chat_section(
    vector_store: Chroma,
    document_filter: str | None = None,
    graph_pipeline: Any = None,
) -> dict[str, Any] | None:
    """Render the Step 4 question-answering panel.

    Handles question input, RAG execution, and the full three-section answer layout.
    Uses st.session_state internally so the answer persists across Streamlit re-runs.

    When ``graph_pipeline`` is provided and ready, routes through
    ``answer_question_with_graph_rag()`` (hybrid vector + graph retrieval).
    Falls back to ``answer_question_with_rag()`` otherwise.

    Args:
        vector_store:    Loaded Chroma vector store used for retrieval.
        document_filter: If set, restrict retrieval to a single document by
            exact document_name match.
        graph_pipeline:  Optional GraphRAGPipeline instance.  When not None and
            ``graph_pipeline.is_ready`` is True, GraphRAG mode is used.

    Returns:
        The current rag_result dict (latest answer in session state), or None if
        no answer has been generated yet.
    """
    render_step_header(
        4,
        "Ask AI a Question",
        "Get grounded answers with source citations",
    )

    if graph_pipeline is not None and graph_pipeline.is_ready:
        _spacy_note = (
            "" if graph_pipeline.spacy_available
            else " (vector-only — install spaCy for entity extraction)"
        )
        st.caption(f"Mode: **GraphRAG** — knowledge graph + vector search{_spacy_note}")

    question = st.text_input(
        "Enter your question about the uploaded documents:",
        placeholder=(
            "e.g. What are the data retention obligations? "
            "Or: Summarize the key points of this document."
        ),
        key="ai_question_input",
    )

    clicked = st.button("Get Answer", key="get_answer_btn")

    if clicked:
        if not question.strip():
            st.warning("Please enter a question before clicking Get Answer.")
        else:
            try:
                from src.map_reduce_rag import route_and_answer  # noqa: PLC0415

                _use_graph = graph_pipeline is not None and graph_pipeline.is_ready
                spinner_msg = (
                    "Running hybrid GraphRAG pipeline (vector + graph)…"
                    if _use_graph
                    else "Searching documents and generating answer…"
                )
                with st.spinner(spinner_msg):
                    rag_result = route_and_answer(
                        question,
                        vector_store,
                        graph_pipeline=graph_pipeline,
                        document_filter=document_filter,
                    )
                st.session_state[_RESULT_KEY] = rag_result
                logger.info(
                    "Answer generated: routing=%s, answer_type=%s, confidence=%.4f",
                    rag_result.get("routing_decision", "unknown"),
                    rag_result.get("answer_type"),
                    rag_result.get("confidence", 0.0),
                )
            except Exception as exc:
                st.error(f"Failed to generate an answer: {exc}")
                logger.error("render_chat_section error: %s", exc)

    # Always render the most recent stored result (survives page re-runs)
    stored = st.session_state.get(_RESULT_KEY)
    if stored is not None:
        _render_rag_result(stored)

    return stored
