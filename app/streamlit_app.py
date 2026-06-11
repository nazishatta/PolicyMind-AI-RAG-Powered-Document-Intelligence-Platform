"""PolicyMind AI — main Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `app` are importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.components.chat_ui import render_chat_section
from app.components.source_display import render_graph_panel, render_sources
from app.components.upload_ui import render_upload_section
from app.components.ui_helpers import render_step_header
from src.logger import get_logger
from src.text_splitter import split_documents_into_chunks
from src.vector_store import create_vector_store, get_collection_stats, reset_vector_store
from src.rag_chain import check_api_keys

logger = get_logger(__name__)

# GraphRAG — self-contained, soft import (spaCy degrades gracefully if absent)
_GRAPH_RAG_AVAILABLE = False
try:
    from src.graph_rag import GraphRAGPipeline, answer_question_with_graph_rag  # noqa: F401
    _GRAPH_RAG_AVAILABLE = True
except Exception as _graph_import_err:
    logger.warning("src.graph_rag import failed: %s", _graph_import_err)

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PolicyMind AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}
/* Step cards */
.step-card {
    background: #f8f9fa;
    border-left: 4px solid #2d6a9f;
    padding: 1rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}
/* Answer card */
.answer-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
/* Metric card */
.metric-card {
    background: #f0f7ff;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
/* Source badge strong */
.badge-strong {
    background: #d4edda;
    color: #155724;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}
/* Source badge moderate */
.badge-moderate {
    background: #fff3cd;
    color: #856404;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}
/* Source badge weak */
.badge-weak {
    background: #f8d7da;
    color: #721c24;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}
/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state initialization (one-time defaults)
# ---------------------------------------------------------------------------
if "graph_pipeline" not in st.session_state:
    st.session_state["graph_pipeline"] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 PolicyMind AI")
    st.markdown("*Intelligent Policy Document Analysis*")
    st.divider()

    st.markdown("### 📋 How It Works")
    st.markdown("""
1. 📄 **Upload** policy PDF documents
2. 🧠 **Index** with ChromaDB + Knowledge Graph
3. 🔎 **Search** semantically across documents
4. 💬 **Ask** natural language questions
5. 📌 **Get** grounded answers with citations
""")
    st.divider()

    st.markdown("### ⚙️ RAG Mode")
    _last_result = st.session_state.get("rag_result")
    if _last_result:
        _routing = _last_result.get("routing_decision", "standard_rag")
        if _routing == "map_reduce":
            st.info("🗺️ Map-Reduce RAG active")
        elif _routing == "graph_rag":
            st.success("🕸️ GraphRAG active")
        else:
            st.info("🔍 Standard RAG active")
    else:
        st.info("🔍 Standard RAG active")
    st.divider()

    st.markdown("### 🛠️ Tech Stack")
    _tech_data = {
        "Layer": ["UI", "Embeddings", "Vector DB", "Graph", "LLM", "PDF"],
        "Technology": ["Streamlit", "all-MiniLM-L6-v2", "ChromaDB",
                       "NetworkX", "Groq LLaMA 3", "PyMuPDF"],
    }
    st.dataframe(_tech_data, hide_index=True, use_container_width=True)
    st.divider()

    # API status
    st.markdown("### 🔑 API Status")
    _api_status = check_api_keys()
    st.caption(f"Config: `{_api_status['env_path']}`")
    if _api_status["env_exists"]:
        st.success("✓ .env file found")
    else:
        st.warning("⚠ .env file not found")
    if _api_status["groq_available"]:
        st.success(f"✓ Groq connected ({_api_status['groq_key_length']} chars)")
    else:
        st.info("○ Groq not configured")
    if _api_status["openai_available"]:
        st.success(f"✓ OpenAI connected ({_api_status['openai_key_length']} chars)")
    else:
        st.info("○ OpenAI not configured")

    # Graph stats (when pipeline is active)
    _sidebar_gp = st.session_state.get("graph_pipeline")
    if _sidebar_gp is not None and _sidebar_gp.is_ready:
        _sidebar_gs = _sidebar_gp.knowledge_graph.get_stats()
        st.success(
            f"🕸️ Graph active: {_sidebar_gs['entity_count']:,} entities | "
            f"{_sidebar_gs['relation_count']:,} relations"
        )
    st.divider()

    # Session stats
    st.markdown("### 📊 Session Stats")
    _pages_ss = st.session_state.get("pages", [])
    _chunks_ss = st.session_state.get("chunks", [])
    _doc_names_ss = list({p.get("document_name", "") for p in _pages_ss if p.get("document_name")})

    _sc1, _sc2 = st.columns(2)
    _sc1.metric("Documents", len(_doc_names_ss))
    _sc2.metric("Pages", len(_pages_ss))
    _sc1.metric("Chunks", len(_chunks_ss))
    if _sidebar_gp is not None and _sidebar_gp.is_ready:
        _gs2 = _sidebar_gp.knowledge_graph.get_stats()
        _sc2.metric("Entities", _gs2["entity_count"])
    st.divider()

    if st.button("Reset session", help="Clear all uploaded data and start over"):
        for _key in (
            "pages", "chunks", "vector_store",
            "search_results", "rag_result",
            "_chat_last_rag_result",
            "graph_pipeline", "use_graph_rag",
        ):
            st.session_state.pop(_key, None)
        st.rerun()

    st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8em; margin-top:1rem;'>
Built by Nazish Atta<br>
MS Data Science · GWU<br>
<a href='https://github.com/nazishatta'>GitHub</a>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="color:white; margin:0; font-size:2.2rem;">
        🔍 PolicyMind AI
    </h1>
    <p style="color:#b8d4f0; margin:0.5rem 0 0 0; font-size:1.1rem;">
        RAG-Powered Document Intelligence Platform
    </p>
    <p style="color:#8ab8e0; margin:0.3rem 0 0 0; font-size:0.9rem;">
        GraphRAG • Map-Reduce • Semantic Search • Grounded Citations
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 1 — Upload document
# ---------------------------------------------------------------------------
pages = render_upload_section()
if pages is not None:
    st.session_state["pages"] = pages

# ---------------------------------------------------------------------------
# Step 2 — Build knowledge base
# ---------------------------------------------------------------------------
render_step_header(
    2,
    "Build Knowledge Base",
    "Index documents with vector embeddings and knowledge graph",
)

if "pages" not in st.session_state:
    st.markdown("""
<div style="text-align:center; padding:2rem;
            background:#fff3cd; border-radius:12px; margin:1rem 0;">
    <h4 style="margin:0 0 0.5rem 0;">⚠️ No documents loaded yet</h4>
    <p style="margin:0;">Complete Step 1 first: upload a document</p>
</div>
""", unsafe_allow_html=True)
else:
    use_graph_rag = st.toggle(
        "Use GraphRAG (entity extraction + knowledge graph)",
        value=st.session_state.get("use_graph_rag", False),
        key="use_graph_rag_toggle",
        help=(
            "When enabled, builds a knowledge graph alongside the vector index. "
            "Requires spaCy (en_core_web_sm) for entity extraction; "
            "falls back to vector-only GraphRAG if spaCy is not installed. "
            "Disable to use the standard ChromaDB vector store."
        ),
    )
    st.session_state["use_graph_rag"] = use_graph_rag

    if use_graph_rag and not _GRAPH_RAG_AVAILABLE:
        st.warning(
            "GraphRAG module failed to import "
            "(check requirements.txt — networkx and spacy are required). "
            "Standard RAG will be used."
        )

    st.info(
        "If relevance scores seem low (all below 40%), click "
        "**Reset & Rebuild Knowledge Base** to fix the vector database metric."
    )

    col_build, col_reset = st.columns(2)
    with col_build:
        build_clicked = st.button("Build Knowledge Base", key="build_kb_btn")
    with col_reset:
        reset_clicked = st.button(
            "Reset & Rebuild Knowledge Base",
            key="reset_kb_btn",
            help=(
                "Deletes the existing ChromaDB collection and rebuilds it with "
                "the correct cosine distance metric. Required if scores were wrong."
            ),
        )

    def _build_kb(reset_first: bool = False) -> None:
        """Build the standard vector store, then optionally build the knowledge graph."""
        # -- Standard vector store (always) ----------------------------------
        try:
            with st.spinner(
                ("Resetting collection… " if reset_first else "")
                + "Chunking text and building vector store…"
            ):
                if reset_first:
                    reset_vector_store()
                chunks = split_documents_into_chunks(st.session_state["pages"])
                vector_store = create_vector_store(chunks)

            st.session_state["chunks"] = chunks
            st.session_state["vector_store"] = vector_store
            st.session_state["graph_pipeline"] = None  # always reset graph on rebuild
            st.session_state.pop("rag_result", None)
            st.session_state.pop("_chat_last_rag_result", None)

            prefix = "Reset & rebuilt" if reset_first else "Built"
            st.success(f"{prefix} knowledge base successfully.")

            kb_stats = get_collection_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents Indexed", kb_stats["total_documents"])
            c2.metric("Total Chunks", kb_stats["total_chunks"])
            c3.metric(
                "Documents",
                ", ".join(kb_stats["documents"]) if kb_stats["documents"] else "—",
            )

        except Exception as exc:
            st.error(f"Failed to build knowledge base: {exc}")
            logger.error("Knowledge base build error (reset=%s): %s", reset_first, exc)
            return  # Don't attempt graph build if vector store failed

        # -- Optional graph build on the same chunks -------------------------
        if use_graph_rag and _GRAPH_RAG_AVAILABLE:
            try:
                from src.graph_rag import GraphRAGPipeline  # noqa: PLC0415

                with st.spinner("Extracting entities and building knowledge graph…"):
                    gp = GraphRAGPipeline()
                    g_result = gp.ingest_chunks_to_graph(chunks)
                    st.session_state["graph_pipeline"] = gp

                entities = g_result["entities_extracted"]
                relations = g_result["relations_created"]
                spacy_used = g_result["spacy_used"]

                if entities > 0:
                    st.success(
                        f"GraphRAG: {entities:,} entities extracted, "
                        f"{relations:,} relations built."
                    )
                else:
                    st.warning(
                        "GraphRAG: no entities extracted. "
                        "Install spaCy model: `python -m spacy download en_core_web_sm`"
                    )

                if not spacy_used:
                    st.info(
                        "spaCy model not found — GraphRAG is in vector-only mode. "
                        "Run: `python -m spacy download en_core_web_sm`"
                    )

                g_stats = gp.knowledge_graph.get_stats()
                gc1, gc2 = st.columns(2)
                gc1.metric("Graph Entities", g_stats["entity_count"])
                gc2.metric("Graph Relations", g_stats["relation_count"])
                render_graph_panel({
                    "entity_count": g_stats["entity_count"],
                    "relation_count": g_stats["relation_count"],
                    "graph_enabled": True,
                    "documents_indexed": kb_stats["total_documents"],
                })

            except Exception as exc:
                st.warning(f"Graph build failed (standard RAG will be used): {exc}")
                logger.error("Graph build error: %s", exc)
                st.session_state["graph_pipeline"] = None

    if build_clicked:
        _build_kb(reset_first=False)

    if reset_clicked:
        _build_kb(reset_first=True)

# ---------------------------------------------------------------------------
# Step 3 — Semantic search
# ---------------------------------------------------------------------------
render_step_header(
    3,
    "Semantic Search",
    "Find relevant passages across your documents",
)

if "vector_store" not in st.session_state:
    st.markdown("""
<div style="text-align:center; padding:2rem;
            background:#fff3cd; border-radius:12px; margin:1rem 0;">
    <h4 style="margin:0 0 0.5rem 0;">⚠️ Knowledge base not built yet</h4>
    <p style="margin:0;">Click <strong>Reset &amp; Rebuild Knowledge Base</strong> to index your documents</p>
</div>
""", unsafe_allow_html=True)
else:
    _stats3 = get_collection_stats()
    _doc_names3 = _stats3["documents"]
    _search_doc_options = ["All Documents"] + _doc_names3
    search_doc_filter = st.selectbox(
        "Search in:",
        options=_search_doc_options,
        key="search_doc_filter",
        help="Restrict the search to a single document, or search across all documents.",
    )
    _search_filter_value = None if search_doc_filter == "All Documents" else search_doc_filter

    search_query = st.text_input(
        "Search for relevant passages:",
        placeholder="e.g. data retention, consent requirements, liability clauses",
        key="semantic_search_input",
    )

    if st.button("Search", key="search_btn"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            try:
                from src.retriever import assess_retrieval_quality, semantic_search  # noqa: PLC0415

                with st.spinner("Searching…"):
                    results = semantic_search(
                        search_query,
                        st.session_state["vector_store"],
                        document_filter=_search_filter_value,
                    )
                    quality = assess_retrieval_quality(results)

                st.session_state["search_results"] = results

                quality_label = quality["quality"].upper()
                avg_pct = quality["avg_score"] * 100
                st.success(
                    f"Found {len(results)} passage(s). "
                    f"Retrieval quality: **{quality_label}** "
                    f"(avg relevance {avg_pct:.1f}%)"
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                logger.error("Semantic search error: %s", exc)

# ---------------------------------------------------------------------------
# Step 4 — Ask AI a question
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    render_step_header(
        4,
        "Ask AI a Question",
        "Get grounded answers with source citations",
    )
    st.markdown("""
<div style="text-align:center; padding:2rem;
            background:#fff3cd; border-radius:12px; margin:1rem 0;">
    <h4 style="margin:0 0 0.5rem 0;">⚠️ Knowledge base not built yet</h4>
    <p style="margin:0;">Click <strong>Reset &amp; Rebuild Knowledge Base</strong> to index your documents</p>
</div>
""", unsafe_allow_html=True)
else:
    _stats4 = get_collection_stats()
    _doc_names4 = _stats4["documents"]
    _answer_doc_options = ["All Documents"] + _doc_names4
    answer_doc_filter = st.selectbox(
        "Answer from:",
        options=_answer_doc_options,
        key="answer_doc_filter",
        help="Restrict the AI answer to evidence from a single document, or use all documents.",
    )
    _answer_filter_value = None if answer_doc_filter == "All Documents" else answer_doc_filter

    _graph_pipeline = st.session_state.get("graph_pipeline")
    rag_result = render_chat_section(
        st.session_state["vector_store"],
        document_filter=_answer_filter_value,
        graph_pipeline=_graph_pipeline,
    )

    if rag_result is not None:
        st.session_state["rag_result"] = rag_result

# ---------------------------------------------------------------------------
# Step 5 — View sources (from the most recent search or AI answer)
# ---------------------------------------------------------------------------
latest_results: list = []

if "rag_result" in st.session_state:
    _rag_result = st.session_state["rag_result"]
    latest_results = _rag_result.get("sources_used") or _rag_result.get("results", [])
elif "search_results" in st.session_state:
    latest_results = st.session_state["search_results"]

if latest_results:
    render_sources(latest_results)

    # Show graph panel when the last answer was from GraphRAG
    _last_rag = st.session_state.get("rag_result", {})
    if _last_rag.get("answer_type") == "graph_rag":
        _active_gp = st.session_state.get("graph_pipeline")
        if _active_gp is not None and _active_gp.knowledge_graph is not None:
            _g = _active_gp.knowledge_graph.get_stats()
            _n_docs = (
                get_collection_stats()["total_documents"]
                if "vector_store" in st.session_state
                else 0
            )
            render_graph_panel({
                "entity_count": _g["entity_count"],
                "relation_count": _g["relation_count"],
                "graph_enabled": True,
                "documents_indexed": _n_docs,
            })
else:
    render_step_header(
        5,
        "View Sources",
        "Explore retrieved evidence and confidence scores",
    )
    st.info("Run a search or ask a question to see source evidence here.")
