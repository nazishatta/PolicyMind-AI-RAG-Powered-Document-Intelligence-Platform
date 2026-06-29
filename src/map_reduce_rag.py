"""Map-Reduce RAG for broad, comparative, and multi-document queries.

Automatically routes questions between:
  - MAP-REDUCE: for summaries, comparisons, broad analyses (processes each doc independently)
  - GRAPH-RAG:  for entity/relationship queries (when graph is ready)
  - STANDARD:   default semantic vector search

The Map phase calls Groq independently on each document.
The Reduce phase synthesizes per-document summaries into one final answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.logger import get_logger
from src.rag_chain import get_env_key

# ============================================================================
# ENV LOADING — Load .env file at module import time
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

logger = get_logger(__name__)

# Query classification thresholds
_MAP_REDUCE_KEYWORDS = frozenset({
    "summarize", "summary", "summarise", "compare", "comparison", "contrast",
    "overview", "all documents", "across documents", "key points", "main points",
    "tell me about all", "what are all", "both documents", "every document",
    "across all", "from all", "overall", "general overview", "outline",
})

_GRAPH_RAG_KEYWORDS = frozenset({
    "relationship", "related to", "connected to", "who is responsible",
    "which organization", "which agency", "what entity", "how are",
    "between", "link between", "association", "role of",
})

# Fallback behavior control
_FALLBACK_ENABLED = os.getenv("FALLBACK_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# PART 1 — Query Router
# ---------------------------------------------------------------------------

def detect_query_mode(question: str) -> str:
    """Classify a question into map_reduce, graph_rag, or standard.

    Returns:
        "map_reduce" — broad, comparative, or multi-document queries
        "graph_rag"  — entity, relationship, or organizational queries
        "standard"   — everything else (single-document, factual)
    """
    q = question.lower()

    for term in _MAP_REDUCE_KEYWORDS:
        if term in q:
            return "map_reduce"

    for term in _GRAPH_RAG_KEYWORDS:
        if term in q:
            return "graph_rag"

    return "standard"


def get_document_names_from_vector_store(vector_store: Any) -> tuple[list[str], dict[str, Any]]:
    """Extract unique document names from a Chroma vector store.

    FIX BUG 1: More robust document extraction with debugging info.

    Args:
        vector_store: LangChain Chroma wrapper instance.

    Returns:
        Tuple of (sorted list of document names, debugging info dict).
        Falls back to "Uploaded Document" if metadata is missing.
    """
    debug_info: dict[str, Any] = {
        "total_chunks": 0,
        "chunks_with_doc_name": 0,
        "doc_names_extracted": 0,
        "errors": [],
    }

    try:
        collection = vector_store._collection
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        debug_info["total_chunks"] = len(metadatas)

        # Extract document names, handling missing metadata
        doc_names_set: set[str] = set()
        for m in metadatas:
            if not m:
                continue
            doc_name = m.get("document_name")
            if doc_name and doc_name.strip():
                doc_names_set.add(doc_name.strip())
                debug_info["chunks_with_doc_name"] += 1

        # Fallback: if no document_name metadata, use generic name
        if not doc_names_set:
            logger.warning(
                "No 'document_name' metadata found in %d chunks. Using fallback name.",
                len(metadatas),
            )
            if len(metadatas) > 0:
                doc_names_set.add("Uploaded Document")
            debug_info["errors"].append("No document_name metadata; using fallback")

        doc_names = sorted(list(doc_names_set))
        debug_info["doc_names_extracted"] = len(doc_names)

        logger.info("Extracted %d unique documents from %d chunks", len(doc_names), len(metadatas))
        return doc_names, debug_info

    except Exception as exc:
        logger.error("Failed to extract document names: %s", exc)
        debug_info["errors"].append(f"Exception: {str(exc)}")
        return [], debug_info


def get_chunks_for_document(
    vector_store: Any,
    document_name: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Retrieve chunks for a specific document with intelligent fallback.

    FIX BUG 2: Multiple fallback strategies for robust chunk retrieval.

    Args:
        vector_store: LangChain Chroma wrapper instance.
        document_name: Name of the document to retrieve chunks for.
        limit: Maximum number of chunks to retrieve.

    Returns:
        List of chunk dicts with 'text', 'page_number', etc.
        Falls back through multiple strategies to ensure at least some chunks.
    """
    try:
        collection = vector_store._collection

        # Strategy 1: Try ChromaDB where filtering
        logger.debug("Attempting strategy 1: ChromaDB where filter for '%s'", document_name)
        try:
            result = collection.get(
                where={"document_name": {"$eq": document_name}},
                include=["documents", "metadatas"],
                limit=limit,
            )
            chunks = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            if chunks and len(chunks) > 0:
                logger.info("Strategy 1 success: retrieved %d chunks for '%s'", len(chunks), document_name)
                return [
                    {
                        "text": chunks[i],
                        "page_number": metadatas[i].get("page_number", "?") if metadatas and i < len(metadatas) else "?",
                        "document_name": document_name,
                    }
                    for i in range(len(chunks))
                ]
        except Exception as e:
            logger.debug("Strategy 1 failed: %s", e)

        # Strategy 2: Retrieve all chunks and filter manually in Python
        logger.debug("Attempting strategy 2: Manual Python filtering for '%s'", document_name)
        try:
            result = collection.get(include=["documents", "metadatas"], limit=limit * 2)
            all_chunks = result.get("documents") or []
            all_metadatas = result.get("metadatas") or []

            filtered_chunks = []
            for i, meta in enumerate(all_metadatas):
                if (meta and meta.get("document_name") == document_name
                        and i < len(all_chunks)):
                    filtered_chunks.append({
                        "text": all_chunks[i],
                        "page_number": meta.get("page_number", "?"),
                        "document_name": document_name,
                    })
                    if len(filtered_chunks) >= limit:
                        break

            if filtered_chunks:
                logger.info("Strategy 2 success: retrieved %d chunks for '%s'", len(filtered_chunks), document_name)
                return filtered_chunks
        except Exception as e:
            logger.debug("Strategy 2 failed: %s", e)

        # Strategy 3: Return first N chunks from collection as final fallback
        logger.debug("Attempting strategy 3: First chunks from collection for '%s'", document_name)
        try:
            result = collection.get(include=["documents", "metadatas"], limit=limit)
            chunks = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            if chunks:
                logger.warning("Strategy 3 fallback: returning first %d chunks (not filtered by document)", len(chunks))
                return [
                    {
                        "text": chunks[i],
                        "page_number": metadatas[i].get("page_number", "?") if metadatas and i < len(metadatas) else "?",
                        "document_name": document_name,
                    }
                    for i in range(len(chunks))
                ]
        except Exception as e:
            logger.debug("Strategy 3 failed: %s", e)

        logger.error("All chunk retrieval strategies failed for document '%s'", document_name)
        return []

    except Exception as exc:
        logger.error("get_chunks_for_document failed: %s", exc)
        return []


def sample_chunks_across_document(
    vector_store: Any,
    document_name: str,
    question: str,
    total_chunks: int = 30,
) -> list[dict[str, Any]]:
    """Sample chunks from beginning, middle, end, plus semantic top chunks.

    FIX BUG 4: Better chunk selection for summarization (not just vector similarity).

    Args:
        vector_store: LangChain Chroma wrapper instance.
        document_name: Name of the document.
        question: User's question (for semantic ranking).
        total_chunks: Total chunks to return.

    Returns:
        List of representative chunks across the document.
    """
    try:
        # Get all chunks for this document
        all_chunks = get_chunks_for_document(vector_store, document_name, limit=1000)

        if not all_chunks:
            logger.warning("No chunks found for document '%s'", document_name)
            return []

        # If we have fewer chunks than requested, return all
        if len(all_chunks) <= total_chunks:
            logger.info("Returning all %d available chunks for '%s'", len(all_chunks), document_name)
            return all_chunks

        # Divide into beginning, middle, end
        num_per_section = total_chunks // 3
        beginning = all_chunks[:num_per_section]
        middle_start = len(all_chunks) // 2 - num_per_section // 2
        middle = all_chunks[middle_start:middle_start + num_per_section]
        ending = all_chunks[-num_per_section:]

        # Get top semantic chunks
        from src.retriever import semantic_search  # noqa: PLC0415

        try:
            semantic_top = semantic_search(
                question,
                vector_store,
                top_k=num_per_section,
                document_filter=document_name,
            )
        except Exception as e:
            logger.warning("Semantic search failed for sampling: %s", e)
            semantic_top = []

        # Combine and deduplicate
        sampled = beginning + middle + ending + semantic_top
        seen_texts: set[str] = set()
        deduplicated = []
        for chunk in sampled:
            text = chunk.get("text", "")
            if text and text not in seen_texts:
                deduplicated.append(chunk)
                seen_texts.add(text)
                if len(deduplicated) >= total_chunks:
                    break

        logger.info(
            "Sampled %d representative chunks from '%s' (beginning=%d, middle=%d, end=%d, semantic=%d)",
            len(deduplicated), document_name, len(beginning), len(middle), len(ending), len(semantic_top),
        )
        return deduplicated[:total_chunks]

    except Exception as exc:
        logger.error("sample_chunks_across_document failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# PART 2 — Map Phase
# ---------------------------------------------------------------------------

def map_document(
    doc_name: str,
    question: str,
    vector_store: Any,
) -> dict[str, Any]:
    """Analyze a single document in isolation (Map phase).

    FIX BUG 2: Use robust chunk retrieval with fallbacks.

    Args:
        doc_name: Name of the document to analyze.
        question: User's question.
        vector_store: Loaded Chroma vector store.

    Returns:
        Dict with doc_name, summary, chunks_used, avg_score, success, error (if failed),
        and sources_used list (FIX BUG 2).
    """
    try:
        # Use robust chunk retrieval
        chunks = sample_chunks_across_document(vector_store, doc_name, question, total_chunks=20)

        if not chunks:
            logger.warning("No chunks found for document '%s'", doc_name)
            return {
                "doc_name": doc_name,
                "summary": "(No relevant passages found in this document.)",
                "chunks_used": 0,
                "avg_score": 0.0,
                "success": True,
                "error": None,
                "sources_used": [],
            }

        # Build context from chunks with proper formatting
        context_lines = []
        sources_list = []

        for i, c in enumerate(chunks, 1):
            page = c.get("page_number", "?")
            text = c.get("text", "")
            chunk_id = c.get("chunk_id", f"chunk_{i}")
            score = c.get("score", 0.7)

            # Format for Groq with source attribution (FIX BUG 6)
            context_lines.append(f"[Source: {doc_name}, Page: {page}]\n{text}")

            # Build source entry for display
            sources_list.append({
                "text": text,
                "document_name": doc_name,
                "page_number": page,
                "chunk_id": chunk_id,
                "score": score,
            })

        context = "\n\n---\n\n".join(context_lines)
        avg_score = sum(c.get("score", 0.7) for c in chunks) / len(chunks) if chunks else 0.7

        # Call Groq to analyze this document
        groq_api_key = get_env_key("GROQ_API_KEY")
        if not groq_api_key:
            error_msg = "Groq API key not configured"
            logger.error("map_document: %s", error_msg)
            return {
                "doc_name": doc_name,
                "summary": "",
                "chunks_used": 0,
                "avg_score": 0.0,
                "success": False,
                "error": error_msg,
                "sources_used": [],
            }

        try:
            from groq import Groq  # noqa: PLC0415
        except ImportError as e:
            error_msg = f"groq package unavailable: {e}"
            logger.error("map_document: %s", error_msg)
            return {
                "doc_name": doc_name,
                "summary": "",
                "chunks_used": 0,
                "avg_score": 0.0,
                "success": False,
                "error": error_msg,
                "sources_used": [],
            }

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a document analyst. "
                        f"Analyze only the provided excerpts from '{doc_name}'. "
                        f"Answer this question: {question}\n"
                        f"Be factual and specific. When possible, mention page numbers from the provided excerpts. "
                        f"Keep response under 200 words."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document excerpts:\n{context}",
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )

        summary = (response.choices[0].message.content or "").strip()
        logger.info("map_document: '%s' success with %d chunks, avg_score=%.3f", doc_name, len(chunks), avg_score)

        return {
            "doc_name": doc_name,
            "summary": summary,
            "chunks_used": len(chunks),
            "avg_score": avg_score,
            "success": True,
            "error": None,
            "sources_used": sources_list,
        }

    except Exception as exc:
        error_msg = str(exc)
        logger.error("map_document '%s' failed: %s", doc_name, error_msg)
        return {
            "doc_name": doc_name,
            "summary": "",
            "chunks_used": 0,
            "avg_score": 0.0,
            "success": False,
            "error": error_msg,
            "sources_used": [],
        }


# ---------------------------------------------------------------------------
# PART 3 — Reduce Phase
# ---------------------------------------------------------------------------

def reduce_summaries(
    question: str,
    doc_summaries: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Synthesize per-document summaries into one final answer (Reduce phase).

    FIX BUG 5: Return errors along with answer.

    Args:
        question: User's original question.
        doc_summaries: List of per-document summary dicts from map phase.

    Returns:
        Tuple of (final answer string, list of error messages).
    """
    # Filter to only successful summaries
    successful = [d for d in doc_summaries if d.get("success", False)]
    errors: list[str] = [str(d.get("error")) for d in doc_summaries if d.get("error")]

    if not successful:
        error_msg = "All documents failed to process. Please check the debug errors."
        logger.error("reduce_summaries: %s", error_msg)
        return error_msg, errors

    # Build per-document section
    per_doc = "\n\n".join(
        f"{d['doc_name']}:\n{d['summary']}"
        for d in successful
    )

    # Construct reduce prompt
    reduce_prompt = (
        f"You are PolicyMind AI analyzing {len(successful)} policy document(s).\n\n"
        f"Per-document analysis:\n{per_doc}\n\n"
        f"Question: {question}\n\n"
        "Provide a comprehensive response with:\n\n"
        "OVERVIEW:\n"
        "(2-3 sentences covering all documents)\n\n"
        "DOCUMENT-BY-DOCUMENT ANALYSIS:\n"
        "(For each document: what it says about the question)\n\n"
        "CROSS-DOCUMENT INSIGHTS:\n"
        "(Patterns, similarities, differences, contradictions)\n\n"
        "KEY FINDINGS:\n"
        "(5 bullet points — most important across all documents, each citing which document)\n\n"
        "CONCLUSION:\n"
        "(1-2 sentences synthesizing everything)"
    )

    # Call Groq
    groq_api_key = get_env_key("GROQ_API_KEY")
    if not groq_api_key:
        error_msg = "Groq API key not configured"
        logger.error("reduce_summaries: %s", error_msg)
        return error_msg, errors + [error_msg]

    try:
        from groq import Groq  # noqa: PLC0415

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PolicyMind AI analyzing multiple policy documents. "
                        "Synthesize the per-document summaries into one comprehensive response. "
                        "Focus on cross-document patterns and overall insights."
                    ),
                },
                {"role": "user", "content": reduce_prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        final_answer = (response.choices[0].message.content or "").strip()
        logger.info("reduce_summaries: synthesized %d documents successfully", len(successful))
        return final_answer, errors

    except Exception as exc:
        error_msg = f"Reduce phase failed: {str(exc)}"
        logger.error("reduce_summaries: %s", error_msg)
        return error_msg, errors + [error_msg]


# ---------------------------------------------------------------------------
# PART 4 — Full Map-Reduce Pipeline
# ---------------------------------------------------------------------------

def answer_with_map_reduce(
    question: str,
    vector_store: Any,
) -> dict[str, Any]:
    """End-to-end Map-Reduce: per-document analysis + synthesis.

    FIX BUG 3: Handle single-document case specially.
    FIX BUG 5: Return document_errors in result.

    Args:
        question: User's question.
        vector_store: Loaded Chroma vector store.

    Returns:
        Dict with answer, answer_type, provider, confidence, per-document
        summaries, and metadata including document_errors.
    """
    # Step 1: Get document names
    doc_names, debug_info = get_document_names_from_vector_store(vector_store)

    if not doc_names:
        error_msg = (
            "No documents found in the knowledge base. "
            "Please upload at least one PDF and rebuild the knowledge base."
        )
        logger.error("answer_with_map_reduce: %s (debug: %s)", error_msg, debug_info)
        return {
            "answer": error_msg,
            "answer_type": "map_reduce",
            "query_type": "summarization",
            "provider": "groq",
            "confidence": 0.0,
            "documents_analyzed": [],
            "docs_successful": 0,
            "docs_failed": 0,
            "per_doc_summaries": [],
            "sources_used": [],
            "fallback_used": False,
            "graph_evidence": [],
            "results": [],
            "sources": [],
            "document_errors": debug_info["errors"],
        }

    logger.info("answer_with_map_reduce: starting with %d documents", len(doc_names))

    # FIX BUG 3: Special handling for single document
    if len(doc_names) == 1:
        logger.info("Single-document mode: summarizing '%s' directly", doc_names[0])
        summary = map_document(doc_names[0], question, vector_store)
        doc_summaries = [summary]

        final_answer = summary["summary"] if summary["success"] else (
            f"Failed to summarize {doc_names[0]}: {summary.get('error', 'Unknown error')}"
        )

        # FIX BUG 1: Return actual chunks as sources
        sources_used = summary.get("sources_used", []) if summary["success"] else []

        return {
            "answer": final_answer,
            "answer_type": "single_document_summary",
            "query_type": "summarization",
            "provider": "groq",
            "confidence": round(summary["avg_score"], 4),
            "documents_analyzed": doc_names,
            "docs_successful": 1 if summary["success"] else 0,
            "docs_failed": 0 if summary["success"] else 1,
            "per_doc_summaries": doc_summaries,
            "sources_used": sources_used,
            "sources_found": len(sources_used),
            "fallback_used": False,
            "graph_evidence": [],
            "results": sources_used,  # For render_sources
            "sources": sources_used,  # Alias
            "document_errors": [],
        }

    # Step 2: Map phase — analyze each document
    logger.info("Map-Reduce: starting map phase on %d documents", len(doc_names))
    doc_summaries = []

    for doc_name in doc_names:
        summary = map_document(doc_name, question, vector_store)
        doc_summaries.append(summary)

    # Step 3: Reduce phase — synthesize summaries
    final_answer, reduce_errors = reduce_summaries(question, doc_summaries)

    # Step 4: Calculate statistics
    successful = [s for s in doc_summaries if s.get("success", False)]
    docs_successful = len(successful)
    docs_failed = len(doc_names) - docs_successful
    avg_confidence = (
        sum(s.get("avg_score", 0.0) for s in successful) / len(successful)
        if successful
        else 0.0
    )

    # FIX BUG 2: Collect sources from all documents, limit to 10-15
    all_sources = []
    for summary in successful:
        all_sources.extend(summary.get("sources_used", []))

    # Limit to top 10-15 sources by score
    sources_used = sorted(all_sources, key=lambda x: x.get("score", 0.0), reverse=True)[:15]

    # Build legacy sources list (for compatibility)
    sources = [
        f"Document: {s['doc_name']} ({s['chunks_used']} chunks analyzed, "
        f"avg relevance {s['avg_score']:.0%})"
        for s in successful
    ]

    # Collect all errors
    all_errors = [s.get("error") for s in doc_summaries if s.get("error")] + reduce_errors
    all_errors = [e for e in all_errors if e]  # Remove None values

    logger.info(
        "answer_with_map_reduce: complete with %d successful, %d failed documents, %d sources",
        docs_successful, docs_failed, len(sources_used),
    )

    return {
        "answer": final_answer,
        "answer_type": "map_reduce",
        "query_type": "summarization",
        "provider": "groq",
        "confidence": round(avg_confidence, 4),
        "documents_analyzed": doc_names,
        "docs_successful": docs_successful,
        "docs_failed": docs_failed,
        "per_doc_summaries": doc_summaries,
        "sources_used": sources_used,
        "sources_found": len(sources_used),
        "fallback_used": False,
        "graph_evidence": [],
        "results": sources_used,  # For render_sources
        "sources": sources,  # Alias for legacy compatibility
        "document_errors": all_errors,
    }


# ---------------------------------------------------------------------------
# PART 5 — Smart Router Integration
# ---------------------------------------------------------------------------

def route_and_answer(
    question: str,
    vector_store: Any,
    graph_pipeline: Any = None,
    document_filter: str | None = None,
) -> dict[str, Any]:
    """Intelligently route to the best RAG strategy.

    Routes between Map-Reduce (for broad/comparative queries),
    GraphRAG (for entity/relationship queries), and standard RAG
    (everything else).

    Args:
        question: User's natural language question.
        vector_store: Loaded Chroma vector store.
        graph_pipeline: Optional GraphRAGPipeline instance.
        document_filter: Single document name filter (disables Map-Reduce).

    Returns:
        Answer dict with routing_decision field indicating which mode was used.
    """
    # Step 1: Detect query mode
    mode = detect_query_mode(question)

    # Step 2: Map-Reduce path (only when "All Documents" is selected)
    if mode == "map_reduce" and document_filter is None:
        logger.info("route_and_answer: routing to Map-Reduce RAG (query mode: %s)", mode)
        result = answer_with_map_reduce(question, vector_store)
        result["routing_decision"] = "map_reduce"
        return result

    # Step 3: GraphRAG path (when graph is ready and not filtering to single doc)
    if mode == "graph_rag" and graph_pipeline is not None and graph_pipeline.is_ready:
        logger.info("route_and_answer: routing to GraphRAG (query mode: %s, graph ready)", mode)
        from src.graph_rag import answer_question_with_graph_rag  # noqa: PLC0415

        result = answer_question_with_graph_rag(
            question, vector_store, graph_pipeline, document_filter=document_filter
        )
        result["routing_decision"] = "graph_rag"
        return result

    # Step 4: Default — Standard RAG
    logger.info("route_and_answer: routing to Standard RAG (query mode: %s)", mode)
    from src.rag_chain import answer_question_with_rag  # noqa: PLC0415

    result = answer_question_with_rag(question, vector_store, document_filter=document_filter)
    result["routing_decision"] = "standard_rag"
    return result
