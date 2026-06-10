"""RAG chain — query classification, smart prompting, confidence routing,
OpenAI + Groq LLM integration."""

from __future__ import annotations

import os
from typing import Any

from langchain_community.vectorstores import Chroma

from src.citation_utils import format_sources
from src.config import TOP_K_RESULTS
from src.logger import get_logger
from src.retriever import smart_search

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Groq import — graceful degradation if package is not installed
# ---------------------------------------------------------------------------
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.debug("groq package not installed — Groq provider unavailable.")

# ---------------------------------------------------------------------------
# Confidence thresholds (calibrated for cosine similarity scores post-fix)
# ---------------------------------------------------------------------------
_SCORE_DOCUMENT: float = 0.65   # >= → Document Answer (strong evidence from docs)
_SCORE_PARTIAL: float = 0.45    # >= → Partial Answer  (docs + general knowledge)
                                  #  < → General Answer  (primarily general knowledge)

# ---------------------------------------------------------------------------
# System prompt (shared by every LLM call)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are PolicyMind AI, a document intelligence assistant for policy research. "
    "Answer ONLY using the provided document context. Be clear and professional. "
    "If the context does not contain enough information to answer fully, say so explicitly, "
    "then provide what general knowledge you have about the topic, clearly labeled as "
    "General Knowledge (not from uploaded documents)."
)


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

_SUMMARIZATION_TERMS = frozenset(
    {"summarize", "summary", "summarise", "overview", "key points",
     "main points", "extract", "outline", "brief"}
)
_FACTUAL_TERMS = frozenset(
    {"what", "who", "when", "where", "how many", "how much",
     "which", "list", "name", "identify"}
)
_ANALYTICAL_TERMS = frozenset(
    {"why", "analyze", "analyse", "compare", "explain", "discuss",
     "evaluate", "assess", "implications", "impact", "effect"}
)


def detect_query_type(question: str) -> str:
    """Classify a question into one of four query types for prompt routing.

    Returns:
        "summarization" — requests a summary, overview, or key-points extraction
        "factual"       — asks what/who/when/where/how-many
        "analytical"    — asks why/analyze/compare/explain/discuss
        "general"       — everything else
    """
    q = question.lower()

    if any(term in q for term in _SUMMARIZATION_TERMS):
        return "summarization"
    if any(term in q for term in _ANALYTICAL_TERMS):
        return "analytical"
    # Check factual terms as word-starts to avoid false positives (e.g. "whatever")
    for term in _FACTUAL_TERMS:
        if q.startswith(term) or f" {term}" in q:
            return "factual"
    return "general"


# ---------------------------------------------------------------------------
# Smart prompt builder
# ---------------------------------------------------------------------------

def build_smart_prompt(question: str, context: str, query_type: str) -> str:
    """Build the user message for any LLM call, structured by query type.

    Args:
        question: The user's original question.
        context: Labelled context string from retrieved chunks.
        query_type: One of summarization / factual / analytical / general.

    Returns:
        User message string ready to pass to any chat-completion API.
    """
    if query_type == "summarization":
        return (
            "You are PolicyMind AI. Based on the following document excerpts, "
            "provide a comprehensive summary. Structure your response as:\n\n"
            "OVERVIEW: (2-3 sentences)\n\n"
            "KEY POINTS:\n"
            "1. \n2. \n3. \n4. \n5. \n\n"
            "CONCLUSION: (1-2 sentences)\n\n"
            f"Context:\n{context}"
        )

    if query_type == "factual":
        return (
            "You are PolicyMind AI. Answer this specific question using only "
            "the document context provided. Be precise and cite page numbers.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

    if query_type == "analytical":
        return (
            "You are PolicyMind AI. Provide a thorough analytical response "
            "based on the document evidence. Structure your response with the headings: "
            "ANALYSIS, EVIDENCE, IMPLICATIONS.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

    # general
    return (
        f"Context from uploaded documents:\n{context}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context_from_results(results: list[dict[str, Any]]) -> str:
    """Join retrieved chunks into a numbered, labelled context string."""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        header = f"[Source {i}: {r['document_name']}, page {r['page_number']}]"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# No-key fallbacks
# ---------------------------------------------------------------------------

def _format_evidence_fallback(results: list[dict[str, Any]]) -> str:
    """Clean evidence display (one block per chunk) when no LLM API key is configured."""
    if not results:
        return "No relevant passages were found in the uploaded documents."

    lines: list[str] = [
        "No LLM API key configured (OpenAI or Groq). "
        "Here is the retrieved evidence from your documents:",
        "",
    ]
    for i, r in enumerate(results, start=1):
        score_pct = r.get("score", 0.0) * 100
        excerpt = r.get("text", "")[:400].replace("\n", " ").strip()
        lines.append(
            f"[{i}]  Document: {r['document_name']}"
            f"  |  Page: {r['page_number']}"
            f"  |  Relevance: {score_pct:.0f}%"
        )
        lines.append(f'     "{excerpt}..."')
        lines.append("")
    return "\n".join(lines)


def _format_summarization_fallback(results: list[dict[str, Any]]) -> str:
    """Structured page-grouped overview for summarization queries without any LLM key."""
    if not results:
        return "No relevant passages were found in the uploaded documents."

    page_best: dict[int, dict[str, Any]] = {}
    for r in results:
        page = int(r.get("page_number", 0))
        if page not in page_best or r["score"] > page_best[page]["score"]:
            page_best[page] = r

    lines: list[str] = [
        "DOCUMENT OVERVIEW (Evidence-Based)",
        "",
        "The following excerpts are extracted directly from your uploaded document:",
        "",
    ]

    for page_num in sorted(page_best.keys()):
        r = page_best[page_num]
        excerpt = r.get("text", "")[:500].strip()
        section_header = f"Page {page_num}"
        lines.append(section_header)
        lines.append("-" * len(section_header))
        lines.append(excerpt)
        lines.append("")

    lines.append(
        "Note: This overview is extracted directly from document text. "
        "Add an OPENAI_API_KEY or GROQ_API_KEY to .env for an AI-generated structured summary."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared user-message builder (with confidence-level hints)
# ---------------------------------------------------------------------------

def _build_llm_user_message(
    question: str, context: str, query_type: str, answer_type: str
) -> str:
    """Return the user message for any LLM call, with confidence-based hints appended."""
    base = build_smart_prompt(question, context, query_type)

    if answer_type == "Partial Answer":
        return (
            base
            + "\n\n(Note: Document relevance is moderate — where the context is "
            "insufficient, supplement with clearly-labelled general knowledge.)"
        )
    if answer_type == "General Answer":
        return (
            base
            + "\n\n(Note: Document relevance is low. Answer primarily from general "
            "knowledge and label that section clearly as 'General Knowledge'.)"
        )
    return base


# ---------------------------------------------------------------------------
# Provider-specific call functions
# ---------------------------------------------------------------------------

def _get_effective_system_prompt(results: list[dict[str, Any]]) -> str:
    """Return the system prompt, appending a cross-document attribution instruction
    when retrieved chunks come from more than one distinct document."""
    unique_docs = {r.get("document_name", "") for r in results if r.get("document_name")}
    if len(unique_docs) > 1:
        return (
            _SYSTEM_PROMPT
            + " When answering, mention which document each piece of information comes from."
        )
    return _SYSTEM_PROMPT


def _call_openai(
    question: str,
    context: str,
    query_type: str,
    answer_type: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> str:
    """Call OpenAI GPT-4o-mini.

    Args:
        question: User's question.
        context: Labelled context string.
        query_type: Detected query type.
        answer_type: Confidence-based routing label.
        system_prompt: System prompt to use (caller may pass multi-doc variant).

    Returns:
        Answer string from the model.
    """
    from openai import OpenAI  # deferred — keeps startup fast when key absent

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_message = _build_llm_user_message(question, context, query_type, answer_type)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_groq(
    question: str,
    context: str,
    query_type: str,
    answer_type: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> str:
    """Call Groq LLaMA 3.1 8B Instant.

    Args:
        question: User's question.
        context: Labelled context string.
        query_type: Detected query type.
        answer_type: Confidence-based routing label.
        system_prompt: System prompt to use (caller may pass multi-doc variant).

    Returns:
        Answer string from the model.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    user_message = _build_llm_user_message(question, context, query_type, answer_type)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Answer generation (provider priority + confidence routing)
# ---------------------------------------------------------------------------

def generate_rag_answer(
    question: str,
    context: str,
    sources: list[str],
    results: list[dict[str, Any]],
    avg_score: float,
    query_type: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Route to the correct LLM provider and confidence tier.

    Provider priority:
        1. OpenAI GPT-4o-mini  — if OPENAI_API_KEY is set and non-empty
        2. Groq LLaMA 3.1 8B   — if GROQ_API_KEY is set, non-empty, and groq package installed
        3. Evidence fallback    — clean formatted output, no LLM

    Args:
        question: User's question.
        context: Combined labelled context string.
        sources: Formatted citation strings.
        results: Raw retrieved chunk dicts.
        avg_score: Average cosine similarity across retrieved chunks.
        query_type: Detected query type (summarization / factual / analytical / general).
        system_prompt: System prompt to pass to the LLM (may include multi-doc instruction).

    Returns:
        Dict with keys: answer, answer_type, fallback_used, provider.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # Determine active provider
    if openai_key:
        provider = "openai"
    elif groq_key and GROQ_AVAILABLE:
        provider = "groq"
    else:
        provider = "none"

    # No LLM available — clean evidence display
    if provider == "none":
        logger.info("No LLM API key configured — returning evidence/summarization fallback.")
        answer = (
            _format_summarization_fallback(results)
            if query_type == "summarization"
            else _format_evidence_fallback(results)
        )
        return {
            "answer": answer,
            "answer_type": "Evidence Only",
            "fallback_used": True,
            "provider": "none",
        }

    # Confidence-based answer-type routing (same logic for both providers)
    if avg_score >= _SCORE_DOCUMENT:
        answer_type, fallback_used = "Document Answer", False
    elif avg_score >= _SCORE_PARTIAL:
        answer_type, fallback_used = "Partial Answer", True
    else:
        answer_type, fallback_used = "General Answer", True

    logger.info(
        "Answer routing: provider=%s, query_type=%s, answer_type=%s, avg_score=%.4f",
        provider, query_type, answer_type, avg_score,
    )

    try:
        if provider == "openai":
            answer = _call_openai(
                question, context, query_type, answer_type, system_prompt=system_prompt
            )
            logger.info("OpenAI answer generated (answer_type=%s).", answer_type)
        else:
            answer = _call_groq(
                question, context, query_type, answer_type, system_prompt=system_prompt
            )
            logger.info("Groq answer generated (answer_type=%s).", answer_type)

    except Exception as exc:
        logger.error("%s call failed — falling back to evidence display: %s", provider, exc)
        answer = (
            _format_summarization_fallback(results)
            if query_type == "summarization"
            else _format_evidence_fallback(results)
        )
        answer_type, fallback_used, provider = "Evidence Only", True, "none"

    return {
        "answer": answer,
        "answer_type": answer_type,
        "fallback_used": fallback_used,
        "provider": provider,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question_with_rag(
    question: str,
    vector_store: Chroma,
    top_k: int = TOP_K_RESULTS,
    document_filter: str | None = None,
) -> dict[str, Any]:
    """End-to-end RAG pipeline: smart search → query classification → LLM answer.

    Provider priority: OpenAI → Groq → Evidence fallback.

    Args:
        question: Natural language question or summarization request.
        vector_store: Loaded Chroma vector store (must use cosine metric).
        top_k: Maximum chunks for regular questions (summarization always uses up to 15).
        document_filter: If set (and not "All Documents"), restrict retrieval to a
            single document by exact document_name match.

    Returns:
        Dict with keys:
            answer        (str)        — final answer text
            answer_type   (str)        — Document Answer / Partial Answer / General Answer / Evidence Only
            query_type    (str)        — summarization / factual / analytical / general
            confidence    (float)      — average cosine similarity score (0–1)
            provider      (str)        — "openai" / "groq" / "none"
            sources_used  (list[str])  — formatted citation strings
            fallback_used (bool)       — True when answer extends beyond document evidence
            sources       (list[str])  — alias for sources_used (UI compatibility)
            results       (list[dict]) — raw retrieved chunk dicts
    """
    try:
        search_result = smart_search(
            question, vector_store, top_k=top_k, document_filter=document_filter
        )
        results: list[dict[str, Any]] = search_result["results"]
        query_type: str = detect_query_type(question)

        sources = format_sources(results)
        avg_score: float = (
            sum(r["score"] for r in results) / len(results) if results else 0.0
        )

        context = build_context_from_results(results)
        system_prompt = _get_effective_system_prompt(results)
        answer_data = generate_rag_answer(
            question, context, sources, results, avg_score, query_type,
            system_prompt=system_prompt,
        )

        return {
            "answer": answer_data["answer"],
            "answer_type": answer_data["answer_type"],
            "query_type": query_type,
            "confidence": round(avg_score, 4),
            "provider": answer_data["provider"],
            "sources_used": sources,
            "fallback_used": answer_data["fallback_used"],
            "sources": sources,
            "results": results,
        }

    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc)
        return {
            "answer": f"An error occurred while processing your question: {exc}",
            "answer_type": "Error",
            "query_type": "general",
            "confidence": 0.0,
            "provider": "none",
            "sources_used": [],
            "fallback_used": True,
            "sources": [],
            "results": [],
        }
