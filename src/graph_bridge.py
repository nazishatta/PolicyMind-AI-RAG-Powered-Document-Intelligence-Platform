"""GraphRAG Bridge — connects backend/app/services/rag_pipeline.py to the Streamlit MVP.
Falls back to standard RAG automatically if GraphRAG services are unavailable.

Import isolation
----------------
The backend package lives at ``backend/`` and uses ``app.*`` imports internally.
The Streamlit app also registers its own ``app`` package (``app/components/`` etc.).
These subpackage namespaces don't overlap, so the bridge safely loads backend
modules by temporarily swapping the ``app`` sys.modules entry, then restoring it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend path constant
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")

# Module-level cache for imported backend symbols (None = not yet attempted;
# empty dict = attempted but failed)
_backend_refs: Optional[dict[str, Any]] = None
_load_attempted: bool = False


# ---------------------------------------------------------------------------
# Groq LLM provider (duck-typed — no subclassing needed; pipeline calls
# .provider_name, .model_id, .complete(), .acomplete())
# ---------------------------------------------------------------------------

class _GroqBridgeProvider:
    """Groq LLaMA provider for the GraphRAG pipeline (bridge-only)."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        try:
            from groq import Groq  # type: ignore
            self._client = Groq(api_key=api_key)
        except ImportError as exc:
            raise ImportError("pip install groq") from exc
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.complete, system_prompt, user_prompt)

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_id(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Backend import helper
# ---------------------------------------------------------------------------

def _load_backend() -> Optional[dict[str, Any]]:
    """Import backend modules, preserving the Streamlit app's ``app`` package.

    The backend uses ``app.*`` imports internally.  We temporarily clear all
    ``app.*`` entries from ``sys.modules``, add ``backend/`` to ``sys.path``,
    import the needed symbols (registering them under ``app.*``), then restore
    the Streamlit app's ``app`` entry.  Subpackage entries (``app.services.*``
    etc.) remain in ``sys.modules`` and are looked up by module name on every
    lazy import — so lazy imports inside backend functions still resolve
    correctly.

    Returns None (and logs a warning) on any ImportError or other failure.
    """
    global _backend_refs, _load_attempted

    if _load_attempted:
        return _backend_refs

    _load_attempted = True

    # 1. Save all current app.* sys.modules entries (Streamlit's)
    saved: dict[str, Any] = {
        k: v
        for k, v in list(sys.modules.items())
        if k == "app" or k.startswith("app.")
    }
    for key in saved:
        sys.modules.pop(key, None)

    # 2. Put backend/ at the front of sys.path
    _inserted = False
    if _BACKEND_DIR in sys.path:
        sys.path.remove(_BACKEND_DIR)
    sys.path.insert(0, _BACKEND_DIR)
    _inserted = True

    try:
        from app.config import Settings  # type: ignore  # noqa: PLC0415
        from app.core.document_loader import PolicyDocument  # type: ignore  # noqa: PLC0415
        from app.core.embeddings import get_embedding_model as _get_emb  # type: ignore  # noqa: PLC0415
        from app.services.graph_service import build_graph_service  # type: ignore  # noqa: PLC0415
        from app.services.llm_service import build_llm_provider  # type: ignore  # noqa: PLC0415
        from app.services.rag_pipeline import GraphRAGPipeline  # type: ignore  # noqa: PLC0415
        from app.services.vector_store import build_vector_store  # type: ignore  # noqa: PLC0415

        _backend_refs = {
            "Settings": Settings,
            "PolicyDocument": PolicyDocument,
            "get_embedding_model": _get_emb,
            "build_graph_service": build_graph_service,
            "build_llm_provider": build_llm_provider,
            "GraphRAGPipeline": GraphRAGPipeline,
            "build_vector_store": build_vector_store,
        }
        logger.info("GraphRAG backend modules loaded from %s", _BACKEND_DIR)
        return _backend_refs

    except ImportError as exc:
        logger.warning("GraphRAG backend unavailable (ImportError): %s", exc)
        _backend_refs = {}
        return None

    except Exception as exc:
        logger.warning("GraphRAG backend failed to load: %s", exc)
        _backend_refs = {}
        return None

    finally:
        # 3. Restore Streamlit app's app.* entries (base "app" is overwritten
        #    so Streamlit component imports still work; backend sub-entries
        #    remain in sys.modules and are looked up by full dotted name)
        for key, mod in saved.items():
            sys.modules[key] = mod

        # 4. Remove backend/ from sys.path
        if _inserted:
            try:
                sys.path.remove(_BACKEND_DIR)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

def _run_async(coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an async callable from a synchronous context.

    Runs the coroutine in a daemon thread with its own event loop so that
    Streamlit's main thread (which may already have a running loop) is never
    blocked or disrupted.

    Falls back to ``nest_asyncio`` if running inside a notebook/IPython
    environment where ``asyncio.run()`` is not available.
    """
    def _thread_run() -> Any:
        return asyncio.run(coro_fn(*args, **kwargs))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_thread_run).result(timeout=180)
    except RuntimeError:
        # asyncio.run() not available — try nest_asyncio
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro_fn(*args, **kwargs))
        except ImportError:
            raise RuntimeError(
                "Cannot run async GraphRAG query. Install nest-asyncio: pip install nest-asyncio"
            )


# ---------------------------------------------------------------------------
# doc_id helper (must match backend/_doc_id in document_loader.py)
# ---------------------------------------------------------------------------

def _doc_id_from_name(document_name: str) -> str:
    """Stable 16-char hex doc_id derived from the document filename.

    Must produce the same result as ``backend/app/core/document_loader._doc_id``
    so that doc_id values used during ingestion match those used for filtering.
    """
    return hashlib.sha256(document_name.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_graph_available() -> bool:
    """Return True if the GraphRAG backend was successfully imported."""
    refs = _load_backend()
    return bool(refs)


def init_graph_pipeline() -> Optional[Any]:
    """Initialise and cache the GraphRAG pipeline in Streamlit session state.

    LLM provider priority (mirrors the standard RAG chain):
        1. Groq  — if GROQ_API_KEY is set and ``groq`` package installed
        2. OpenAI — if OPENAI_API_KEY is set
        3. Mock   — deterministic offline stub

    The pipeline uses in-memory vector and graph stores (no Neo4j needed).
    Result is cached in ``st.session_state["graph_pipeline"]`` so it
    survives Streamlit re-runs within the same session.

    Returns:
        Initialized GraphRAGPipeline, or None if backend is unavailable.
    """
    try:
        import streamlit as st  # type: ignore  # noqa: PLC0415
    except ImportError:
        st = None  # type: ignore

    if st is not None and "graph_pipeline" in st.session_state:
        return st.session_state["graph_pipeline"]

    refs = _load_backend()
    if not refs:
        return None

    try:
        Settings = refs["Settings"]
        get_embedding_model = refs["get_embedding_model"]
        build_graph_service = refs["build_graph_service"]
        build_llm_provider = refs["build_llm_provider"]
        build_vector_store = refs["build_vector_store"]
        GraphRAGPipeline = refs["GraphRAGPipeline"]

        # Settings with llm_provider="mock" avoids API-key validation errors
        try:
            settings = Settings(llm_provider="mock")
        except Exception as exc:
            logger.warning("Settings() failed (%s) — using minimal defaults.", exc)

            class _FallbackSettings:
                chunk_size = 512
                chunk_overlap = 64
                top_k_chunks = 5
                top_k_graph = 3
                graph_vector_alpha = 0.7
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                embedding_device = "cpu"
                llm_max_tokens = 2048
                llm_temperature = 0.0

                def entity_count(self) -> int:
                    return 0

            settings = _FallbackSettings()  # type: ignore

        # Select LLM provider
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if groq_key:
            try:
                llm = _GroqBridgeProvider(groq_key)
                logger.info("GraphRAG: using Groq LLaMA 3 provider.")
            except Exception as exc:
                logger.warning("Groq provider failed (%s) — falling back to mock.", exc)
                llm = build_llm_provider("mock", model="", temperature=0.0, max_tokens=2048)
        elif openai_key:
            try:
                llm = build_llm_provider(
                    "openai",
                    model="gpt-4o-mini",
                    temperature=0.0,
                    max_tokens=2048,
                    openai_api_key=openai_key,
                )
                logger.info("GraphRAG: using OpenAI GPT-4o-mini provider.")
            except Exception as exc:
                logger.warning("OpenAI provider failed (%s) — falling back to mock.", exc)
                llm = build_llm_provider("mock", model="", temperature=0.0, max_tokens=2048)
        else:
            llm = build_llm_provider("mock", model="", temperature=0.0, max_tokens=2048)
            logger.info("GraphRAG: using Mock provider (no API key found).")

        vector_store = build_vector_store("memory")
        graph_service = build_graph_service("memory")
        embedding_model = get_embedding_model(
            getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            getattr(settings, "embedding_device", "cpu"),
        )

        pipeline = GraphRAGPipeline(
            settings=settings,  # type: ignore
            vector_store=vector_store,
            graph_service=graph_service,
            llm=llm,
            embedding_model=embedding_model,
        )

        if st is not None:
            st.session_state["graph_pipeline"] = pipeline

        logger.info("GraphRAG pipeline initialised successfully.")
        return pipeline

    except Exception as exc:
        logger.error("GraphRAG pipeline init failed: %s", exc)
        return None


def ingest_documents_to_graph(pages_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Ingest a list of page dicts into the GraphRAG pipeline.

    Args:
        pages_list: List of page dicts in the format produced by
            ``src.document_loader`` — each dict has keys:
            ``document_name`` (str), ``page_number`` (int), ``text`` (str).

    Returns:
        Summary dict with keys:
            documents_ingested  (int)
            total_chunks        (int)
            entities_extracted  (int)
            graph_nodes         (int)
            errors              (list[str])
    """
    try:
        import streamlit as st  # type: ignore  # noqa: PLC0415
        pipeline = st.session_state.get("graph_pipeline")
    except ImportError:
        pipeline = None

    if pipeline is None:
        pipeline = init_graph_pipeline()

    if pipeline is None:
        return {
            "documents_ingested": 0,
            "total_chunks": 0,
            "entities_extracted": 0,
            "graph_nodes": 0,
            "errors": ["GraphRAG pipeline is not available."],
        }

    refs = _load_backend()
    if not refs:
        return {
            "documents_ingested": 0,
            "total_chunks": 0,
            "entities_extracted": 0,
            "graph_nodes": 0,
            "errors": ["Backend modules unavailable."],
        }

    PolicyDocument = refs["PolicyDocument"]

    # Group pages by document_name
    docs_map: dict[str, list[dict[str, Any]]] = {}
    for page in pages_list:
        name = page.get("document_name", "unknown")
        docs_map.setdefault(name, []).append(page)

    total_chunks = 0
    total_entities = 0
    total_nodes = 0
    docs_ingested = 0
    errors: list[str] = []

    for doc_name, doc_pages in docs_map.items():
        try:
            doc_id = _doc_id_from_name(doc_name)
            pages_tuples = [
                (int(p.get("page_number", 1)), str(p.get("text", "")))
                for p in sorted(doc_pages, key=lambda x: x.get("page_number", 1))
            ]
            # Filter out empty pages
            pages_tuples = [(pn, txt) for pn, txt in pages_tuples if txt.strip()]

            if not pages_tuples:
                errors.append(f"{doc_name}: no text content after filtering.")
                continue

            doc = PolicyDocument(
                doc_id=doc_id,
                title=doc_name,
                pages=pages_tuples,
            )

            result = pipeline.ingest(doc)
            docs_ingested += 1
            total_chunks += result.chunk_count
            total_entities += result.entities_extracted
            total_nodes += result.graph_nodes_added
            logger.info(
                "Ingested '%s': %d chunks, %d entities, %d new nodes",
                doc_name, result.chunk_count, result.entities_extracted, result.graph_nodes_added,
            )

        except Exception as exc:
            errors.append(f"{doc_name}: {exc}")
            logger.error("ingest_documents_to_graph: failed for '%s': %s", doc_name, exc)

    return {
        "documents_ingested": docs_ingested,
        "total_chunks": total_chunks,
        "entities_extracted": total_entities,
        "graph_nodes": total_nodes,
        "errors": errors,
    }


def query_with_graph(
    question: str,
    document_filter: Optional[str] = None,
) -> dict[str, Any]:
    """Run a query through the GraphRAG pipeline.

    Args:
        question: The user's question.
        document_filter: If set (and not "All Documents"), restrict retrieval
            to a single document by matching its doc_id (derived from filename).

    Returns:
        Standardised result dict compatible with ``render_graph_answer()``:
            answer          (str)
            answer_type     (str)  — "cited" / "partial" / "refused" / "no_corpus"
            provider        (str)  — LLM provider name
            confidence      (float | None)
            citations       (list[dict])
            graph_evidence  (list[dict])
            retrieved_chunks (list[dict])
            evidence_quality (str)
            limitations     (list[str])
            latency_ms      (float)
            graph_used      (bool)
    """
    try:
        import streamlit as st  # type: ignore  # noqa: PLC0415
        pipeline = st.session_state.get("graph_pipeline")
    except ImportError:
        pipeline = None

    if pipeline is None:
        return _error_result("GraphRAG pipeline is not initialised.")

    # Convert document_filter (filename) to doc_id for the pipeline
    doc_id_filter: Optional[str] = None
    if document_filter and document_filter != "All Documents":
        doc_id_filter = _doc_id_from_name(document_filter)

    try:
        response = _run_async(
            pipeline.query,
            question,
            doc_id=doc_id_filter,
            include_graph=True,
        )
    except Exception as exc:
        logger.error("query_with_graph: pipeline.query() failed: %s", exc)
        return _error_result(f"GraphRAG query failed: {exc}")

    try:
        return _normalise_response(response)
    except Exception as exc:
        logger.error("query_with_graph: response normalisation failed: %s", exc)
        return _error_result(f"Response processing failed: {exc}")


def get_graph_stats() -> dict[str, Any]:
    """Return current knowledge-graph and document statistics.

    Returns:
        Dict with keys:
            entity_count      (int)
            relation_count    (int)
            graph_enabled     (bool)  — True if any entities were extracted
            documents_indexed (int)
            document_names    (list[str])
    """
    try:
        import streamlit as st  # type: ignore  # noqa: PLC0415
        pipeline = st.session_state.get("graph_pipeline")
    except ImportError:
        pipeline = None

    _empty = {
        "entity_count": 0,
        "relation_count": 0,
        "graph_enabled": False,
        "documents_indexed": 0,
        "document_names": [],
    }

    if pipeline is None:
        return _empty

    try:
        entity_count = pipeline._gs.entity_count()
        relation_count = pipeline._gs.relation_count()
        docs = pipeline.list_documents()
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "graph_enabled": entity_count > 0,
            "documents_indexed": len(docs),
            "document_names": [d.get("doc_title", "") for d in docs if d.get("doc_title")],
        }
    except Exception as exc:
        logger.error("get_graph_stats failed: %s", exc)
        return _empty


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_response(response: Any) -> dict[str, Any]:
    """Convert an AnswerResponse (Pydantic model) to a plain dict."""
    citations = [
        {
            "chunk_id": c.chunk_id,
            "doc_title": c.doc_title,
            "page_number": c.page_number,
            "excerpt": c.excerpt,
            "relevance_score": c.relevance_score,
        }
        for c in (response.citations or [])
    ]
    graph_evidence = [
        {
            "entity": e.entity,
            "relation": e.relation,
            "target": e.target,
            "confidence": e.confidence,
        }
        for e in (response.graph_evidence or [])
    ]
    retrieved_chunks = [
        {
            "chunk_id": rc.chunk_id,
            "doc_title": rc.doc_title,
            "page_number": rc.page_number,
            "text": rc.text,
            "relevance_score": rc.relevance_score,
        }
        for rc in (response.retrieved_chunks or [])
    ]
    return {
        "answer": response.answer,
        "answer_type": response.answer_type,
        "provider": response.provider,
        "confidence": response.confidence,
        "citations": citations,
        "graph_evidence": graph_evidence,
        "retrieved_chunks": retrieved_chunks,
        "evidence_quality": response.evidence_quality,
        "limitations": list(response.limitations or []),
        "latency_ms": response.latency_ms,
        "graph_used": bool(graph_evidence),
    }


def _error_result(message: str) -> dict[str, Any]:
    return {
        "answer": message,
        "answer_type": "no_corpus",
        "provider": "none",
        "confidence": None,
        "citations": [],
        "graph_evidence": [],
        "retrieved_chunks": [],
        "evidence_quality": "insufficient",
        "limitations": [message],
        "latency_ms": 0.0,
        "graph_used": False,
    }
