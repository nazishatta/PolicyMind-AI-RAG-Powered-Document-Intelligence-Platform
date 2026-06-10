"""Shared pytest fixtures.

All fixtures use in-memory providers and a mock LLM so the test suite
runs fully offline with no API keys and no external services.
"""

from __future__ import annotations

import warnings

# Suppress an import-time UserWarning emitted by fastapi/testclient.py when it
# imports from starlette.testclient with the httpx backend.
# StarletteDeprecationWarning inherits from UserWarning (not DeprecationWarning),
# so it must be filtered as UserWarning.  The filter is registered here (before
# the fastapi.testclient import below) because the warning fires at import time,
# before pytest applies its own filterwarnings from pyproject.toml.
warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated",
    category=UserWarning,
)

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.document_loader import PolicyDocument
from app.core.embeddings import EmbeddingModel
from app.core.text_chunker import TextChunk
from app.main import create_app
from app.services.graph_service import InMemoryGraphService
from app.services.llm_service import MockProvider
from app.services.rag_pipeline import GraphRAGPipeline
from app.services.vector_store import InMemoryVectorStore


SAMPLE_TEXT = """\
The National Climate Strategy 2030 commits Member States to reduce greenhouse
gas emissions by 55 percent relative to 1990 levels. The European Commission
will oversee compliance and allocate EUR 15 billion from the Just Transition Fund.
Article 12 establishes binding targets for the energy sector, including a minimum
30 percent share of renewable energy in final energy consumption by 2025.
Penalties for non-compliance are defined under Regulation (EU) 2021/1119.
"""


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings to use all in-memory/mock providers."""
    import os
    os.environ.update({
        "LLM_PROVIDER": "mock",
        "VECTOR_STORE_PROVIDER": "memory",
        "GRAPH_PROVIDER": "memory",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    })
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return get_settings()


@pytest.fixture()
def sample_document() -> PolicyDocument:
    return PolicyDocument(
        doc_id="test_doc_abc123",
        title="National Climate Strategy 2030",
        pages=[(1, SAMPLE_TEXT)],
        source_label="EU Commission",
        metadata={"source": "test"},
    )


@pytest.fixture()
def mock_embedding_model() -> EmbeddingModel:
    """Stub embedding model that returns random unit vectors (no download)."""

    class _StubEmbedding(EmbeddingModel):
        def embed(self, texts: list[str]) -> np.ndarray:
            rng = np.random.default_rng(seed=42)
            vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

    return _StubEmbedding()


@pytest.fixture()
def in_memory_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture()
def in_memory_graph() -> InMemoryGraphService:
    return InMemoryGraphService()


@pytest.fixture()
def pipeline(
    test_settings,
    mock_embedding_model,
    in_memory_vector_store,
    in_memory_graph,
) -> GraphRAGPipeline:
    return GraphRAGPipeline(
        settings=test_settings,
        vector_store=in_memory_vector_store,
        graph_service=in_memory_graph,
        llm=MockProvider(),
        embedding_model=mock_embedding_model,
    )


@pytest.fixture()
def api_client(test_settings, pipeline) -> TestClient:
    from app.api.dependencies import get_pipeline

    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)
