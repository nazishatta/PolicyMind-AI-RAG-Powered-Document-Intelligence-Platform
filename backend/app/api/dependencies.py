"""FastAPI dependency providers.

All heavy objects (pipeline, vector store, graph service, LLM) are
constructed once at first request and reused for the lifetime of the
process.  Tests override these via app.dependency_overrides.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.embeddings import get_embedding_model
from app.services.graph_service import build_graph_service
from app.services.llm_service import build_llm_provider
from app.services.rag_pipeline import GraphRAGPipeline
from app.services.vector_store import build_vector_store


@lru_cache(maxsize=1)
def get_pipeline() -> GraphRAGPipeline:
    s = get_settings()
    return GraphRAGPipeline(
        settings=s,
        vector_store=build_vector_store(
            provider=s.vector_store_provider,
            chroma_persist_dir=s.chroma_persist_dir,
            chroma_collection=s.chroma_collection,
        ),
        graph_service=build_graph_service(
            provider=s.graph_provider,
            neo4j_uri=s.neo4j_uri,
            neo4j_user=s.neo4j_user,
            neo4j_password=s.neo4j_password,
            neo4j_database=s.neo4j_database,
        ),
        llm=build_llm_provider(
            provider=s.llm_provider,
            model=s.resolved_llm_model,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            anthropic_api_key=s.anthropic_api_key,
            openai_api_key=s.openai_api_key,
        ),
        embedding_model=get_embedding_model(
            model_name=s.embedding_model,
            device=s.embedding_device,
        ),
    )
