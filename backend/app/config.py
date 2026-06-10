"""Central configuration for PolicyMind-AI.

All values are read from environment variables (or .env).
Defaults are chosen so that the app runs fully offline in
"mock" mode with no API keys and no external services.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "PolicyMind-AI"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── LLM Provider ─────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "mock"] = "mock"
    anthropic_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)
    llm_model: str = ""          # empty → auto-selected per provider
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # ── Embeddings ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ── Vector Store ─────────────────────────────────────────
    vector_store_provider: Literal["chroma", "memory"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "policy_chunks"

    # ── Graph Layer ──────────────────────────────────────────
    graph_provider: Literal["neo4j", "memory"] = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="", repr=False)
    neo4j_database: str = "neo4j"

    # ── Chunking & Retrieval ──────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_chunks: int = 5
    top_k_graph: int = 3
    # Score fusion weight: fused = alpha*vector + (1-alpha)*graph_boost
    graph_vector_alpha: float = Field(default=0.7, ge=0.0, le=1.0)

    # ── Document Ingestion ────────────────────────────────────
    # Maximum size (MB) for a single URL-fetched document.
    # Raise this value for larger documents; lower it to protect memory.
    max_document_size_mb: int = 50

    # ── Derived properties ────────────────────────────────────
    @property
    def resolved_llm_model(self) -> str:
        _defaults = {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "mock": "mock-model",
        }
        return self.llm_model or _defaults[self.llm_provider]

    @property
    def graph_enabled(self) -> bool:
        return self.graph_provider == "neo4j" and bool(self.neo4j_password)

    @property
    def max_document_size_bytes(self) -> int:
        return self.max_document_size_mb * 1024 * 1024

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "Settings":
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
