"""Embedding model loader for PolicyMind AI — uses HuggingFace sentence-transformers."""

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME
from src.logger import get_logger

logger = get_logger(__name__)

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the sentence-transformer embedding model (singleton).

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embedding_model  # noqa: PLW0603

    if _embedding_model is None:
        try:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", EMBEDDING_MODEL_NAME, exc)
            raise

    return _embedding_model
