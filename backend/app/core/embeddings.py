"""Local embedding model wrapper (Sentence Transformers).

Embeddings run fully offline after the initial model download.
The model is lazy-loaded and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CACHE: dict[str, "EmbeddingModel"] = {}


class EmbeddingModel:
    """Thin, cache-friendly wrapper around a SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[object] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded: %s on %s", self.model_name, self.device)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns float32 array of shape (N, D)."""
        self._load()
        sanitised = [t if t.strip() else " " for t in texts]
        vecs = self._model.encode(  # type: ignore[union-attr]
            sanitised,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (D,)."""
        return self.embed([text])[0]


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> EmbeddingModel:
    """Return a process-level cached :class:`EmbeddingModel`."""
    key = f"{model_name}:{device}"
    if key not in _CACHE:
        _CACHE[key] = EmbeddingModel(model_name=model_name, device=device)
    return _CACHE[key]
