"""Basic smoke tests confirming core src/ modules import and behave sanely."""
from __future__ import annotations


def test_logger_importable() -> None:
    from src.logger import get_logger

    logger = get_logger(__name__)
    assert logger is not None


def test_text_splitter_importable() -> None:
    from src.text_splitter import split_documents_into_chunks

    assert callable(split_documents_into_chunks)


def test_citation_utils_importable() -> None:
    from src.citation_utils import format_sources

    assert callable(format_sources)
