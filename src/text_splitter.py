"""Text splitting utilities using LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.logger import get_logger

logger = get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_documents_into_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a list of page dicts into smaller text chunks.

    Args:
        pages: List of dicts with keys: document_name, page_number, text.

    Returns:
        List of chunk dicts with keys: chunk_id, document_name, page_number, text.
    """
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    try:
        for page in pages:
            doc_name = page.get("document_name", "unknown")
            page_num = page.get("page_number", 0)
            text = page.get("text", "")

            if not text.strip():
                continue

            split_texts = _splitter.split_text(text)
            for piece in split_texts:
                chunks.append(
                    {
                        "chunk_id": f"{doc_name}_p{page_num}_c{chunk_index}",
                        "document_name": doc_name,
                        "page_number": page_num,
                        "text": piece,
                    }
                )
                chunk_index += 1

        logger.info(
            "Split %d pages into %d chunks (chunk_size=%d, overlap=%d)",
            len(pages),
            len(chunks),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
    except Exception as exc:
        logger.error("Error splitting documents into chunks: %s", exc)
        raise

    return chunks
