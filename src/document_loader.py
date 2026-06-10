"""Document loading utilities: save uploaded PDFs and extract text via PyMuPDF."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.config import RAW_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def save_uploaded_file(uploaded_file: Any) -> str:
    """Save a Streamlit UploadedFile to data/raw/ and return the file path.

    Args:
        uploaded_file: Streamlit UploadedFile object with .name and .read() / file-like interface.

    Returns:
        Absolute string path to the saved file.
    """
    dest_dir = Path(RAW_DATA_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / uploaded_file.name

    try:
        with open(dest_path, "wb") as f:
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, f)
        logger.info("Saved uploaded file to %s", dest_path)
        return str(dest_path)
    except Exception as exc:
        logger.error("Failed to save uploaded file '%s': %s", uploaded_file.name, exc)
        raise


def extract_text_from_pdf(file_path: str) -> list[dict[str, Any]]:
    """Extract text page-by-page from a PDF using PyMuPDF (fitz).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts with keys: document_name, page_number, text.
    """
    pages: list[dict[str, Any]] = []
    doc_name = Path(file_path).name

    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "text": text,
                    }
                )
        doc.close()
        logger.info("Extracted %d pages from '%s'", len(pages), doc_name)
    except Exception as exc:
        logger.error("Failed to extract text from '%s': %s", file_path, exc)
        raise

    return pages


def load_pdf_with_metadata(file_path: str) -> list[dict[str, Any]]:
    """Extract text from a PDF and return pages with full metadata.

    This is the primary entry point — it calls extract_text_from_pdf internally.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of page dicts with keys: document_name, page_number, text.
    """
    logger.info("Loading PDF with metadata: %s", file_path)
    return extract_text_from_pdf(file_path)


def process_multiple_pdfs(
    uploaded_files: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process a list of Streamlit UploadedFile objects and combine all extracted pages.

    Errors on individual files are caught and logged — processing continues with
    the remaining files so that one bad PDF does not abort the entire batch.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of:
            all_pages   — combined list of page dicts (document_name, page_number, text)
                          from every successfully processed file.
            summary     — dict with keys:
                            total_files      (int)       number of files processed OK
                            total_pages      (int)       total pages across all files
                            files_processed  (list[str]) names of successfully loaded files
    """
    if not uploaded_files:
        logger.info("process_multiple_pdfs: no files provided.")
        return [], {"total_files": 0, "total_pages": 0, "files_processed": []}

    all_pages: list[dict[str, Any]] = []
    files_processed: list[str] = []

    for uploaded_file in uploaded_files:
        fname = getattr(uploaded_file, "name", str(uploaded_file))
        logger.info("process_multiple_pdfs: processing '%s'", fname)
        try:
            file_path = save_uploaded_file(uploaded_file)
            pages = load_pdf_with_metadata(file_path)
            all_pages.extend(pages)
            files_processed.append(fname)
            logger.info(
                "process_multiple_pdfs: '%s' — %d pages extracted.", fname, len(pages)
            )
        except Exception as exc:
            logger.error(
                "process_multiple_pdfs: skipping '%s' due to error: %s", fname, exc
            )

    summary: dict[str, Any] = {
        "total_files": len(files_processed),
        "total_pages": len(all_pages),
        "files_processed": files_processed,
    }
    logger.info(
        "process_multiple_pdfs: done — %d/%d files, %d pages total.",
        len(files_processed),
        len(uploaded_files),
        len(all_pages),
    )
    return all_pages, summary
