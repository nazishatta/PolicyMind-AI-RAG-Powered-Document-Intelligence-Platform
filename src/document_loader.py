"""Document loading utilities: save uploaded PDFs and extract text via PyMuPDF."""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests

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

    This is the primary entry point â€” it calls extract_text_from_pdf internally.

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

    Errors on individual files are caught and logged â€” processing continues with
    the remaining files so that one bad PDF does not abort the entire batch.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of:
            all_pages   â€” combined list of page dicts (document_name, page_number, text)
                          from every successfully processed file.
            summary     â€” dict with keys:
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
                "process_multiple_pdfs: '%s' â€” %d pages extracted.", fname, len(pages)
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
        "process_multiple_pdfs: done â€” %d/%d files, %d pages total.",
        len(files_processed),
        len(uploaded_files),
        len(all_pages),
    )
    return all_pages, summary

class _URLUploadedFile:
    """Lightweight wrapper that mimics Streamlit's UploadedFile interface
    (.name + file-like .read()/.seek()) so a downloaded PDF can flow through
    the exact same save_uploaded_file() / process_multiple_pdfs() pipeline
    used for manually uploaded files.
    """

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buffer = BytesIO(data)

    def seek(self, pos: int) -> int:
        return self._buffer.seek(pos)

    def read(self, *args: Any) -> bytes:
        return self._buffer.read(*args)


def download_pdf_from_url(url: str, timeout: int = 30) -> Any:
    """Download a PDF from a URL and wrap it to match the UploadedFile interface.

    Validates the response is actually a PDF (Content-Type header and/or the
    %PDF magic bytes) before accepting it, and enforces a maximum size to avoid
    accidentally downloading huge files.

    Args:
        url: Direct URL to a PDF file.
        timeout: Request timeout in seconds.

    Returns:
        A _URLUploadedFile object that behaves like a Streamlit UploadedFile,
        ready to pass into save_uploaded_file() / process_multiple_pdfs().

    Raises:
        ValueError: if the URL does not point to a valid PDF, or the file is
            too large.
        requests.RequestException: on network/HTTP errors.
    """
    max_size_bytes = 50 * 1024 * 1024  # 50 MB cap

    logger.info("download_pdf_from_url: fetching %s", url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    content_length = int(response.headers.get("Content-Length", 0))

    if content_length and content_length > max_size_bytes:
        raise ValueError(
            f"File too large ({content_length / 1024 / 1024:.1f} MB). "
            f"Maximum allowed is {max_size_bytes / 1024 / 1024:.0f} MB."
        )

    data = response.content

    if len(data) > max_size_bytes:
        raise ValueError(
            f"Downloaded file too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed is {max_size_bytes / 1024 / 1024:.0f} MB."
        )

    is_pdf_content_type = "application/pdf" in content_type
    is_pdf_magic_bytes = data[:5] == b"%PDF-"

    if not (is_pdf_content_type or is_pdf_magic_bytes):
        raise ValueError(
            f"URL does not point to a valid PDF file (Content-Type: '{content_type}')."
        )

    # Derive a safe filename from the URL, falling back to a generic name.
    url_path = urlparse(url).path
    candidate_name = Path(url_path).name
    if not candidate_name or not candidate_name.lower().endswith(".pdf"):
        candidate_name = "downloaded_document.pdf"

    logger.info(
        "download_pdf_from_url: downloaded %d bytes as '%s'", len(data), candidate_name
    )
    return _URLUploadedFile(candidate_name, data)
