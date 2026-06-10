"""Document loading: URL-fetch or raw-text ingestion.

Supports publicly accessible PDF and plain-text documents via URL, and
raw text passed directly in the API payload.  All network access uses
httpx with configurable timeouts and a hard file-size cap.

No documents are stored in the repository. Users must supply their own
source URLs or text payloads. Obtain document URLs from public sources
such as government portals, UN repositories, World Bank Open Knowledge,
or EUR-Lex. Review the licensing terms for each document before use.

Parser preference:  pdfplumber (higher fidelity) → PyPDF2 (fallback).
Install both: pip install pdfplumber PyPDF2
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 10.0       # seconds to establish connection
_READ_TIMEOUT    = 60.0       # seconds to read response body (per chunk)
_TIMEOUT = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT,
                         write=None, pool=None)

_STREAM_CHUNK_BYTES = 65_536  # 64 KB read chunks

_PDF_MIME          = "application/pdf"
_OCTET_STREAM_MIMES = frozenset({"application/octet-stream", "binary/octet-stream"})
_TEXT_MIME_PREFIXES = ("text/",)

# Max characters accepted from a raw-text payload (≈ 150–200 pages)
MAX_TEXT_CHARS = 500_000


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Base class for all document ingestion errors."""


class DocumentFetchError(IngestionError):
    """URL is unreachable, returns a non-success HTTP status, or times out."""


class DocumentSizeError(IngestionError):
    """Document exceeds the configured size limit.

    Adjust MAX_DOCUMENT_SIZE_MB in .env to raise the cap.
    """


class UnsupportedContentTypeError(IngestionError):
    """Server returned a content-type that PolicyMind-AI cannot parse.

    Supported: application/pdf, text/* (plain, html, markdown, csv).
    """


class DocumentParseError(IngestionError):
    """The PDF or text content could not be parsed into readable pages."""


class DocumentEmptyError(IngestionError):
    """The document was fetched and parsed but contains no usable text.

    This occurs with scanned / image-only PDFs that have no embedded text layer.
    Consider running OCR on the source PDF before ingestion.
    """


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class PolicyDocument:
    """In-memory representation of a fetched and parsed policy document.

    Attributes:
        doc_id:       Hex-truncated SHA-256 of the raw content bytes.
                      Stable across re-ingestion of the same document.
        title:        Human-readable title (supplied or auto-detected).
        pages:        Ordered list of (page_number, cleaned_text) pairs.
        source_url:   Origin URL, if the document was fetched remotely.
        source_label: Short organisation / programme label supplied by caller.
        metadata:     Supplementary key-value pairs (content-type, file size, etc.)
    """

    doc_id: str
    title: str
    pages: list[tuple[int, str]]      # (page_number, cleaned_text)
    source_url: Optional[str] = None
    source_label: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(text for _, text in self.pages if text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(text) for _, text in self.pages)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _doc_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _clean(text: str) -> str:
    """Normalise whitespace and remove PDF extraction artefacts."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # rejoin hyphenated line-breaks
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)           # strip non-ASCII control chars
    return text.strip()


def _infer_title(url: Optional[str], first_page_text: str) -> str:
    """Heuristically derive a document title from URL stem or first-page text."""
    if url:
        stem = urlparse(url).path.rstrip("/").split("/")[-1]
        stem = re.sub(r"\.\w{2,5}$", "", stem)
        if stem and len(stem) > 3:
            return stem.replace("_", " ").replace("-", " ").title()
    for line in first_page_text.splitlines():
        line = line.strip()
        if 10 < len(line) <= 200 and not re.match(r"^\d", line):
            return line
    return "Untitled Policy Document"


def _is_pdf(content_type: str, url: str) -> bool:
    ct = content_type.lower().split(";")[0].strip()
    return ct == _PDF_MIME or url.lower().split("?")[0].endswith(".pdf")


def _validate_content_type(content_type: str, url: str) -> None:
    """Raise :class:`UnsupportedContentTypeError` if content type cannot be parsed."""
    ct = content_type.lower().split(";")[0].strip()
    if not ct or ct in _OCTET_STREAM_MIMES:
        return  # unknown / binary — let the parser decide
    if ct == _PDF_MIME:
        return
    if any(ct.startswith(p) for p in _TEXT_MIME_PREFIXES):
        return
    raise UnsupportedContentTypeError(
        f"Content type {content_type!r} is not supported. "
        "PolicyMind-AI accepts PDF (application/pdf) and plain-text (text/*) documents. "
        f"The server at {url!r} returned: {content_type!r}. "
        "If the document is a valid PDF, rename the URL or override content-type detection."
    )


def _parse_pdf_bytes(content: bytes) -> list[tuple[int, str]]:
    """Parse PDF bytes into a list of (page_number, cleaned_text) tuples.

    Attempts pdfplumber first; falls back to PyPDF2.
    Raises :class:`DocumentParseError` if neither library is available.
    Raises :class:`DocumentEmptyError` if the PDF yields no extractable text.
    """
    pages: list[tuple[int, str]] = []

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                pages.append((i, _clean(raw)))
        logger.debug("PDF parsed with pdfplumber: %d page(s)", len(pages))

    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(content))
            for i, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                pages.append((i, _clean(raw)))
            logger.debug("PDF parsed with PyPDF2: %d page(s)", len(pages))

        except ImportError as exc:
            raise DocumentParseError(
                "No PDF parser installed. Run: pip install pdfplumber PyPDF2"
            ) from exc

    except Exception as exc:
        raise DocumentParseError(f"PDF parsing failed: {exc}") from exc

    # Detect image-only / scanned PDFs
    non_empty = [text for _, text in pages if text.strip()]
    if not non_empty:
        raise DocumentEmptyError(
            f"The PDF has {len(pages)} page(s) but no extractable text. "
            "It may be a scanned or image-only PDF. "
            "Apply OCR (e.g. ocrmypdf) before ingestion, "
            "or use a text-layer version of the document."
        )

    return pages


# ---------------------------------------------------------------------------
# Internal fetch helper (isolated for unit testing)
# ---------------------------------------------------------------------------

async def _fetch_content(url: str, max_bytes: int) -> tuple[bytes, str]:
    """Fetch URL content with size guard and content-type validation.

    Returns:
        (raw_bytes, resolved_content_type)

    Raises:
        DocumentFetchError           – network or HTTP error
        DocumentSizeError            – response exceeds max_bytes
        UnsupportedContentTypeError  – server returned unsupported MIME type
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:

        # Step 1: HEAD request to check size and content-type cheaply.
        # Failure is non-fatal — many servers do not support HEAD.
        try:
            head = await client.head(url)
            if head.is_success:
                ct_head = head.headers.get("content-type", "").split(";")[0].strip()
                if ct_head:
                    _validate_content_type(ct_head, url)
                cl = head.headers.get("content-length", "")
                if cl.isdigit() and int(cl) > max_bytes:
                    raise DocumentSizeError(
                        f"Document is {int(cl) / 1_048_576:.1f} MB, which exceeds "
                        f"the {max_bytes / 1_048_576:.0f} MB limit. "
                        "Set MAX_DOCUMENT_SIZE_MB in your .env to raise the cap, "
                        "or use a smaller document."
                    )
        except IngestionError:
            raise
        except Exception as exc:
            logger.debug("HEAD request skipped or failed (%s): %s", url, exc)

        # Step 2: Streaming GET with live size guard.
        buffer = bytearray()
        content_type = ""
        try:
            async with client.stream("GET", url) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    hint = (
                        "Verify the URL is publicly accessible and does not "
                        "require authentication."
                        if status in (401, 403)
                        else "Check that the URL is correct and the resource exists."
                        if status == 404
                        else ""
                    )
                    raise DocumentFetchError(
                        f"HTTP {status} received from {url!r}. {hint}".strip()
                    ) from exc

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip()
                )
                _validate_content_type(content_type, url)

                async for chunk in response.aiter_bytes(_STREAM_CHUNK_BYTES):
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        raise DocumentSizeError(
                            f"Download exceeded the {max_bytes / 1_048_576:.0f} MB limit "
                            f"after {len(buffer) / 1_048_576:.1f} MB. "
                            "Set MAX_DOCUMENT_SIZE_MB in your .env to raise the cap."
                        )

        except IngestionError:
            raise
        except httpx.ConnectError as exc:
            raise DocumentFetchError(
                f"Could not connect to {url!r}. "
                "Verify the URL is reachable from your network."
            ) from exc
        except httpx.TimeoutException as exc:
            raise DocumentFetchError(
                f"Request to {url!r} timed out after {_READ_TIMEOUT}s. "
                "The server may be overloaded. Try again or use a different source URL."
            ) from exc
        except httpx.RequestError as exc:
            raise DocumentFetchError(
                f"Network error fetching {url!r}: {exc}"
            ) from exc

    return bytes(buffer), content_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def load_from_url(
    url: str,
    *,
    title: Optional[str] = None,
    source_label: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
) -> PolicyDocument:
    """Fetch a publicly accessible PDF or plain-text URL and parse it.

    Args:
        url:          Publicly accessible URL of the document.
        title:        Override the auto-detected title.
        source_label: Short label for the originating organisation.
        max_bytes:    Maximum download size in bytes (default 50 MB).
                      Override via MAX_DOCUMENT_SIZE_MB in .env.

    Returns:
        A :class:`PolicyDocument` ready for chunking and indexing.

    Raises:
        DocumentFetchError           – network error or non-2xx HTTP status
        DocumentSizeError            – document exceeds max_bytes
        UnsupportedContentTypeError  – server returned an unrecognised MIME type
        DocumentParseError           – PDF could not be parsed
        DocumentEmptyError           – parsed document contains no text
    """
    content, content_type = await _fetch_content(url, max_bytes)
    doc_id = _doc_id(content)

    if _is_pdf(content_type, url):
        pages = _parse_pdf_bytes(content)
    else:
        # Plain text / HTML — treat as single page
        try:
            decoded = content.decode("utf-8", errors="replace")
        except Exception as exc:
            raise DocumentParseError(f"Could not decode text content: {exc}") from exc
        cleaned = _clean(decoded)
        if not cleaned:
            raise DocumentEmptyError(
                "The document at {url!r} returned an empty response after cleaning."
            )
        pages = [(1, cleaned)]

    first_text = pages[0][1] if pages else ""
    resolved_title = title or _infer_title(url, first_text)

    logger.info(
        "Fetched and parsed '%s' from %s — %d page(s), doc_id=%s",
        resolved_title, url, len(pages), doc_id,
    )
    return PolicyDocument(
        doc_id=doc_id,
        title=resolved_title,
        pages=pages,
        source_url=url,
        source_label=source_label,
        metadata={
            "content_type": content_type,
            "url": url,
            "raw_bytes": len(content),
        },
    )


def load_from_text(
    text: str,
    *,
    title: Optional[str] = None,
    source_label: Optional[str] = None,
) -> PolicyDocument:
    """Wrap a raw text string as a single-page :class:`PolicyDocument`.

    Args:
        text:         Document text (max 500 000 characters).
        title:        Optional title override.
        source_label: Optional source organisation label.

    Returns:
        A :class:`PolicyDocument` ready for chunking and indexing.

    Raises:
        ValueError          – text is empty or exceeds MAX_TEXT_CHARS
        DocumentEmptyError  – text contains no usable content after cleaning
    """
    if not text or not text.strip():
        raise DocumentEmptyError(
            "The provided text payload is empty. "
            "Supply at least a few sentences of policy content."
        )
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentSizeError(
            f"Text payload is {len(text):,} characters, which exceeds the "
            f"{MAX_TEXT_CHARS:,}-character limit. "
            "Split the document into smaller sections or use a URL-based source."
        )

    cleaned = _clean(text)
    if not cleaned:
        raise DocumentEmptyError(
            "The text payload contains no readable content after normalisation."
        )

    doc_id = _doc_id(text.encode())
    resolved_title = title or _infer_title(None, cleaned)

    logger.info(
        "Loaded text document '%s' — %d chars, doc_id=%s",
        resolved_title, len(cleaned), doc_id,
    )
    return PolicyDocument(
        doc_id=doc_id,
        title=resolved_title,
        pages=[(1, cleaned)],
        source_label=source_label,
        metadata={"source": "raw_text", "char_count": len(cleaned)},
    )
