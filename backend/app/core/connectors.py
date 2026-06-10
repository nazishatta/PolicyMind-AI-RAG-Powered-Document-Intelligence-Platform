"""Document connector protocol and built-in implementations.

A DocumentConnector is anything with an async ``fetch()`` method that
returns a :class:`~app.core.document_loader.PolicyDocument`.

The protocol makes it straightforward to add future connectors for
specific public-data APIs (World Bank, EUR-Lex, UNDP, UN Docs, etc.)
without modifying the ingestion route or pipeline.

Example future connector skeleton::

    class EURLexConnector:
        \"\"\"Fetch a regulation from EUR-Lex by CELEX number.\"\"\"

        BASE = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{}"

        def __init__(self, celex_id: str, source_label: str = "EUR-Lex") -> None:
            self._celex = celex_id
            self._label = source_label

        async def fetch(self) -> PolicyDocument:
            url = self.BASE.format(self._celex)
            return await load_from_url(url, source_label=self._label)

        @property
        def source_description(self) -> str:
            return f"EUR-Lex CELEX:{self._celex}"
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from app.core.document_loader import (
    PolicyDocument,
    load_from_text,
    load_from_url,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentConnector(Protocol):
    """Structural interface for all document connectors.

    Any class that implements ``fetch()`` and ``source_description``
    satisfies this protocol — no explicit inheritance required.
    """

    async def fetch(self) -> PolicyDocument:
        """Retrieve and parse the document, returning a :class:`PolicyDocument`."""
        ...

    @property
    def source_description(self) -> str:
        """Human-readable description of this connector's data source."""
        ...


# ---------------------------------------------------------------------------
# URLConnector
# ---------------------------------------------------------------------------

class URLConnector:
    """Fetch a publicly accessible PDF or plain-text document by URL.

    The URL must resolve to a document that does not require authentication.
    For documents behind paywalls or access controls, use a pre-downloaded
    text payload via :class:`TextConnector` instead.

    Users are responsible for verifying that the source document is licensed
    for the intended use. This connector does not cache or store the fetched
    content on disk.

    Args:
        url:          Publicly accessible document URL.
        title:        Optional title override. Auto-detected if omitted.
        source_label: Short organisation label (e.g. "UNDP", "World Bank").
        max_bytes:    Maximum download size. Default 50 MB.
    """

    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        source_label: Optional[str] = None,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.url = url
        self.title = title
        self.source_label = source_label
        self.max_bytes = max_bytes

    async def fetch(self) -> PolicyDocument:
        logger.info("URLConnector: fetching %s", self.url)
        return await load_from_url(
            self.url,
            title=self.title,
            source_label=self.source_label,
            max_bytes=self.max_bytes,
        )

    @property
    def source_description(self) -> str:
        return f"URL: {self.url}"


# ---------------------------------------------------------------------------
# TextConnector
# ---------------------------------------------------------------------------

class TextConnector:
    """Wrap a raw text string as an immediately available document.

    Use this connector when:
    - The document text was extracted externally (e.g. via OCR).
    - A URL is not publicly accessible.
    - The user supplies text directly in the API payload.

    Args:
        text:         The document text content.
        title:        Optional title override.
        source_label: Short organisation label.
    """

    def __init__(
        self,
        text: str,
        title: Optional[str] = None,
        source_label: Optional[str] = None,
    ) -> None:
        self.text = text
        self.title = title
        self.source_label = source_label

    async def fetch(self) -> PolicyDocument:
        return load_from_text(
            self.text,
            title=self.title,
            source_label=self.source_label,
        )

    @property
    def source_description(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Text payload: '{preview}...'"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def connector_from_request(
    *,
    url: Optional[str] = None,
    text: Optional[str] = None,
    title: Optional[str] = None,
    source_label: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
) -> DocumentConnector:
    """Return the appropriate :class:`DocumentConnector` for a request.

    Exactly one of ``url`` or ``text`` must be provided.

    Raises:
        ValueError: If both or neither are provided.
    """
    if url and text:
        raise ValueError("Provide 'url' or 'text', not both.")
    if not url and not text:
        raise ValueError("Provide either 'url' or 'text'.")

    if url:
        return URLConnector(url, title=title, source_label=source_label, max_bytes=max_bytes)
    return TextConnector(text, title=title, source_label=source_label)  # type: ignore[arg-type]
