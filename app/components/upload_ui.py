"""Streamlit upload section — multi-PDF uploader, summary table, and per-document preview."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.ui_helpers import render_step_header
from src.document_loader import process_multiple_pdfs
from src.logger import get_logger

logger = get_logger(__name__)


def render_upload_section() -> list[dict[str, Any]] | None:
    """Render the multi-file PDF uploader with a summary table and document previews.

    Accepts one or more PDF files.  Each file is saved to data/raw/ and text is
    extracted with PyMuPDF.  Failures on individual files are skipped; the rest
    are still processed so a single bad PDF does not abort a multi-file upload.

    Returns:
        Combined list of page dicts (document_name, page_number, text) from all
        successfully processed files, or None if nothing was uploaded.
    """
    render_step_header(
        1,
        "Upload Documents",
        "Upload one or more PDF policy documents",
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF policy documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload one or more PDF files. All documents will be indexed together "
            "into a single knowledge base."
        ),
    )

    # Streamlit returns an empty list (not None) when accept_multiple_files=True
    # and nothing has been dropped yet.
    if not uploaded_files:
        st.markdown("""
<div style="text-align:center; padding:3rem;
            color:#888; border:2px dashed #ddd;
            border-radius:12px; margin:2rem 0;">
    <h3>📄 No documents uploaded yet</h3>
    <p>Upload a PDF policy document to get started</p>
    <p style="font-size:0.85rem;">Supports: Policy reports,
    World Bank documents, Government publications,
    Research papers</p>
</div>
""", unsafe_allow_html=True)
        return None

    try:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)…"):
            all_pages, summary = process_multiple_pdfs(uploaded_files)

        if not summary["files_processed"]:
            st.error(
                "No files could be processed successfully. "
                "Check that the uploads are valid PDF files and try again."
            )
            return None

        total_files = summary["total_files"]
        total_pages = summary["total_pages"]
        failed = len(uploaded_files) - total_files

        # Success / partial-success banner
        if failed == 0:
            st.success(
                f"{total_files} document(s) loaded — {total_pages} total pages extracted"
            )
        else:
            st.warning(
                f"{total_files} document(s) loaded — {total_pages} total pages extracted "
                f"({failed} file(s) could not be processed and were skipped)"
            )

        # Summary table: File Name | Pages | Status
        page_counts: dict[str, int] = {}
        for page in all_pages:
            doc = page.get("document_name", "")
            page_counts[doc] = page_counts.get(doc, 0) + 1

        failed_names = {
            getattr(f, "name", str(f))
            for f in uploaded_files
            if getattr(f, "name", str(f)) not in summary["files_processed"]
        }

        all_names = summary["files_processed"] + sorted(failed_names)
        table_data = {
            "File Name": all_names,
            "Pages": [
                page_counts.get(n, 0) for n in all_names
            ],
            "Status": [
                "✅ Loaded" if n in summary["files_processed"] else "❌ Failed"
                for n in all_names
            ],
        }
        st.dataframe(table_data, width="stretch")

        # Totals row below the table
        col1, col2 = st.columns(2)
        col1.metric("Files uploaded", len(uploaded_files))
        col2.metric("Total pages", total_pages)

        # Per-document text preview
        with st.expander("Preview documents"):
            for fname in summary["files_processed"]:
                st.markdown(f"**{fname}**")
                file_pages = [p for p in all_pages if p.get("document_name") == fname]
                preview_text = "\n\n".join(p["text"] for p in file_pages)[:500]
                st.text(preview_text if preview_text else "(no text extracted)")
                st.markdown("---")

        return all_pages

    except Exception as exc:
        st.error(f"Failed to process uploaded files: {exc}")
        logger.error("render_upload_section error: %s", exc)
        return None
