"""Shared UI helpers for PolicyMind AI Streamlit components."""
from __future__ import annotations

import streamlit as st


def render_step_header(number: int, title: str, subtitle: str = "") -> None:
    """Render a numbered step card with an optional subtitle."""
    subtitle_html = (
        f'<p style="margin:0; color:#666; font-size:0.9rem;">{subtitle}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
<div class="step-card">
    <span style="color:#2d6a9f; font-weight:bold; font-size:0.85rem;">STEP {number}</span>
    <h3 style="margin:0.2rem 0; color:#1e3a5f;">{title}</h3>
    {subtitle_html}
</div>
""",
        unsafe_allow_html=True,
    )
