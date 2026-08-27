"""
frontend/utils/mermaid.py
--------------------------
Mermaid diagram rendering utilities for the Lolly-RAG frontend.

Features:
1. Native Streamlit rendering via ``st.mermaid_chart`` and ``streamlit-markdown``.
2. Graceful fallback to markdown blocks if rendering fails.
"""

from __future__ import annotations

import logging
import re

import streamlit as st
from streamlit_markdown import st_markdown

logger = logging.getLogger(__name__)

# Matches a full mermaid code fence (case-insensitive)
_FENCE_RE = re.compile(
    r"```(?:mermaid|MERMAID|Mermaid)[ \t]*\r?\n?([\s\S]*?)```",
    re.IGNORECASE,
)

def render_mermaid(code: str, key_suffix: str = "") -> None:
    """
    Renders a raw Mermaid diagram using Streamlit's native ``st.mermaid_chart``.
    Falls back to a markdown code block if rendering fails.

    Parameters
    ----------
    code:
        Raw Mermaid source without surrounding ``` fences.
    key_suffix:
        Optional suffix for element keys.
    """
    code = code.strip()
    if not code:
        return

    try:
        st.mermaid_chart(code, width="stretch")
    except Exception as exc:
        logger.warning(
            "st.mermaid_chart failed (%s). Falling back to Markdown/code block.",
            exc,
        )
        with st.expander(":material/error: Diagram could not be rendered", expanded=False):
            st.code(code, language="mermaid")

# Alias for backward compatibility
render_mermaid_with_retry = render_mermaid

def render_message_with_mermaid(text: str, key_suffix: str = "") -> None:
    """
    Render a message containing a mix of plain Markdown and Mermaid diagram blocks.

    Splits ``text`` on Mermaid code fences:
    - Plain-text segments are rendered with ``st_markdown()``.
    - Mermaid segments are rendered natively with ``st.mermaid_chart()``.

    Parameters
    ----------
    text:
        Message content potentially containing one or more ` ```mermaid ``` ` blocks.
    key_suffix:
        Unique suffix propagated to child element keys.
    """
    if not text:
        return

    parts = _FENCE_RE.split(text)

    for idx, part in enumerate(parts):
        if not part:
            continue

        if idx % 2 == 0:
            # Plain Markdown segment
            st_markdown(part, key=f"md_{key_suffix}_{idx}" if key_suffix else None)
        else:
            # Mermaid diagram source
            render_mermaid(part.strip(), key_suffix=f"mermaid_{key_suffix}_{idx}")
