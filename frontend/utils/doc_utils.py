"""
doc_utils.py
------------
Constants and API helper functions for unstructured document ingestion,
metadata parsing, folder management, and chunk inspection in the Streamlit frontend.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Configuration & URLs
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
INGEST_DOC_URL = f"{BACKEND_URL}/ingest/documents"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_TYPES: List[str] = [
    "pdf", "docx", "pptx", "ppt", "txt", "md", "csv", "xlsx", "xls",
    "png", "jpg", "jpeg", "webp", "bmp", "tiff",
]

FILE_TYPE_INFO: Dict[str, Dict[str, str]] = {
    "pdf": {"icon": "📕", "label": "PDF Document", "color": "#EF4444"},
    "docx": {"icon": "📘", "label": "Word Document", "color": "#3B82F6"},
    "pptx": {"icon": "📊", "label": "PowerPoint Presentation", "color": "#EA580C"},
    "ppt": {"icon": "📊", "label": "PowerPoint 97-2003", "color": "#C2410C"},
    "txt": {"icon": "📄", "label": "Text File", "color": "#64748B"},
    "md": {"icon": "📝", "label": "Markdown File", "color": "#8B5CF6"},
    "csv": {"icon": "📊", "label": "CSV Table", "color": "#10B981"},
    "xlsx": {"icon": "📈", "label": "Excel Workbook", "color": "#059669"},
    "xls": {"icon": "📈", "label": "Excel 97-2003", "color": "#047857"},
    "png": {"icon": "🖼️", "label": "PNG Image (OCR)", "color": "#06B6D4"},
    "jpg": {"icon": "🖼️", "label": "JPEG Image (OCR)", "color": "#0284C7"},
    "jpeg": {"icon": "🖼️", "label": "JPEG Image (OCR)", "color": "#0284C7"},
    "webp": {"icon": "🖼️", "label": "WebP Image (OCR)", "color": "#0EA5E9"},
    "bmp": {"icon": "🖼️", "label": "Bitmap Image (OCR)", "color": "#38BDF8"},
    "tiff": {"icon": "🖼️", "label": "TIFF Image (OCR)", "color": "#38BDF8"},
}

INITIAL_FOLDERS: List[str] = [
    "Root",
]


# ---------------------------------------------------------------------------
# API Helper Functions
# ---------------------------------------------------------------------------
def fetch_documents() -> List[Dict[str, Any]]:
    """Fetch the list of all ingested documents from the backend."""
    try:
        resp = requests.get(INGEST_DOC_URL, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception as exc:
        logger.error("Failed to fetch document list: %s", exc)
        st.error(f"Failed to fetch document list: {exc}")
        return []


def fetch_document_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Fetch chunks for a specific document."""
    try:
        resp = requests.get(f"{INGEST_DOC_URL}/{doc_id}/chunks", timeout=10)
        resp.raise_for_status()
        return resp.json().get("chunks", [])
    except Exception as exc:
        logger.error("Failed to fetch chunks for doc %s: %s", doc_id, exc)
        return []


def upload_file(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    description: str,
    folder: str,
    force: bool = False,
    engine: str = "pandas",
) -> Dict[str, Any]:
    """POST a file to backend with folder metadata and selected engine."""
    folder_prefix = f"[{folder}] " if folder and folder != "Root" else ""
    full_description = f"{folder_prefix}{description}".strip()

    resp = requests.post(
        INGEST_DOC_URL,
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={
            "user_id": user_id,
            "description": full_description,
            "force": str(force).lower(),
            "engine": engine,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def update_document_metadata(doc_id: str, folder: str, description: str) -> bool:
    """Update folder/description metadata for a document."""
    folder_prefix = f"[{folder}] " if folder and folder != "Root" else ""
    full_description = f"{folder_prefix}{description}".strip()
    try:
        resp = requests.put(
            f"{INGEST_DOC_URL}/{doc_id}",
            json={"description": full_description},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as exc:
        logger.error("Failed to update document %s: %s", doc_id, exc)
        st.error(f"Failed to update document: {exc}")
        return False


def delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks from Neo4j."""
    try:
        resp = requests.delete(f"{INGEST_DOC_URL}/{doc_id}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", doc_id, exc)
        st.error(f"Failed to delete document: {exc}")
        return False


def delete_all_in_folder(files: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Delete all documents in a given list of files. Returns (succeeded, failed)."""
    succeeded, failed = 0, 0
    for doc in files:
        doc_id = doc.get("id")
        if doc_id:
            if delete_document(doc_id):
                succeeded += 1
            else:
                failed += 1
    return succeeded, failed


def parse_folder(description: Optional[str]) -> Tuple[str, str]:
    """Extract folder name from description if prefixed like '[Folder] rest of desc'."""
    desc = (description or "").strip()
    if desc.startswith("[") and "]" in desc:
        end_idx = desc.index("]")
        folder = desc[1:end_idx].strip()
        clean_desc = desc[end_idx + 1 :].strip()
        return folder or "Root", clean_desc
    return "Root", desc


def get_all_folders(docs: List[Dict[str, Any]]) -> List[str]:
    """Return sorted unique list of all folders (discovered + custom added in session)."""
    if "custom_folders" not in st.session_state:
        st.session_state["custom_folders"] = list(INITIAL_FOLDERS)

    # Collect folders discovered from existing documents
    discovered = {f for doc in docs for f, _ in [parse_folder(doc.get("description", ""))]}

    # Union with session custom folders
    all_f = set(st.session_state["custom_folders"]).union(discovered)
    if "Root" not in all_f:
        all_f.add("Root")

    # Return with 'Root' first, then alphabetical
    others = sorted([f for f in all_f if f != "Root"])
    return ["Root"] + others


# Backward-compatibility aliases with underscore prefixes
_fetch_documents = fetch_documents
_fetch_document_chunks = fetch_document_chunks
_upload_file = upload_file
_update_document_metadata = update_document_metadata
_delete_document = delete_document
_delete_all_in_folder = delete_all_in_folder
_parse_folder = parse_folder
_get_all_folders = get_all_folders
