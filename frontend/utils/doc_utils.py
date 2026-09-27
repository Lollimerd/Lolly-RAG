"""
doc_utils.py
------------
Constants and API helper functions for unstructured document ingestion,
metadata parsing, folder management, and chunk inspection in the Streamlit frontend.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
INGEST_DOC_URL = f"{BACKEND_URL}/ingest/documents"
INGEST_DOC_STREAM_URL = f"{BACKEND_URL}/ingest/documents/stream"

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

INITIAL_FOLDERS: List[str] = ["Root"]

def _engine_for(filename: str, engine: Optional[str]) -> str:
    """Args: filename: Name of file, engine: Explicit engine or None."""
    return engine or ("apoc" if filename.lower().endswith(".csv") else "pandas")

def _full_desc(folder: str, description: str) -> str:
    """Args: folder: Folder name, description: Description text."""
    return f"[{folder}] {description}".strip() if folder and folder != "Root" else (description or "").strip()

def fetch_documents() -> List[Dict[str, Any]]:
    """Args: None."""
    try:
        resp = requests.get(INGEST_DOC_URL, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception as exc:
        logger.error("Failed to fetch document list: %s", exc)
        st.error(f"Failed to fetch document list: {exc}")
        return []

def fetch_document_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Args: doc_id: Document ID."""
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
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """Args: file_bytes: Content bytes, filename: File name, user_id: User ID, description: Doc desc, folder: Folder name, force: Overwrite flag, engine: Engine name."""
    resp = requests.post(
        INGEST_DOC_URL,
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={"user_id": user_id, "description": _full_desc(folder, description), "force": str(force).lower(), "engine": _engine_for(filename, engine)},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

def upload_file_stream(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    description: str,
    folder: str,
    force: bool = False,
    engine: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Args: file_bytes: Content bytes, filename: File name, user_id: User ID, description: Doc desc, folder: Folder name, force: Overwrite flag, engine: Engine name, progress_callback: Progress callback."""
    engine_name = _engine_for(filename, engine)
    data = {"user_id": user_id, "description": _full_desc(folder, description), "force": str(force).lower(), "engine": engine_name}
    resp = requests.post(INGEST_DOC_STREAM_URL, files={"file": (filename, file_bytes, "application/octet-stream")}, data=data, stream=True, timeout=600)
    if resp.status_code == 404:
        logger.warning("Streaming endpoint 404; falling back to standard upload")
        return upload_file(file_bytes, filename, user_id, description, folder, force, engine_name)
    resp.raise_for_status()
    final_result: Dict[str, Any] = {"status": "error", "message": "No response"}
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8").strip()
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
                if progress_callback:
                    progress_callback(event)
                if event.get("type") in ("complete", "error"):
                    final_result = event
            except json.JSONDecodeError:
                pass
    return final_result

def update_document_metadata(doc_id: str, folder: str, description: str) -> bool:
    """Args: doc_id: Document ID, folder: Folder name, description: Description."""
    try:
        resp = requests.put(f"{INGEST_DOC_URL}/{doc_id}", json={"description": _full_desc(folder, description)}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as exc:
        logger.error("Failed to update document %s: %s", doc_id, exc)
        st.error(f"Failed to update document: {exc}")
        return False

def delete_document(doc_id: str) -> bool:
    """Args: doc_id: Document ID."""
    try:
        resp = requests.delete(f"{INGEST_DOC_URL}/{doc_id}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", doc_id, exc)
        st.error(f"Failed to delete document: {exc}")
        return False

def delete_all_in_folder(files: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Args: files: List of doc dicts."""
    succeeded = sum(1 for d in files if d.get("id") and delete_document(d["id"]))
    return succeeded, len(files) - succeeded

def parse_folder(description: Optional[str]) -> Tuple[str, str]:
    """Args: description: Raw document description."""
    desc = (description or "").strip()
    if desc.startswith("[") and "]" in desc:
        idx = desc.index("]")
        return desc[1:idx].strip() or "Root", desc[idx + 1:].strip()
    return "Root", desc

def get_all_folders(docs: List[Dict[str, Any]]) -> List[str]:
    """Args: docs: Ingested document records."""
    if "custom_folders" not in st.session_state:
        st.session_state["custom_folders"] = list(INITIAL_FOLDERS)
    discovered = {parse_folder(doc.get("description", ""))[0] for doc in docs}
    all_f = set(st.session_state["custom_folders"]) | discovered | {"Root"}
    return ["Root"] + sorted(f for f in all_f if f != "Root")
