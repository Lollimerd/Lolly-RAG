"""Utility helpers: tool-call counter, Docker container discovery, and document formatting."""

import json
import logging
import os
import re
import socket
import time
from typing import Any, List

import docker
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Per-request tool-call counter and retrieval timer
_tool_call_counts: dict[str, int] = {}
_tool_call_start_times: dict[str, float] = {}
_latest_session_id: str = ""
MAX_TOOL_CALLS = 2
MAX_RETRIEVAL_DURATION_SECONDS = 30.0

def reset_tool_call_count(session_id: str) -> None:
    """Args: session_id: Session identifier."""
    global _latest_session_id
    if session_id: _latest_session_id = session_id
    _tool_call_counts[session_id], _tool_call_start_times[session_id] = 0, time.time()

def get_active_session_id(session_id: str = "") -> str:
    """Args: session_id: Optional session identifier."""
    return session_id or _latest_session_id

def get_tool_call_count(session_id: str = "") -> int:
    """Args: session_id: Optional session identifier."""
    return _tool_call_counts.get(get_active_session_id(session_id), 0)

def get_tool_call_start_time(session_id: str = "") -> float | None:
    """Args: session_id: Optional session identifier."""
    return _tool_call_start_times.get(get_active_session_id(session_id))

def increment_tool_call_count(session_id: str = "") -> int:
    """Args: session_id: Optional session identifier."""
    sid = get_active_session_id(session_id)
    _tool_call_counts[sid] = _tool_call_counts.get(sid, 0) + 1
    return _tool_call_counts[sid]

def check_retrieval_hard_stop(session_id: str = "") -> tuple[bool, str]:
    """Args: session_id: Optional session identifier."""
    sid = get_active_session_id(session_id)
    if _tool_call_counts.get(sid, 0) >= MAX_TOOL_CALLS:
        return True, "[HARD STOP] Tool call limit reached (max 1 per query). Do NOT call this tool again. Formulate your final answer now."
    start = _tool_call_start_times.get(sid)
    if start is not None and (elapsed := time.time() - start) >= MAX_RETRIEVAL_DURATION_SECONDS:
        return True, f"[HARD STOP] Retrieval duration limit reached ({elapsed:.1f}s >= {MAX_RETRIEVAL_DURATION_SECONDS}s max). Do NOT call this tool again."
    return False, ""

def escape_lucene_chars(text: str) -> str:
    """Args: text: Query string to escape."""
    return re.sub(r'([+\-&|!(){}[\]^"~*?:\\/])', r"\\\1", text)

def find_container_by_port(port: int) -> str:
    """Args: port: Port number to inspect."""
    if not port: return "Invalid port"
    try:
        client, target = docker.from_env(), str(port)
        for c in client.containers.list():
            for k, mappings in c.ports.items():
                if k.split("/")[0] == target or (mappings and any(m.get("HostPort") == target for m in mappings)):
                    return c.name
        return "No matching container found"
    except docker.errors.DockerException:
        return f"Self ({socket.gethostname()}) - Docker socket not mounted?" if os.path.exists("/.dockerenv") else "Docker daemon not running or not accessible"
    except Exception as e:
        return f"An error occurred: {e}"

def _format_scalar(value: Any) -> str:
    """Args: value: Scalar object to stringify."""
    return "true" if value is True else "false" if value is False else "null" if value is None else str(value)

def _format_value_readable(value: Any, indent: int = 0, max_list_items: int = 10) -> str:
    """Args: value: Data object, indent: Indentation level, max_list_items: Truncation threshold."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value: return "{}"
        lines = []
        for k, v in value.items():
            lines += [f"{pad}{k}:", _format_value_readable(v, indent + 1, max_list_items)] if isinstance(v, (dict, list)) else [f"{pad}{k}: {_format_scalar(v)}"]
        return "\n".join(lines)
    if isinstance(value, list):
        if not value: return "[]"
        sliced, omitted = value[:max_list_items], len(value) - min(max_list_items, len(value))
        if all(not isinstance(x, (dict, list)) for x in sliced):
            return ", ".join(_format_scalar(x) for x in sliced) + (f" …(+{omitted})" if omitted else "")
        lines = []
        for item in sliced:
            lines += [f"{pad}-", _format_value_readable(item, indent + 1, max_list_items)] if isinstance(item, (dict, list)) else [f"{pad}- {_format_scalar(item)}"]
        if omitted: lines.append(f"{pad}- …(+{omitted} more)")
        return "\n".join(lines)
    return f"{pad}{_format_scalar(value)}"

def format_docs_with_metadata(docs: list[Document]) -> str:
    """Args: docs: Retrieved document list."""
    blocks = []
    for doc in docs:
        meta_lines = [f"{k}: {_format_scalar(v)}" if not isinstance(v, (dict, list)) else f"{k}:\n{_format_value_readable(v, indent=1)}" for k, v in doc.metadata.items()]
        blocks.append(f"\n--------- CONTENT ---------\n{doc.page_content}\n--------- METADATA ---------\n" + "\n".join(meta_lines))
    result = "\n\n".join(blocks)
    print(f"\n{'='*100}\n--- 📄 RETRIEVED CONTEXT FOR LLM ---\n{result}\n\n--- 📊 Documents retrieved: {len(docs)} ---\n{'='*100}\n")
    return result

def sanitize_doc_size(doc: Document, max_content_len: int = 2500, max_metadata_str_len: int = 3500) -> Document:
    """Args: doc: Document object, max_content_len: Max content length, max_metadata_str_len: Max string field length."""
    content = (doc.page_content or "")[:max_content_len] + ("\n... [truncated] ..." if len(doc.page_content or "") > max_content_len else "")
    new_meta: dict[str, Any] = {}
    for k, v in doc.metadata.items():
        if isinstance(v, str):
            new_meta[k] = v[:max_metadata_str_len] + ("\n... [truncated] ..." if len(v) > max_metadata_str_len else "")
        elif isinstance(v, list):
            new_meta[k] = [(i[:max_metadata_str_len] + "..." if isinstance(i, str) and len(i) > max_metadata_str_len else i) for i in v]
        else:
            new_meta[k] = v
    return Document(page_content=content, metadata=new_meta)