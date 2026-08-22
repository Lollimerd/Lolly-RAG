from langchain_core.documents import Document
from typing import List, Any
import json, docker, re, os, socket

import time

# ---------------------------------------------------------------------------
# Per-request tool-call counter and retrieval duration timer
# ---------------------------------------------------------------------------
_tool_call_counts: dict[str, int] = {}
_tool_call_start_times: dict[str, float] = {}
_latest_session_id: str = ""

MAX_TOOL_CALLS = 1
MAX_RETRIEVAL_DURATION_SECONDS = 30.0

def reset_tool_call_count(session_id: str) -> None:
    """Call once at the start of each agent invocation to reset the counter and timer."""
    global _latest_session_id
    if session_id:
        _latest_session_id = session_id
    _tool_call_counts[session_id] = 0
    _tool_call_start_times[session_id] = time.time()


def get_active_session_id(session_id: str = "") -> str:
    """Return the passed session_id or fallback to the latest active session ID."""
    return session_id or _latest_session_id


def get_tool_call_count(session_id: str = "") -> int:
    """Return how many times the tool has been called for this request."""
    sid = get_active_session_id(session_id)
    return _tool_call_counts.get(sid, 0)


def get_tool_call_start_time(session_id: str = "") -> float | None:
    """Return the start time of the retrieval request."""
    sid = get_active_session_id(session_id)
    return _tool_call_start_times.get(sid)


def increment_tool_call_count(session_id: str = "") -> int:
    """Increment and return the new count."""
    sid = get_active_session_id(session_id)
    new_count = _tool_call_counts.get(sid, 0) + 1
    _tool_call_counts[sid] = new_count
    return new_count


def check_retrieval_hard_stop(session_id: str = "") -> tuple[bool, str]:
    """
    Checks if retrieval should hard stop based on either:
    1. Maximum tool call count reached (max 1 per query)
    2. Maximum retrieval duration reached (30 seconds max)

    Returns (should_stop, message).
    """
    sid = get_active_session_id(session_id)
    current_count = _tool_call_counts.get(sid, 0)
    start_time = _tool_call_start_times.get(sid)

    if current_count >= MAX_TOOL_CALLS:
        return True, (
            "[HARD STOP] Tool call limit reached (max 1 per query). "
            "You have already searched the knowledge base the maximum number of times. "
            "Do NOT call this tool again. Formulate your final answer now using what you have."
        )

    if start_time is not None:
        elapsed = time.time() - start_time
        if elapsed >= MAX_RETRIEVAL_DURATION_SECONDS:
            return True, (
                f"[HARD STOP] Retrieval duration limit reached ({elapsed:.1f}s >= {MAX_RETRIEVAL_DURATION_SECONDS}s max). "
                "Retrieval duration has exceeded 30 seconds. "
                "Do NOT call this tool again. Formulate your final answer now using what you have."
            )

    return False, ""


def escape_lucene_chars(text: str) -> str:
    """
    Escapes special characters in a string for safe use in a Lucene query.
    """
    # List of special characters in Lucene syntax
    special_chars = r'([+\-&|!(){}\[\]^"~*?:\\/])'
    # Prepend each special character with a backslash
    return re.sub(special_chars, r"\\\1", text)


# --- Dynamic Container Discovery ---
def find_container_by_port(port: int) -> str:
    """Inspects running Docker containers to find which one is using the specified port."""
    if not port:
        return "Invalid port"

    try:
        # Connect to the Docker daemon
        client = docker.from_env()
        containers = client.containers.list()
        target_port = str(port)

        for container in containers:
            # The .ports attribute is a dictionary like: {'7687/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '7687'}]}
            port_mappings = container.ports
            for port_key, host_mappings in port_mappings.items():
                # when backend is dockerised
                if port_key.split("/")[0] == target_port:
                    return container.name

                # when running from bash/uvicorn
                if host_mappings:
                    for mapping in host_mappings:
                        if mapping.get("HostPort") == str(port):
                            return container.name  # Found it!
        return "No matching container found"

    except docker.errors.DockerException:
        if os.path.exists("/.dockerenv"):
            hostname = socket.gethostname()
            return f"Self ({hostname}) - Docker socket not mounted?"
        return "Docker daemon not running or not accessible"
    except Exception as e:
        return f"An error occurred: {e}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _format_value_readable(
    value: Any, indent: int = 0, max_list_items: int = 10
) -> str:
    pad = "  " * indent
    next_pad = "  " * (indent + 1)

    # Dict: pretty print as key: value with nesting
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_format_value_readable(v, indent + 1, max_list_items))
            else:
                lines.append(f"{pad}{k}: {_format_scalar(v)}")
        return "\n".join(lines)

    # List: if primitives, inline; if complex, bullets per item
    if isinstance(value, list):
        if not value:
            return "[]"
        # Truncate long lists for readability
        sliced = value[:max_list_items]
        omitted = len(value) - len(sliced)
        if all(not isinstance(x, (dict, list)) for x in sliced):
            return f"{', '.join(_format_scalar(x) for x in sliced)}" + (
                f" …(+{omitted})" if omitted > 0 else ""
            )
        lines = []
        for item in sliced:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_format_value_readable(item, indent + 1, max_list_items))
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        if omitted > 0:
            lines.append(f"{pad}- …(+{omitted} more)")
        return "\n".join(lines)

    # Fallback scalar
    return f"{pad}{_format_scalar(value)}"


def format_docs_with_metadata(docs: list[Document]) -> str:
    """Formats documents and metadata into a Unicode-safe, human-readable string.

    - Preserves Unicode characters (no JSON escaping).
    - Prints nested metadata (dicts/lists) as readable sections and bullet lists.
    - Truncates very long lists to keep context compact.
    """
    formatted_blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        # Page content (already Unicode-safe in Python 3)
        content_section = f"\n--------- CONTENT ---------\n{doc.page_content}"

        # Metadata as readable lines
        metadata_lines: list[str] = []
        for key, value in doc.metadata.items():
            # Section header for complex values
            if isinstance(value, (dict, list)):
                metadata_lines.append(f"{key}:")
                metadata_lines.append(_format_value_readable(value, indent=1))
            else:
                metadata_lines.append(f"{key}: {_format_scalar(value)}")
        metadata_str = "\n".join(metadata_lines)

        metadata_section = f"\n--------- METADATA ---------\n{metadata_str}"

        formatted_blocks.append(content_section + metadata_section)

    final_context_str = "\n\n".join(formatted_blocks)

    # Debug log (stdout) for developers; does not change returned value
    print("\n" + "=" * 100)
    print("--- 📄 RETRIEVED CONTEXT FOR LLM ---")
    print(final_context_str)
    print(f"\n--- 📊 Documents retrieved: {len(docs)} ---")
    print("=" * 100 + "\n")

    return final_context_str


def sanitize_doc_size(doc: Document, max_content_len: int = 2500, max_metadata_str_len: int = 3500) -> Document:
    """
    Truncates a Document's page_content and string-based metadata fields to prevent context explosion and memory overflow.
    """
    # 1. Truncate page content
    content = doc.page_content or ""
    if len(content) > max_content_len:
        content = content[:max_content_len] + "\n... [truncated to prevent context overflow] ..."
    
    # 2. Truncate long string fields in metadata
    new_metadata: dict[str, Any] = {}
    for k, v in doc.metadata.items():
        if isinstance(v, str):
            if len(v) > max_metadata_str_len:
                new_metadata[k] = v[:max_metadata_str_len] + "\n... [truncated] ..."
            else:
                new_metadata[k] = v
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, str) and len(item) > max_metadata_str_len:
                    new_list.append(item[:max_metadata_str_len] + "...")
                else:
                    new_list.append(item)
            new_metadata[k] = new_list
        else:
            new_metadata[k] = v
            
    return Document(page_content=content, metadata=new_metadata)

import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """