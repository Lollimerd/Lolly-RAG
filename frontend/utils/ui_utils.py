import json
import logging
import os
from typing import List
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CONFIG_URL = f"{BACKEND_URL}/config"
CHATS_URL = f"{BACKEND_URL}/user"
CHAT_HISTORY_URL = f"{BACKEND_URL}/chat"
USERS_URL = f"{BACKEND_URL}/users"
AGENT_URL = f"{BACKEND_URL}/agent/ask"

def fetch_all_users(retry_count=2):
    """Args: retry_count: Number of retries."""
    for _ in range(retry_count):
        try:
            res = requests.get(USERS_URL, timeout=5)
            if res.ok and (d := res.json()).get("status") == "success": return d.get("users", [])
        except requests.exceptions.RequestException as e: logger.error("Error fetching users: %s", e)
    return []

def delete_chat_api(session_id):
    """Args: session_id: Session ID."""
    try: requests.delete(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
    except requests.exceptions.RequestException as e: st.warning(f"Could not delete chat: {str(e)[:50]}")

def delete_user_api(user_id):
    """Args: user_id: User ID."""
    try: requests.delete(f"{BACKEND_URL}/user/{user_id}/", timeout=5)
    except requests.exceptions.RequestException as e: st.warning(f"Could not delete user: {str(e)[:50]}")

def fetch_user_chats(user_id, retry_count=2):
    """Args: user_id: User ID, retry_count: Number of retries."""
    for _ in range(retry_count):
        try:
            res = requests.get(f"{CHATS_URL}/{user_id}/chats", timeout=5)
            if res.ok and (d := res.json()).get("status") == "success": return d.get("chats", [])
        except requests.exceptions.RequestException as e: logger.error("Error fetching chats: %s", e)
    return []

def fetch_chat_history(session_id, retry_count=2):
    """Args: session_id: Session ID, retry_count: Number of retries."""
    for _ in range(retry_count):
        try:
            res = requests.get(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
            if res.ok and (d := res.json()).get("status") == "success":
                return [m for m in d.get("messages", []) if isinstance(m, dict) and "role" in m and "content" in m]
        except requests.exceptions.RequestException as e: logger.error("Error fetching history: %s", e)
    return []

def extract_title_and_question(input_string):
    """Args: input_string: Input text."""
    title, question, is_q = "", "", False
    for line in input_string.strip().split("\n"):
        if line.startswith("Title:"): title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"): question = line.split("Question: ", 1)[1].strip(); is_q = True
        elif is_q: question += "\n" + line.strip()
    return title, question

def format_docs(docs):
    """Args: docs: Document list."""
    return "\n\n".join(doc.page_content for doc in docs)

class Document:
    """Placeholder for LangChain's Document class."""
    def __init__(self, page_content: str, metadata: dict):
        """Args: page_content: Text content, metadata: Metadata dictionary."""
        self.page_content, self.metadata = page_content, metadata

def format_docs_with_metadata(docs: List[Document]) -> str:
    """Args: docs: Document list."""
    blocks = [f"Content: \n{doc.page_content}\n--- METADATA ---\n{json.dumps(doc.metadata, indent=2)}" for doc in docs]
    return "\n\n======================================================\n\n".join(blocks)

def display_container_name():
    """Args: None."""
    try:
        with st.sidebar:
            with st.spinner("Connecting to database..."):
                res = requests.get(CONFIG_URL, timeout=4)
                name = res.json().get("container_name", "N/A") if res.ok else "N/A"
                st.success(f"DB: **{name}**", icon=":material/database:")
    except requests.exceptions.RequestException:
        st.sidebar.error("Database connection offline", icon=":material/error:")

def get_system_config():
    """Args: None."""
    try:
        res = requests.get(CONFIG_URL)
        return res.json() if res.ok else None
    except requests.exceptions.RequestException as e:
        logger.error("Could not fetch config: %s", e)
        return None

def get_database_summary():
    """Args: None."""
    try:
        res = requests.get(f"{BACKEND_URL}/stats/summary")
        if res.ok: return res.json()
    except Exception as e: logger.error("Error fetching DB summary: %s", e)
    return {"total_documents": 0, "total_chunks": 0, "total_users": 0, "total_sessions": 0, "total_messages": 0}

def get_import_history(limit: int = 20):
    """Args: limit: Maximum entries."""
    try:
        res = requests.get(f"{BACKEND_URL}/stats/history", params={"limit": limit})
        if res.ok: return res.json()
    except Exception as e: logger.error("Error fetching history: %s", e)
    return []

def get_entity_counts():
    """Args: None."""
    try:
        res = requests.get(f"{BACKEND_URL}/stats/entity_counts")
        if res.ok: return res.json()
    except Exception as e: logger.error("Error fetching entity counts: %s", e)
    return {"nodes": {}, "relationships": {}}

def search_nodes(search_term: str, limit: int = 10):
    """Args: search_term: Search query, limit: Maximum matches."""
    try:
        res = requests.get(f"{BACKEND_URL}/graph/search", params={"term": search_term, "limit": limit})
        if res.ok: return res.json()
    except Exception as e: logger.error("Error searching nodes: %s", e)
    return []

def get_graph_sample(node_types: list, rel_types: list, limit: int = 50, focus_node_id: str = ""):
    """Args: node_types: Node types, rel_types: Relationship types, limit: Max elements, focus_node_id: Root node ID."""
    try:
        res = requests.post(f"{BACKEND_URL}/graph/sample", json={"node_types": node_types, "rel_types": rel_types, "limit": limit, "focus_node_id": focus_node_id})
        if res.ok: return res.json()
    except Exception as e: logger.error("Error fetching graph sample: %s", e)
    return {"nodes": [], "edges": []}
