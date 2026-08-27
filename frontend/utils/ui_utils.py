import html
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHATS_URL = f"{BACKEND_URL}/user"
CHAT_HISTORY_URL = f"{BACKEND_URL}/chat"
USERS_URL = f"{BACKEND_URL}/users"
AGENT_URL = f"{BACKEND_URL}/agent/ask"
CONFIG_URL = f"{BACKEND_URL}/config"


def clear_api_cache() -> None:
    """Clears cached API data to ensure immediate UI synchronization after state mutations."""
    fetch_all_users.clear()
    get_system_config.clear()
    get_database_summary.clear()
    get_import_history.clear()
    get_entity_counts.clear()


# --- API Helper Functions with Error Handling & Caching ---
@st.cache_data(ttl=15, show_spinner=False)
def fetch_all_users(retry_count: int = 2) -> list:
    """Fetch all users with retry logic and caching."""
    for attempt in range(retry_count):
        try:
            response = requests.get(USERS_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data.get("users", [])
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            logger.error(f"Timeout fetching users after {retry_count} attempts")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching users: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []


def delete_chat_api(session_id: str) -> None:
    """Delete a chat session."""
    try:
        requests.delete(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
        logger.info(f"Chat {session_id} deleted")
        clear_api_cache()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting chat {session_id}: {e}")
        st.warning(f"Could not delete chat: {str(e)[:50]}")


def delete_user_api(user_id: str) -> None:
    """Delete a user and all their data."""
    try:
        requests.delete(f"{BACKEND_URL}/user/{user_id}/", timeout=5)
        logger.info(f"User {user_id} deleted")
        clear_api_cache()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        st.warning(f"Could not delete user: {str(e)[:50]}")


def delete_import_log_api(import_id: str) -> bool:
    """Delete an import log."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/ingest/record/{import_id}", timeout=5
        )
        response.raise_for_status()
        logger.info(f"Import log {import_id} deleted")
        clear_api_cache()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting import log {import_id}: {e}")
        st.error(f"Could not delete import log: {str(e)[:100]}")
        return False


def update_import_log_api(import_id: str, data: dict) -> bool:
    """Update an import log."""
    try:
        response = requests.put(
            f"{BACKEND_URL}/ingest/record/{import_id}", json=data, timeout=5
        )
        response.raise_for_status()
        logger.info(f"Import log {import_id} updated")
        clear_api_cache()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error updating import log {import_id}: {e}")
        st.error(f"Could not update import log: {str(e)[:100]}")
        return False


def fetch_user_chats(user_id: str, retry_count: int = 2) -> list:
    """Fetch user's chat sessions with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(f"{CHATS_URL}/{user_id}/chats", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data.get("chats", [])
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            logger.error(f"Timeout fetching chats for user {user_id}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching chats for user {user_id}: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []


def fetch_chat_history(session_id: str, retry_count: int = 2) -> list:
    """Fetch chat history with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                messages = data.get("messages", [])
                validated = []
                for msg in messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        validated.append(msg)
                return validated
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            logger.error(f"Timeout fetching history for {session_id}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching history for {session_id}: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []


def extract_title_and_question(input_string: str) -> tuple[str, str]:
    lines = input_string.strip().split("\n")
    title = ""
    question = ""
    is_question = False

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = True
        elif is_question:
            question += "\n" + line.strip()

    return title, question


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def format_docs_with_metadata(docs: List[Document]) -> str:
    formatted_blocks = []
    for doc in docs:
        metadata_str = json.dumps(doc.metadata, indent=2)
        block = f"Content: \n{doc.page_content}\n--- METADATA ---\n{metadata_str}"
        formatted_blocks.append(block)

    return "\n\n======================================================\n\n".join(
        formatted_blocks
    )


def display_container_name():
    """Fetches and displays the Neo4j container name in the sidebar."""
    config = get_system_config()
    if config and config.get("status") == "success":
        container_name = config.get("container_name", "N/A")
        st.sidebar.success(f"DB Connected: **{container_name}**", icon=":material/database:")
    else:
        st.sidebar.error("DB Status: Connection failed", icon=":material/error:")


@st.cache_data(ttl=60, show_spinner=False)
def get_system_config() -> Optional[dict]:
    """Fetches configuration from the backend API with caching."""
    try:
        response = requests.get(CONFIG_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch config: {e}")
        return None


@st.cache_data(ttl=15, show_spinner=False)
def get_database_summary() -> dict:
    """Get summary statistics from the database via API with caching."""
    try:
        response = requests.get(f"{BACKEND_URL}/stats/summary", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Error fetching DB summary: {e}")

    return {
        "total_questions": 0,
        "total_tags": 0,
        "total_answers": 0,
        "total_users": 0,
        "total_imports": 0,
        "last_import": None,
    }


@st.cache_data(ttl=15, show_spinner=False)
def get_import_history(limit: int = 20) -> list:
    """Get recent import history from API with caching."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/stats/history", params={"limit": limit}, timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Error fetching import history: {e}")
    return []


@st.cache_data(ttl=15, show_spinner=False)
def get_entity_counts() -> dict:
    """Get counts for all entity types from API with caching."""
    try:
        response = requests.get(f"{BACKEND_URL}/stats/entity_counts", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Error fetching entity counts: {e}")
    return {"nodes": {}, "relationships": {}}


def search_nodes(search_term: str, limit: int = 10) -> list:
    """Search for nodes by title, name, or display_name via API."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/graph/search",
            params={"term": search_term, "limit": limit},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Error searching nodes: {e}")
    return []


def get_graph_sample(
    node_types: list,
    rel_types: list,
    limit: int = 50,
    focus_node_id: str = "",
) -> dict:
    """Fetch a sample of nodes and relationships for visualization via API."""
    try:
        payload = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit,
            "focus_node_id": focus_node_id,
        }
        response = requests.post(f"{BACKEND_URL}/graph/sample", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.warning(f"Error fetching graph sample: {e}")

    return {"nodes": [], "edges": []}
