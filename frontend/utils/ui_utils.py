import html
import streamlit as st
import streamlit.components.v1 as components

import json
import logging
import os
import re
import uuid
import requests
from typing import List
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHATS_URL = f"{BACKEND_URL}/user"
CHAT_HISTORY_URL = f"{BACKEND_URL}/chat"
USERS_URL = f"{BACKEND_URL}/users"
AGENT_URL = f"{BACKEND_URL}/agent/ask"


# --- API Helper Functions with Error Handling ---
def fetch_all_users(retry_count=2):
    """Fetch all users with retry logic."""
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
                st.warning("Connection timeout, retrying...")
                continue
            logger.error(f"Timeout fetching users after {retry_count} attempts")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching users: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []


def delete_chat_api(session_id):
    """Delete a chat session."""
    try:
        requests.delete(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
        logger.info(f"Chat {session_id} deleted")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting chat {session_id}: {e}")
        st.warning(f"Could not delete chat: {str(e)[:50]}")


def delete_user_api(user_id):
    """Delete a user and all their data."""
    try:
        requests.delete(f"{BACKEND_URL}/user/{user_id}/", timeout=5)
        logger.info(f"User {user_id} deleted")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        st.warning(f"Could not delete user: {str(e)[:50]}")


def delete_import_log_api(import_id):
    """Delete an import log."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/ingest/record/{import_id}", timeout=5
        )
        response.raise_for_status()
        logger.info(f"Import log {import_id} deleted")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting import log {import_id}: {e}")
        st.error(f"Could not delete import log: {str(e)[:100]}")
        return False


def update_import_log_api(import_id, data):
    """Update an import log."""
    try:
        response = requests.put(
            f"{BACKEND_URL}/ingest/record/{import_id}", json=data, timeout=5
        )
        response.raise_for_status()
        logger.info(f"Import log {import_id} updated")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error updating import log {import_id}: {e}")
        st.error(f"Could not update import log: {str(e)[:100]}")
        return False


def fetch_user_chats(user_id, retry_count=2):
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


def fetch_chat_history(session_id, retry_count=2):
    """Fetch chat history with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                messages = data.get("messages", [])
                # Validate messages have required fields
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


def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")
    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# This is a placeholder for LangChain's Document class
class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Formats a list of Documents into a single string, where each
    document's page_content is followed by its corresponding metadata.
    """
    # Create a list of formatted strings, one for each document
    formatted_blocks = []
    for doc in docs:
        # Format the metadata as a pretty JSON string
        metadata_str = json.dumps(doc.metadata, indent=2)

        # Create a combined block for the document's content and its metadata
        block = f"Content: \n{doc.page_content}\n--- METADATA ---\n{metadata_str}"
        formatted_blocks.append(block)

    # Join all the individual document blocks with a clear separator
    return "\n\n======================================================\n\n".join(
        formatted_blocks
    )


def render_message_with_mermaid(content, key_suffix=""):
    """Parses a message and renders Markdown and Mermaid blocks separately."""
    parts = re.split(
        r"(```mermaid\s+.*?\s*```)", content, flags=re.DOTALL | re.IGNORECASE
    )

    for i, part in enumerate(parts):
        part = part.strip()

        if part.lower().startswith("```mermaid"):
            # Extract mermaid code by removing fences
            mermaid_code = part.removeprefix("```mermaid").removesuffix("```").strip()

            if mermaid_code:
                try:
                    # Generate a unique ID for this diagram
                    unique_id = f"mermaid-{uuid.uuid4()}"

                    # Escape the code to prevent HTML injection/breaking
                    escaped_code = html.escape(mermaid_code)

                    # st_mermaid(mermaid_code)
                    mermaid_html = f"""
                        <div class="mermaid" id="{unique_id}">
                            {escaped_code}
                        </div>
                        <script type="module">
                            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                            mermaid.initialize({{ startOnLoad: true }});
                            try {{
                                await mermaid.run({{
                                    querySelector: '#{unique_id}'
                                }});
                            }} catch(e) {{
                                console.error('Mermaid error:', e);
                                const div = document.getElementById('{unique_id}');
                                if (div) {{
                                    div.innerHTML = '<pre style="color:red; background: #fee; padding: 10px; border-radius: 5px;">' + e.message + '</pre>';
                                }}
                            }}
                        </script>
                    """
                    st.iframe(mermaid_html)
                except Exception as e:
                    st.error(f"Failed to render Mermaid diagram: {e}")
                    st.code(mermaid_code, language="mermaid")
        elif part:
            # Render regular markdown
            st.markdown(part)


CONFIG_URL = f"{BACKEND_URL}/config"  # New API endpoint


# --- 🆕 Function to fetch and display container name ---
def display_container_name():
    """Fetches and displays the Neo4j container name in the sidebar."""
    try:
        with st.sidebar:
            with st.spinner("Connecting to DB..."):
                response = requests.get(CONFIG_URL)
                response.raise_for_status()
                data = response.json()
                container_name = data.get("container_name", "N/A")
                st.success(f"DB Connected: **{container_name}**", icon="🐳")
    except requests.exceptions.RequestException:
        st.sidebar.error("**DB Status:** Connection failed.")


# --- Config Func ---
def get_system_config():
    """Fetches configuration from the backend API."""
    try:
        response = requests.get(CONFIG_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch config: {e}")
        return None  # Return None on failure


def get_database_summary():
    """Get summary statistics from the database via API."""
    try:
        response = requests.get(f"{BACKEND_URL}/stats/summary")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching DB summary: {e}")

    return {
        "total_questions": 0,
        "total_tags": 0,
        "total_answers": 0,
        "total_users": 0,
        "total_imports": 0,
        "last_import": None,
    }


def get_import_history(limit: int = 20):
    """Get recent import history from API."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/stats/history", params={"limit": limit}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching import history: {e}")
    return []


def get_entity_counts():
    """Get counts for all entity types from API."""
    try:
        response = requests.get(f"{BACKEND_URL}/stats/entity_counts")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching entity counts: {e}")
    return {"nodes": {}, "relationships": {}}


def search_nodes(search_term: str, limit: int = 10):
    """Search for nodes by title, name, or display_name via API."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/graph/search",
            params={"term": search_term, "limit": limit},
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error searching nodes: {e}")
    return []


def get_graph_sample(
    node_types: list,
    rel_types: list,
    limit: int = 50,
    focus_node_id: str = "",
):
    """Fetch a sample of nodes and relationships for visualization via API."""
    try:
        payload = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit,
            "focus_node_id": focus_node_id,
        }
        response = requests.post(f"{BACKEND_URL}/graph/sample", json=payload)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching graph sample: {e}")

    return {"nodes": [], "edges": []}
