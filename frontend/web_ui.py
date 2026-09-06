# web_ui.py
import json
import logging
import re
import uuid
import httpx
from httpx_sse import connect_sse
import requests
import streamlit as st
from streamlit_markdown import st_markdown

from utils.ui_utils import (
    display_container_name,
    get_system_config,
    fetch_all_users,
    fetch_user_chats,
    fetch_chat_history,
    delete_user_api,
    delete_chat_api,
    BACKEND_URL,
    AGENT_URL,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Lolly-RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "# Lolly-RAG: Agentic GraphRAG Knowledge Platform",
    },
)

# --- Initialize Session State AND Sync with Backend ---
if "chats" not in st.session_state:
    st.session_state.chats = {}


# Helper function to format markdown preview during streaming
def format_stream_preview(text: str) -> str:
    """Mask mermaid blocks as plain code blocks during streaming to avoid premature rendering errors."""
    preview = re.sub(r"```mermaid", "```text", text, flags=re.IGNORECASE)
    if preview.count("```") % 2 == 1:
        return preview + "\n```"
    return preview + " ▌"


# Helper function to get the active chat object
def get_active_chat():
    """Get the active chat data, creating it if necessary."""
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id or chat_id not in st.session_state.chats:
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
            chat_id = st.session_state.active_chat_id
        else:
            return None

    chat = st.session_state.chats[chat_id]

    # Lazy load history if not loaded yet
    if not chat.get("loaded", False):
        msgs = fetch_chat_history(chat_id)
        chat["messages"] = msgs
        chat["loaded"] = True

    return chat


# --- Sidebar ---
with st.sidebar:
    try:
        display_container_name()
    except Exception as e:
        logger.error(f"Error displaying container info: {e}")
        st.warning("Could not connect to backend for system info", icon=":material/warning:")

    # --- System Info ---
    with st.expander("System details", icon=":material/info:", expanded=False):
        try:
            config_data = get_system_config()
            if config_data and isinstance(config_data, dict):
                st_markdown(f"**Model:** `{config_data.get('ollama_model', 'N/A')}`")
                st_markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
                st_markdown(f"**Container:** `{config_data.get('container_name', 'N/A')}`")
                st_markdown(f"**Neo4j user:** `{config_data.get('neo4j_user', 'N/A')}`")
                if config_data.get("status") != "success":
                    st.warning("Some services may not be connected", icon=":material/warning:")
            else:
                st.error("Backend may be offline", icon=":material/error:")
        except Exception as e:
            logger.error(f"Error in system info: {e}")
            st.error(f"Error: {str(e)[:100]}", icon=":material/error:")

    st.subheader("Settings", help="User and workspace configuration")

    # --- User Management ---
    existing_users = fetch_all_users()
    user_options = existing_users + ["+ Create New User"]

    default_index = 0
    if "user_name" in st.session_state and st.session_state.user_name in existing_users:
        default_index = existing_users.index(st.session_state.user_name)

    selected_option = st.selectbox(
        "Active user",
        user_options,
        index=default_index,
        help="Select active user workspace",
    )

    if selected_option == "+ Create New User":
        new_name_input = st.text_input(
            "New username",
            placeholder="Enter username",
            key="new_user_input",
        ).strip()
        name = new_name_input if new_name_input else "test"
    else:
        name = selected_option

    # Store name in session state to detect changes
    if "user_name" not in st.session_state:
        st.session_state.user_name = name

    # Detect name change to reload chats
    if st.session_state.user_name != name:
        st.session_state.user_name = name
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        st.rerun()

    # User deletion with popover confirmation
    if selected_option != "+ Create New User" and name:
        with st.popover(
            "Delete user",
            icon=":material/delete:",
            help="Permanently delete this user and all data",
        ):
            st_markdown(f"**Delete user `{name}`?**")
            st.caption("This will permanently remove the user and all associated chats.")
            if st.button("Confirm delete", type="primary", key="confirm_delete_user"):
                delete_user_api(name)
                st.session_state.chats = {}
                st.session_state.active_chat_id = None
                st.rerun()

    # Load chats from backend if empty
    if not st.session_state.chats and st.session_state.get("user_name"):
        backend_chats = fetch_user_chats(st.session_state.user_name)
        for chat in backend_chats:
            s_id = chat.get("session_id")
            last_msg = chat.get("last_message", "New chat")
            title = (last_msg[:18] + "...") if last_msg else "New chat"

            if s_id:
                st.session_state.chats[s_id] = {
                    "title": title,
                    "messages": [],
                    "loaded": False,
                    "thoughts": [],
                }

    if (
        "active_chat_id" not in st.session_state
        or st.session_state.active_chat_id is None
    ):
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
        else:
            first_chat_id = str(uuid.uuid4())
            st.session_state.active_chat_id = first_chat_id
            st.session_state.chats[first_chat_id] = {
                "title": "New chat",
                "messages": [],
                "thoughts": [],
                "loaded": True,
            }

    # --- Chat Management ---
    st.subheader("Chats", help="Navigate and manage your chat sessions")

    with st.container(horizontal=True):
        if st.button("New chat", icon=":material/add:", width="stretch"):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = {
                "title": "New chat",
                "messages": [],
                "thoughts": [],
                "loaded": True,
            }
            st.session_state.active_chat_id = new_chat_id
            st.rerun()

        if st.button(
            "",
            icon=":material/refresh:",
            key="sidebar_refresh",
            help="Refresh chats from database",
        ):
            st.session_state.chats = {}
            st.rerun()

    # Scrollable container for chat session buttons
    chat_ids = list(st.session_state.chats.keys())
    chat_container = st.container(height=300, border=True)

    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.active_chat_id
        title_label = chat_data.get("title", "New chat")

        col_title, col_del = chat_container.columns([0.78, 0.22])

        with col_title:
            if st.button(
                title_label,
                key=f"chat_btn_{chat_id}",
                width="stretch",
                icon=":material/chat:",
                type="primary" if is_active else "secondary",
                disabled=is_active,
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

        with col_del:
            if st.button(
                "",
                icon=":material/delete:",
                key=f"delete_chat_{chat_id}",
                help="Delete chat",
            ):
                delete_chat_api(chat_id)
                del st.session_state.chats[chat_id]
                if st.session_state.active_chat_id == chat_id:
                    remaining_chats = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = (
                        remaining_chats[-1] if remaining_chats else None
                    )
                st.rerun()

    if st.button("Clear active chat", icon=":material/cleaning_services:", width="stretch"):
        active_chat = get_active_chat()
        if active_chat:
            active_chat["thoughts"] = []
            active_chat["messages"] = []
            st.rerun()

    st.caption("OPSEC © LOLLIMERD 2025")

active_chat = get_active_chat()

# --- Main Content Area ---
if active_chat is None:
    st.error("No active chat available", icon=":material/error:")
else:
    # --- Hero Header ---
    st.html(
        """
        <div style="text-align: center; padding: 20px 16px 8px;">
            <h1 style="font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px;">
                Lolly-RAG
            </h1>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
                <span style="background: rgba(124, 58, 237, 0.15); border: 1px solid rgba(124, 58, 237, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #E9D5FF; font-weight: 500;">
                    ⚡ Private RAG
                </span>
                <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #BFDBFE; font-weight: 500;">
                    🦙 Ollama
                </span>
                <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #D1FAE5; font-weight: 500;">
                    🧠 Qwen 3.5
                </span>
                <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #DDD6FE; font-weight: 500;">
                    🕸️ GraphRAG
                </span>
            </div>
            <p style="font-size: 0.95rem; color: #94A3B8; max-width: 680px; margin: 0 auto;">
                Ask questions to get real-time graph reasoning, structured tables, and visual Mermaid diagrams from your knowledge base.
            </p>
        </div>
        """
    )

    st.caption(f":material/person: Signed in as **{st.session_state.get('user_name', 'Guest')}**")

    # Quick Suggestion Pills on empty chats
    suggested_prompt = None
    if not active_chat["messages"]:
        SUGGESTIONS = {
            ":blue[:material/description:] What documents are available?": "What documents are currently ingested in the knowledge base?",
            ":orange Get to know the Assistant": "Tell me about yourself",
            ":green[:material/hub:] Explore knowledge graph": "Show me the structure of the knowledge graph and key entities with a mermaid diagram",
            ":violet[:material/summarize:] Summarize key concepts": "Summarize the key topics and concepts extracted from my ingested documents",
        }
        selected_pill = st.pills(
            "Suggested queries",
            options=list(SUGGESTIONS.keys()),
            label_visibility="collapsed",
            key="chat_suggestion_pills",
        )
        if selected_pill:
            suggested_prompt = SUGGESTIONS[selected_pill]

    # Display past messages
    for i, message in enumerate(active_chat["messages"]):
        role = message.get("role", "user")
        if role == "user":
            with st.chat_message(name="user", avatar=":material/person:"):
                st_markdown(message.get("content", ""), key=f"user_msg_{i}")
        else:
            with st.chat_message(name="assistant", avatar=":material/smart_toy:"):
                thought = message.get("thought")
                if thought and thought.strip():
                    with st.expander("Thought process", icon=":material/psychology:", expanded=False):
                        st.markdown(thought)
                st_markdown(message.get("content", ""), key=f"ai_msg_{i}")

    # Input handling
    input_prompt = st.chat_input("Ask your question...", submit_mode="disable")
    prompt = input_prompt or suggested_prompt

    if prompt:
        active_chat = get_active_chat()

        if active_chat:
            active_chat["messages"].append({"role": "user", "content": prompt})

            # Set a title for new chats based on the first message
            if active_chat["title"] in ("New chat", "New Chat") or active_chat["title"].startswith("Chat "):
                active_chat["title"] = prompt[:18] + ("..." if len(prompt) > 18 else "")

            with st.chat_message(name="user", avatar=":material/person:"):
                st_markdown(prompt, key=f"prompt_live_{len(active_chat['messages'])}")

            with st.chat_message(name="assistant", avatar=":material/smart_toy:"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()

                session_id = st.session_state.get("active_chat_id", "")
                thought_content = ""
                answer_content = ""
                tools_used = False
                has_error = False

                with status_placeholder.status("Thinking...", expanded=True) as status_box:
                    thought_stream_placeholder = st.empty()
                    try:
                        user_id = st.session_state.get("user_name", "test_user")
                        payload = {
                            "question": prompt,
                            "session_id": session_id,
                            "user_id": user_id,
                        }
                        with httpx.Client(timeout=120) as client:
                            with connect_sse(client, "POST", AGENT_URL, json=payload) as event_source:
                                for sse in event_source.iter_sse():
                                    if not sse.data or sse.data.strip() == "[DONE]":
                                        break

                                    try:
                                        data = json.loads(sse.data)
                                    except json.JSONDecodeError:
                                        continue

                                    msg_type = data.get("type")
                                    if msg_type == "done":
                                        break

                                    if msg_type == "status":
                                        tools_used = True
                                        message_text = data.get("message", "")
                                        status_state = data.get("status")
                                        status_box.update(label=message_text, state="running", expanded=True)
                                        if status_state == "running":
                                            st.info(message_text, icon=":material/sync:")
                                        elif status_state == "complete":
                                            st.success(message_text, icon=":material/check_circle:")

                                    elif msg_type == "token":
                                        chunk_content = data.get("content", "")
                                        chunk_thought = data.get("reasoning_content", "")

                                        if chunk_thought:
                                            thought_content += chunk_thought
                                            status_box.update(label="Thinking...", state="running", expanded=True)
                                            thought_stream_placeholder.markdown(format_stream_preview(thought_content))

                                        if chunk_content:
                                            if thought_content:
                                                thought_stream_placeholder.markdown(thought_content)
                                            status_box.update(
                                                label="Thought process",
                                                state="complete",
                                                expanded=False,
                                            )
                                            answer_content += chunk_content
                                            answer_placeholder.markdown(format_stream_preview(answer_content))

                                    elif msg_type == "error":
                                        has_error = True
                                        err_msg = data.get("content", "Unknown error")
                                        status_box.update(label="Error occurred", state="error", expanded=True)
                                        st.error(err_msg, icon=":material/error:")

                    except httpx.TimeoutException as e:
                        has_error = True
                        logger.error(f"Request timeout: {e}")
                        status_box.update(label="Request timeout", state="error", expanded=True)
                        st.error("The request timed out. Please try again later.", icon=":material/timer_off:")
                    except (requests.exceptions.RequestException, httpx.RequestError) as e:
                        has_error = True
                        logger.error(f"Connection error: {e}")
                        status_box.update(label="Connection error", state="error", expanded=True)
                        st.error(f"Could not connect to the API at {BACKEND_URL}. Is the backend running?", icon=":material/wifi_off:")
                    except Exception as e:
                        has_error = True
                        logger.error(f"Unexpected error: {e}")
                        status_box.update(label="Error occurred", state="error", expanded=True)
                        st.error(f"Unexpected error: {str(e)[:200]}", icon=":material/error:")

                    # Finalize status container state
                    if not has_error:
                        if thought_content.strip():
                            thought_stream_placeholder.markdown(thought_content)
                            status_box.update(label="Thought process", state="complete", expanded=False)
                        elif tools_used:
                            status_box.update(label="Actions complete", state="complete", expanded=False)

                # If no thoughts were generated and no tools were run, remove empty status container
                if not has_error and not thought_content.strip() and not tools_used:
                    status_placeholder.empty()

                # Final rendering and storage
                answer_placeholder.empty()
                if answer_content:
                    st_markdown(answer_content, key=f"answer_live_{session_id}_{len(active_chat['messages'])}")
                elif not has_error:
                    st.warning("No response content received", icon=":material/warning:")

                active_chat["messages"].append(
                    {
                        "role": "assistant",
                        "thought": thought_content,
                        "content": answer_content,
                    }
                )
                st.rerun()
