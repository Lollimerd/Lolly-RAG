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
from utils.doc_utils import SUPPORTED_TYPES, upload_file, upload_file_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

if "chats" not in st.session_state:
    st.session_state.chats = {}

def format_stream_preview(text: str) -> str:
    """Args: text: Markdown string."""
    preview = re.sub(r"```mermaid", "```text", text, flags=re.IGNORECASE)
    return (preview + "\n```") if preview.count("```") % 2 == 1 else preview + " ▌"

def get_active_chat():
    """Args: None."""
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id or chat_id not in st.session_state.chats:
        if not st.session_state.chats:
            return None
        chat_id = list(st.session_state.chats.keys())[-1]
        st.session_state.active_chat_id = chat_id
    chat = st.session_state.chats[chat_id]
    if not chat.get("loaded", False):
        chat["messages"] = fetch_chat_history(chat_id)
        chat["loaded"] = True
    return chat

def _ingest_attachments(attached_files, user_id: str, status_box) -> None:
    """Args: attached_files: Uploaded files, user_id: User identifier, status_box: Streamlit status widget."""
    for f in attached_files:
        status_box.update(label=f"Ingesting {f.name} into knowledge graph...", state="running", expanded=True)
        try:
            upload_file_stream(
                file_bytes=f.getvalue(),
                filename=f.name,
                user_id=user_id,
                description=f"Attached in chat by {user_id}",
                folder="Chat Uploads",
                force=True,
                progress_callback=lambda ev: status_box.update(
                    label=f"Ingesting {f.name}: {ev.get('message', '')}", state="running", expanded=True
                ) if ev.get("message") else None,
            )
            status_box.write(f":material/check_circle: Ingested `{f.name}`")
        except Exception as upload_err:
            logger.error("Failed to ingest attached file %s: %s", f.name, upload_err)
            status_box.write(f":material/error: Failed to ingest `{f.name}`: {upload_err}")
    status_box.update(label="Thinking...", state="running", expanded=True)

def _stream_agent_response(
    payload: dict, status_box, thought_placeholder, answer_placeholder
) -> tuple[str, str, bool, bool]:
    """Args: payload: Request body dict, status_box: Status widget, thought_placeholder: Thought widget, answer_placeholder: Answer widget."""
    thought_content, answer_content, tools_used, has_error = "", "", False, False
    try:
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
                        msg_text = data.get("message", "")
                        status_box.update(label=msg_text, state="running", expanded=True)
                        if data.get("status") == "running":
                            st.info(msg_text, icon=":material/sync:")
                        elif data.get("status") == "complete":
                            st.success(msg_text, icon=":material/check_circle:")
                    elif msg_type == "token":
                        chunk_content, chunk_thought = data.get("content", ""), data.get("reasoning_content", "")
                        if chunk_thought:
                            thought_content += chunk_thought
                            status_box.update(label="Thinking...", state="running", expanded=True)
                            thought_placeholder.markdown(format_stream_preview(thought_content))
                        if chunk_content:
                            if thought_content:
                                thought_placeholder.markdown(thought_content)
                            status_box.update(label="Thought process", state="complete", expanded=False)
                            answer_content += chunk_content
                            answer_placeholder.markdown(format_stream_preview(answer_content))
                    elif msg_type == "error":
                        has_error = True
                        status_box.update(label="Error occurred", state="error", expanded=True)
                        st.error(data.get("content", "Unknown error"), icon=":material/error:")
    except httpx.TimeoutException as exc:
        has_error = True
        logger.error("Request timeout: %s", exc)
        status_box.update(label="Request timeout", state="error", expanded=True)
        st.error("The request timed out. Please try again later.", icon=":material/timer_off:")
    except (requests.exceptions.RequestException, httpx.RequestError) as exc:
        has_error = True
        logger.error("Connection error: %s", exc)
        status_box.update(label="Connection error", state="error", expanded=True)
        st.error(f"Could not connect to the API at {BACKEND_URL}. Is the backend running?", icon=":material/wifi_off:")
    except Exception as exc:
        has_error = True
        logger.error("Unexpected error: %s", exc)
        status_box.update(label="Error occurred", state="error", expanded=True)
        st.error(f"Unexpected error: {str(exc)[:200]}", icon=":material/error:")
    return thought_content, answer_content, tools_used, has_error

# --- Sidebar ---
with st.sidebar:
    try:
        display_container_name()
    except Exception as e:
        logger.error("Error displaying container info: %s", e)
        st.warning("Could not connect to backend for system info", icon=":material/warning:")

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
            logger.error("Error in system info: %s", e)
            st.error(f"Error: {str(e)[:100]}", icon=":material/error:")

    st.subheader("Settings", help="User and workspace configuration")
    existing_users = fetch_all_users()
    user_options = existing_users + ["+ Create New User"]
    default_idx = existing_users.index(st.session_state.user_name) if st.session_state.get("user_name") in existing_users else 0
    selected_option = st.selectbox("Active user", user_options, index=default_idx, help="Select active user workspace")

    if selected_option == "+ Create New User":
        new_input = st.text_input("New username", placeholder="Enter username", key="new_user_input").strip()
        name = new_input or "test"
    else:
        name = selected_option

    if "user_name" not in st.session_state or st.session_state.user_name != name:
        st.session_state.user_name = name
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        st.rerun()

    if selected_option != "+ Create New User" and name:
        with st.popover("Delete user", icon=":material/delete:", help="Permanently delete this user and all data"):
            st_markdown(f"**Delete user `{name}`?**")
            st.caption("This will permanently remove the user and all associated chats.")
            if st.button("Confirm delete", type="primary", key="confirm_delete_user"):
                delete_user_api(name)
                st.session_state.chats = {}
                st.session_state.active_chat_id = None
                st.rerun()

    if not st.session_state.chats and st.session_state.get("user_name"):
        for chat in fetch_user_chats(st.session_state.user_name):
            s_id = chat.get("session_id")
            if s_id:
                last_msg = chat.get("last_message", "New chat") or "New chat"
                st.session_state.chats[s_id] = {
                    "title": (last_msg[:18] + "...") if len(last_msg) > 18 else last_msg,
                    "messages": [], "loaded": False, "thoughts": [],
                }

    if not st.session_state.get("active_chat_id") or st.session_state.active_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
        else:
            first_chat_id = str(uuid.uuid4())
            st.session_state.active_chat_id = first_chat_id
            st.session_state.chats[first_chat_id] = {"title": "New chat", "messages": [], "thoughts": [], "loaded": True}

    st.subheader("Chats", help="Navigate and manage your chat sessions")
    with st.container(horizontal=True):
        if st.button("New chat", icon=":material/add:", width="stretch"):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = {"title": "New chat", "messages": [], "thoughts": [], "loaded": True}
            st.session_state.active_chat_id = new_chat_id
            st.rerun()
        if st.button("", icon=":material/refresh:", key="sidebar_refresh", help="Refresh chats from database"):
            st.session_state.chats = {}
            st.rerun()

    chat_ids = list(st.session_state.chats.keys())
    chat_container = st.container(height=300, border=True)
    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        is_active = (chat_id == st.session_state.active_chat_id)
        col_title, col_del = chat_container.columns([0.78, 0.22])
        with col_title:
            if st.button(
                chat_data.get("title", "New chat"), key=f"chat_btn_{chat_id}", width="stretch",
                icon=":material/chat:", type="primary" if is_active else "secondary", disabled=is_active,
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()
        with col_del:
            if st.button("", icon=":material/delete:", key=f"delete_chat_{chat_id}", help="Delete chat"):
                delete_chat_api(chat_id)
                del st.session_state.chats[chat_id]
                if st.session_state.active_chat_id == chat_id:
                    remaining = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = remaining[-1] if remaining else None
                st.rerun()

    if st.button("Clear active chat", icon=":material/cleaning_services:", width="stretch"):
        active_chat = get_active_chat()
        if active_chat:
            active_chat["thoughts"], active_chat["messages"] = [], []
            st.rerun()

    st.caption("OPSEC © LOLLIMERD 2025")

active_chat = get_active_chat()

if active_chat is None:
    st.error("No active chat available", icon=":material/error:")
else:
    st.html(
        """
        <div style="text-align: center; padding: 20px 16px 8px;">
            <h1 style="font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px;">
                Lolly-RAG
            </h1>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
                <span style="background: rgba(124, 58, 237, 0.15); border: 1px solid rgba(124, 58, 237, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #E9D5FF; font-weight: 500;">⚡ Private RAG</span>
                <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #BFDBFE; font-weight: 500;">🦙 Ollama</span>
                <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #D1FAE5; font-weight: 500;">🧠 Qwen 3.5</span>
                <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.4); padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; color: #DDD6FE; font-weight: 500;">🕸️ GraphRAG</span>
            </div>
            <p style="font-size: 0.95rem; color: #94A3B8; max-width: 680px; margin: 0 auto;">
                Ask questions to get real-time graph reasoning, structured tables, and visual Mermaid diagrams from your knowledge base.
            </p>
        </div>
        """
    )
    st.caption(f":material/person: Signed in as **{st.session_state.get('user_name', 'Guest')}**")

    suggested_prompt = None
    if not active_chat["messages"]:
        SUGGESTIONS = {
            ":orange [:material/smart_toy:] Get to know the Assistant": "Tell me about yourself",
            ":blue[:material/description:] What documents are available?": "What documents are currently ingested in the knowledge base?",
            ":green[:material/hub:] Explore knowledge graph": "Show me the structure of the knowledge graph and key entities with a mermaid diagram",
            ":violet[:material/summarize:] Summarize key concepts": "Summarize the key topics and concepts extracted from my ingested documents",
        }
        selected_pill = st.pills("Suggested queries", options=list(SUGGESTIONS.keys()), label_visibility="collapsed", key="chat_suggestion_pills")
        if selected_pill:
            suggested_prompt = SUGGESTIONS[selected_pill]

    for i, message in enumerate(active_chat["messages"]):
        if message.get("role", "user") == "user":
            with st.chat_message(name="user", avatar=":material/person:"):
                files_str = message.get("files", "")
                if files_str:
                    st.caption(" • ".join(f":material/attachment: `{fn.strip()}`" for fn in files_str.split(",") if fn.strip()))
                st_markdown(message.get("content", ""), key=f"user_msg_{i}")
        else:
            with st.chat_message(name="assistant", avatar=":material/smart_toy:"):
                thought = message.get("thought")
                if thought and thought.strip():
                    with st.expander("Thought process", icon=":material/psychology:", expanded=False):
                        st.markdown(thought)
                st_markdown(message.get("content", ""), key=f"ai_msg_{i}")

    chat_val = st.chat_input("Ask a question or attach files...", accept_file="multiple", file_type=SUPPORTED_TYPES, submit_mode="disable")
    prompt = chat_val.text.strip() if chat_val else (suggested_prompt or "")
    attached_files = getattr(chat_val, "files", []) or [] if chat_val else []

    if not prompt and attached_files:
        prompt = f"Please analyze and summarize the attached document(s): {', '.join(f.name for f in attached_files)}"

    if prompt or attached_files:
        active_chat = get_active_chat()
        if active_chat:
            file_names = [f.name for f in attached_files]
            user_msg = {"role": "user", "content": prompt}
            if file_names:
                user_msg["files"] = ", ".join(file_names)
            active_chat["messages"].append(user_msg)

            if active_chat["title"] in ("New chat", "New Chat") or active_chat["title"].startswith("Chat "):
                active_chat["title"] = prompt[:18] + ("..." if len(prompt) > 18 else "")

            with st.chat_message(name="user", avatar=":material/person:"):
                if file_names:
                    st.caption(" • ".join(f":material/attachment: `{fn}`" for fn in file_names))
                st_markdown(prompt, key=f"prompt_live_{len(active_chat['messages'])}")

            with st.chat_message(name="assistant", avatar=":material/smart_toy:"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                session_id = st.session_state.get("active_chat_id", "")
                user_id = st.session_state.get("user_name", "test_user")

                with status_placeholder.status("Processing...", expanded=True) as status_box:
                    if attached_files:
                        _ingest_attachments(attached_files, user_id, status_box)

                    thought_stream_placeholder = st.empty()
                    payload = {"question": prompt, "session_id": session_id, "user_id": user_id, "attached_files": file_names or None}
                    thought_content, answer_content, tools_used, has_error = _stream_agent_response(
                        payload, status_box, thought_stream_placeholder, answer_placeholder
                    )

                    if not has_error:
                        if thought_content.strip():
                            thought_stream_placeholder.markdown(thought_content)
                            status_box.update(label="Thought process", state="complete", expanded=False)
                        elif tools_used:
                            status_box.update(label="Actions complete", state="complete", expanded=False)

                if not has_error and not thought_content.strip() and not tools_used:
                    status_placeholder.empty()

                answer_placeholder.empty()
                if answer_content:
                    st_markdown(answer_content, key=f"answer_live_{session_id}_{len(active_chat['messages'])}")
                elif not has_error:
                    st.warning("No response content received", icon=":material/warning:")

                active_chat["messages"].append({"role": "assistant", "thought": thought_content, "content": answer_content})
                st.rerun()
