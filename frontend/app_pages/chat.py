import json
import logging
import uuid
import httpx
from httpx_sse import connect_sse
import streamlit as st
from streamlit_markdown import st_markdown
from utils.mermaid import render_message_with_mermaid
from utils.ui_utils import (
    AGENT_URL,
    BACKEND_URL,
    delete_chat_api,
    fetch_chat_history,
    fetch_user_chats,
)

logger = logging.getLogger(__name__)

# --- Initialize Chat Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = {}


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

    if not chat.get("loaded", False):
        msgs = fetch_chat_history(chat_id)
        chat["messages"] = msgs
        chat["loaded"] = True

    return chat


# Load chats from backend if not yet loaded
current_user = st.session_state.get("user_name", "test_user")
if not st.session_state.chats and current_user:
    backend_chats = fetch_user_chats(current_user)
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

if "active_chat_id" not in st.session_state or st.session_state.active_chat_id is None:
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

# --- Sidebar: Session & Conversation Manager ---
with st.sidebar:
    st.subheader("Conversations", help="Navigate and manage your active conversations")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        if st.button("New chat", icon=":material/add:", type="primary", width="stretch"):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {
                "title": "New chat",
                "messages": [],
                "thoughts": [],
                "loaded": True,
            }
            st.session_state.active_chat_id = new_id
            st.rerun()
    with c2:
        if st.button("Refresh", icon=":material/refresh:", help="Reload chat list from database", width="stretch"):
            st.session_state.chats = {}
            st.rerun()

    chat_ids = list(st.session_state.chats.keys())
    chat_container = st.container(height=280, border=True)

    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        title_label = chat_data.get("title", "New chat")
        is_active = (chat_id == st.session_state.active_chat_id)

        c1, c2 = chat_container.columns([0.7, 0.3])
        with c1:
            if st.button(
                title_label,
                key=f"select_chat_{chat_id}",
                disabled=is_active,
                icon=":material/chat_bubble_outline:" if not is_active else ":material/chat:",
                width="stretch"
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

        with c2:
            with st.popover("", icon=":material/delete:", help="Delete conversation", width="stretch"):
                st.caption(f"Delete '{title_label}'?")
                if st.button("Confirm delete", key=f"conf_del_{chat_id}", type="primary", width="stretch"):
                    delete_chat_api(chat_id)
                    del st.session_state.chats[chat_id]
                    if st.session_state.active_chat_id == chat_id:
                        rem = list(st.session_state.chats.keys())
                        st.session_state.active_chat_id = rem[-1] if rem else None
                    st.rerun()

    # Clear active conversation history
    with st.popover("Clear conversation history", icon=":material/mop:", width="stretch"):
        st.caption("Clear all messages in the active chat?")
        if st.button("Confirm clear", key="confirm_clear_history", type="primary", width="stretch"):
            active_c = get_active_chat()
            if active_c:
                active_c["thoughts"] = []
                active_c["messages"] = []
                st.rerun()

active_chat = get_active_chat()

# --- Main Chat Area ---
if active_chat is None:
    st.error("No active chat session available. Please create a new chat in the sidebar.")
else:
    # Header Banner Card
    with st.container(border=True):
        st.markdown("### 🧠 Lolly-RAG Intelligent Technical Assistant")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.badge("Hybrid GraphRAG", icon=":material/hub:")
        with col_b2:
            st.badge("Dense Vector + Fulltext", icon=":material/search:")
        with col_b3:
            st.badge("StackExchange Knowledge", icon=":material/code:")
        with col_b4:
            st.badge("Mermaid Architecture Visualizer", icon=":material/schema:")

        st.caption(
            "Ask technical questions across your uploaded documentation and developer knowledge graph. "
            "Get grounded answers with verifiable citations, source chunks, code examples, and architecture diagrams."
        )

    # Render Historical Messages
    for i, message in enumerate(active_chat["messages"]):
        role = message.get("role", "user")
        if role == "user":
            author_name = st.session_state.get("user_name", "User")
        else:
            author_name = "Assistant"

        with st.chat_message(name=author_name, avatar=":material/person:" if role == "user" else ":material/smart_toy:"):
            if role == "assistant":
                if message.get("thought"):
                    with st.expander("Agent reasoning & thought process", expanded=False):
                        render_message_with_mermaid(message["thought"], key_suffix=f"{i}-thought")
                render_message_with_mermaid(message.get("content", ""), key_suffix=f"{i}-content")
            else:
                st_markdown(message.get("content", ""), key=f"user_msg_{i}")

    # Chat Input & Streaming Execution
    if prompt := st.chat_input("Ask a technical question or query your documents..."):
        active_chat = get_active_chat()
        if active_chat:
            active_chat["messages"].append({"role": "user", "content": prompt})

            if active_chat["title"] in ("New chat", "New Chat") or active_chat["title"].startswith("Chat "):
                active_chat["title"] = prompt[:18] + "..."

            with st.chat_message(name=st.session_state.get("user_name", "User"), avatar=":material/person:"):
                st_markdown(prompt, key="current_user_prompt")

            with st.chat_message(name="Assistant", avatar=":material/smart_toy:"):
                status_container_loc = st.empty()
                thought_container_loc = st.empty()
                answer_container_loc = st.empty()

                with thought_container_loc.container():
                    thought_container = st.expander("Agent reasoning & thought process", expanded=True)
                    with thought_container:
                        thought_placeholder = st.empty()

                with answer_container_loc.container():
                    answer_placeholder = st.empty()

                thought_content = ""
                answer_content = ""

                with status_container_loc.container():
                    with st.status("Initializing retrieval agent...", expanded=True) as status_box:
                        try:
                            session_id = st.session_state.active_chat_id
                            user_id = st.session_state.get("user_name", "test_user")
                            payload = {
                                "question": prompt,
                                "session_id": session_id,
                                "user_id": user_id,
                            }
                            timeout = httpx.Timeout(60, read=120)
                            with httpx.Client(timeout=timeout) as client:
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
                                            message = data.get("message", "")
                                            status_state = data.get("status")
                                            status_box.update(label=message, state="running", expanded=True)
                                            if status_state == "running":
                                                st.info(message, icon=":material/sync:")
                                            elif status_state == "complete":
                                                st.success(message, icon=":material/check_circle:")

                                        elif msg_type == "token":
                                            status_box.update(
                                                label="Synthesizing response...",
                                                state="complete",
                                                expanded=False,
                                            )
                                            chunk_content = data.get("content", "")
                                            chunk_thought = data.get("reasoning_content", "")

                                            if chunk_content:
                                                answer_content += chunk_content
                                                answer_placeholder.markdown(answer_content + "▌")

                                            if chunk_thought:
                                                thought_content += chunk_thought
                                                thought_placeholder.markdown(thought_content + "▌")

                                        elif msg_type == "error":
                                            err_msg = data.get("content", "Unknown error")
                                            status_box.update(label="Error occurred", state="error", expanded=True)
                                            st.error(err_msg, icon=":material/error:")
                                            
                                        elif msg_type == "final_response":
                                            answer_content = data.get("content", answer_content)
                                            thought_content = data.get("reasoning_content", thought_content)

                        except httpx.TimeoutException as e:
                            logger.error(f"Request timeout: {e}")
                            status_box.update(label="Request timed out", state="error", expanded=True)
                            st.error("The request timed out. Please try again later.", icon=":material/schedule:")
                        except (httpx.RequestError, Exception) as e:
                            logger.error(f"Connection error: {e}")
                            status_box.update(label="Connection error", state="error", expanded=True)
                            st.error(f"Could not connect to the backend API ({BACKEND_URL}). Is the backend running?", icon=":material/link_off:")

                answer_placeholder.empty()
                thought_placeholder.empty()

                with thought_container:
                    if thought_content:
                        render_message_with_mermaid(thought_content, key_suffix="stream-thought")
                    else:
                        st.caption("No internal reasoning emitted.")

                if answer_content:
                    render_message_with_mermaid(answer_content, key_suffix="stream-content")
                else:
                    st.warning("No response content received.")

                active_chat["messages"].append(
                    {
                        "role": "assistant",
                        "thought": thought_content,
                        "content": answer_content,
                    }
                )
                st.rerun()
