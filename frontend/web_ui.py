# app.py
import json
import requests
import uuid
import httpx
import logging
from httpx_sse import connect_sse
import streamlit as st

import streamlit.components.v1 as components

# from streamlit_timeline import timeline
from utils.ui_utils import (
    render_message_with_mermaid,
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
    page_title="Lolly-Rag",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.google.com",
        "Report a bug": "https://www.google.com",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# --- Initialize Session State AND Sync with Backend ---
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Initialize empty first


# Helper function to get the active chat object
def get_active_chat():
    """Get the active chat data, creating it if necessary."""
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id or chat_id not in st.session_state.chats:
        # Fallback to latest chat if ID is missing or invalid
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


# --- sidebar init ---
with st.sidebar:
    try:
        display_container_name()
    except Exception as e:
        logger.error(f"Error displaying container info: {e}")
        st.warning("Could not connect to backend for system info")

    # --- System Info ---
    with st.expander("System Info & DB Details", expanded=False):
        try:
            config_data = get_system_config()
            if config_data and isinstance(config_data, dict):
                st.markdown(
                    f"**Ollama Model:** `{config_data.get('ollama_model', 'N/A')}`"
                )
                st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
                st.markdown(
                    f"**DB connected:** `{config_data.get('container_name', 'N/A')}`"
                )
                st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
                if config_data.get("status") != "success":
                    st.warning("⚠️ Some services may not be connected")
            else:
                st.error("Could not retrieve system info - backend may be offline")
        except Exception as e:
            logger.error(f"Error in system info: {e}")
            st.error(f"Error: {str(e)[:100]}")

    st.sidebar.title("⚙️ Settings", help="config settings here")

    # --- General Settings ---
    # Fetch existing users
    existing_users = fetch_all_users()

    # Add option for new user
    user_options = existing_users + ["+ Create New User"]

    # Determine default index
    default_index = 0
    if "user_name" in st.session_state and st.session_state.user_name in existing_users:
        default_index = existing_users.index(st.session_state.user_name)

    selected_option = st.sidebar.selectbox(
        "Select User", user_options, index=default_index
    )

    if selected_option == "+ Create New User":
        name = st.sidebar.text_input("Enter New Username", key="new_user_input")
        if not name:
            name = "test"  # default fallback
    else:
        name = selected_option

    # Store name in session state to detect changes
    if "user_name" not in st.session_state:
        st.session_state.user_name = name

    # Detect name change to reload chats
    if st.session_state.user_name != name:
        st.session_state.user_name = name
        # Clear current chats to force reload logic below
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        # We need to rerun to let the logic below fetch new chats
        st.rerun()

    # --- User Deletion UI ---
    if selected_option != "+ Create New User" and name:
        if st.sidebar.button(
            "🗑️ Delete User", help="Permanently delete this user and all data"
        ):
            delete_user_api(name)
            # Clear state and refresh
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            st.rerun()

    st.sidebar.markdown("---")

    # Logic to load chats if empty (initial load or name change)
    if not st.session_state.chats and st.session_state.get("user_name"):
        backend_chats = fetch_user_chats(st.session_state.user_name)
        for chat in backend_chats:
            s_id = chat.get("session_id")
            last_msg = chat.get("last_message", "New Chat")
            title = (last_msg[:15] + "...") if last_msg else "New Chat"

            if s_id:  # Only add if we have a valid session_id
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
        # If we have chats from DB, pick the latest one (which is on top of sidebar)
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
        else:
            # Create a default first chat if DB is empty
            first_chat_id = str(uuid.uuid4())
            st.session_state.active_chat_id = first_chat_id
            st.session_state.chats[first_chat_id] = {
                "title": "New Chat",
                "messages": [],
                "thoughts": [],
                "loaded": True,
            }

    # --- Chat Management UI ---
    st.subheader("Chats", help="navigate your chats here")

    col_new, col_refresh = st.columns([0.8, 0.2])
    with col_new:
        if st.button("+ New Chat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = {
                "title": "New Chat",
                "messages": [],
                "thoughts": [],
                "loaded": True,
            }
            st.session_state.active_chat_id = new_chat_id
            st.rerun()

    with col_refresh:
        if st.button(
            "🔄", key="sidebar_refresh", help="Refresh Chats & Re-render Diagrams"
        ):
            # Clear local state to force reload from backend
            st.session_state.chats = {}
            st.rerun()

    # --- Display chats as individual buttons ---
    # Create a reversed list of chat IDs to display the newest chats first
    chat_ids = list(st.session_state.chats.keys())

    # Create a container with fixed height inside the sidebar (scrollable pane)
    chat_container = st.container(height=300, border=True)

    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]

        # Use columns to place chat button and delete button on the same line
        col1, col2 = chat_container.columns([0.3, 0.7])

        with col1:
            # Button to delete the chat
            if st.button("🗑️", key=f"delete_chat_{chat_id}", use_container_width=True):
                # Call API to delete from DB
                delete_chat_api(chat_id)

                # Remove the chat from the dictionary
                del st.session_state.chats[chat_id]

                # If the deleted chat was the active one, select a new active chat (latest remaining)
                if st.session_state.active_chat_id == chat_id:
                    remaining_chats = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = (
                        remaining_chats[-1] if remaining_chats else None
                    )
                st.rerun()

        with col2:
            # Button to select the chat
            # Handle possible missing keys safely
            title_label = chat_data.get("title", "New Chat")
            if st.button(
                title_label,
                key=f"chat_button_{chat_id}",
                use_container_width=True,
                disabled=(chat_id == st.session_state.active_chat_id),
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    # --- Clear history for the ACTIVE chat ---
    if st.button("Clear Active Chat History", use_container_width=True):
        active_chat = get_active_chat()
        if active_chat:
            active_chat["thoughts"] = []
            active_chat["messages"] = []
            st.rerun()

    st.write("OPSEC ©LOLLIMERD 2025")

    # # --- Debug Mode ---
    # with st.expander("🛠️ Debug Info"):
    #     st.write("Active Chat ID:", st.session_state.active_chat_id)
    #     if (
    #         st.session_state.active_chat_id
    #         and st.session_state.active_chat_id in st.session_state.chats
    #     ):
    #         chat = st.session_state.chats[st.session_state.active_chat_id]
    #         st.write("Message Count:", len(chat.get("messages", [])))
    #         st.write("Last 3 Messages (Raw):")
    #         st.json(chat.get("messages", [])[-3:])

active_chat = get_active_chat()

# --- Main Content Area ---
if active_chat is None:
    st.error("No active chat available")
else:
    # --- UI Elements ---
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
            <h1 style="font-size: 3.5rem; background: linear-gradient(to right, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
                Lolly-Rag
            </h1>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(124, 58, 237, 0.2); border: 1px solid rgba(124, 58, 237, 0.5); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #E9D5FF;">
                    ★ PRIVATE-RAG
                </span>
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.5); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #BFDBFE;">
                    ★ Ollama
                </span>
                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #D1FAE5;">
                     Verified: MOE -> Qwen3.5
                </span>
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.5); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #BFDBFE;">
                    ☍ GraphRAG
                </span>
            </div>
            <p style="font-size: 1.1rem; color: #94A3B8; max-width: 800px; margin: 0 auto;">
                Ask a question to get a real-time analysis from the knowledge graph. 
                Generate tables, graphs, and inferential analysis from your data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(body=f"Welcome back, {st.session_state.get('user_name', 'Guest')}")

    # When displaying past messages from the ACTIVE chat:
    for i, message in enumerate(active_chat["messages"]):
        role = message.get("role", "user")
        if role == "user":
            author_name = st.session_state.get("user_name", "User")
        elif role == "assistant":
            author_name = "Assistant"
        else:
            author_name = role.capitalize() if isinstance(role, str) else "Author"

        with st.chat_message(name=author_name):
            if role == "assistant":
                if message.get("thought"):
                    with st.expander("Show Agent Thoughts"):
                        render_message_with_mermaid(
                            message["thought"], key_suffix=f"{i}-thought"
                        )
                render_message_with_mermaid(
                    message.get("content", ""), key_suffix=f"{i}-content"
                )
            else:
                st.markdown(message.get("content", ""))

    # --- Main Interaction Logic (operates on the active chat) ---
    if prompt := st.chat_input("Ask your question..."):
        # Get the active chat
        active_chat = get_active_chat()

        if active_chat:
            # Append to the active chat's message list
            active_chat["messages"].append({"role": "user", "content": prompt})

            # Set a title for new chats based on the first message
            if active_chat["title"] == "New Chat" or active_chat["title"].startswith(
                "Chat "
            ):
                active_chat["title"] = prompt[:15] + "..."  # Truncate for display

            with st.chat_message(name=st.session_state.get("user_name", "User")):
                st.markdown(prompt)

            with st.chat_message(name="Assistant"):
                # 1. Define Visual Layout Order via Placeholders
                status_container_loc = st.empty()
                thought_container_loc = st.empty()
                answer_container_loc = st.empty()

                # 2. Setup Thoughts Container (so we have a target to write to)
                with thought_container_loc.container():
                    thought_container = st.expander(
                        "Show Agent Thoughts", expanded=True
                    )
                    with thought_container:
                        thought_placeholder = st.empty()

                # 3. Setup Answer Placeholder target
                with answer_container_loc.container():
                    answer_placeholder = st.empty()

                thought_content = ""
                answer_content = ""

                # 4. Run Logic inside Status Container (Visual position #1)
                with status_container_loc.container():
                    with st.status(
                        "🚀 Initializing Agent...", expanded=True
                    ) as status_box:
                        try:
                            # prepare payload
                            session_id = st.session_state.active_chat_id
                            user_id = st.session_state.get("user_name", "test_user")
                            payload = {
                                "question": prompt,
                                "session_id": session_id,
                                "user_id": user_id,
                            }
                            timeout = httpx.Timeout(60, read=60)
                            with httpx.Client(timeout=timeout) as client:
                                with connect_sse(
                                    client,
                                    "POST",
                                    AGENT_URL,
                                    json=payload,
                                ) as event_source:
                                    for sse in event_source.iter_sse():
                                        if sse.data:
                                            try:
                                                data = json.loads(sse.data)
                                                msg_type = data.get("type")

                                                # --- Handle Status Events ---
                                                if msg_type == "status":
                                                    stage = data.get("stage")
                                                    message = data.get("message", "")
                                                    status_state = data.get("status")

                                                    # Update the container label to show current activity
                                                    status_box.update(
                                                        label=message, state="running", expanded=True
                                                    )

                                                    if status_state == "running":
                                                        st.info(message, icon="🔄")
                                                    elif status_state == "complete":
                                                        st.success(message, icon="✅")

                                                # --- Handle Token Events ---
                                                elif msg_type == "token":
                                                    # Collapse status box once generation starts
                                                    status_box.update(
                                                        label="✅ Analysis Complete. Generating Response...",
                                                        state="complete",
                                                        expanded=False,
                                                    )

                                                    chunk_content = data.get(
                                                        "content", ""
                                                    )
                                                    chunk_thought = data.get(
                                                        "reasoning_content", ""
                                                    )

                                                    answer_content += chunk_content
                                                    thought_content += chunk_thought

                                                    # Render Answer
                                                    if answer_content:
                                                        answer_placeholder.markdown(
                                                            answer_content + "▌"
                                                        )

                                                    # Render Thoughts
                                                    if thought_content:
                                                        thought_placeholder.markdown(
                                                            thought_content + "▌"
                                                        )

                                                # --- Handle Errors ---
                                                elif msg_type == "error":
                                                    status_box.update(
                                                        label="❌ Error Occurred",
                                                        state="error",
                                                        expanded=True,
                                                    )
                                                    st.error(
                                                        data.get(
                                                            "content", "Unknown error"
                                                        )
                                                    )

                                            except json.JSONDecodeError as e:
                                                logger.error(
                                                    f"Error decoding JSON: {sse.data}, \n{e}"
                                                )
                                                continue

                            # --- Final Processing and Rendering ---
                            # Remove type cursors and render final markdown/mermaid
                            answer_placeholder.empty()
                            thought_placeholder.empty()

                            with thought_container:
                                if thought_content:
                                    render_message_with_mermaid(
                                        thought_content, key_suffix="stream-thought"
                                    )
                                else:
                                    st.info("No agent thoughts captured")

                            if answer_content:
                                render_message_with_mermaid(
                                    answer_content, key_suffix="stream-content"
                                )
                            else:
                                st.warning("No response content received")

                            # Append the final response to the active chat's message list
                            active_chat["messages"].append(
                                {
                                    "role": "assistant",
                                    "thought": thought_content,
                                    "content": answer_content,
                                }
                            )
                            st.rerun()  # Rerun to update the chat list in the sidebar if the title changed

                        except httpx.TimeoutException as e:
                            logger.error(f"Request timeout: {e}")
                            st.error(
                                "The request timed out. The server is taking too long to respond. Please try again later."
                            )
                        except requests.exceptions.ConnectionError as e:
                            logger.error(f"Connection error: {e}")
                            st.error(
                                f"Could not connect to the API at {BACKEND_URL}. Is the backend running?"
                            )
                        except requests.exceptions.RequestException as e:
                            logger.error(f"Request exception: {e}")
                            st.error(f"API Error: {str(e)[:200]}")
                        except Exception as e:
                            logger.error(f"Unexpected error: {e}")
                            st.error(f"Unexpected error: {str(e)[:200]}")

    # --- Auto-Scroll to Bottom ---
    st.iframe(
        """
        <script>
            var scrollingElement = (document.scrollingElement || document.body);
            scrollingElement.scrollTop = scrollingElement.scrollHeight;
        </script>
        """
    )
