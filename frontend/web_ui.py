import logging
import streamlit as st

from utils.ui_utils import (
    delete_user_api,
    display_container_name,
    fetch_all_users,
    get_system_config,
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
        "About": "# Lolly-RAG: Private Agentic GraphRAG Knowledge Base",
    },
)

# --- Global User & Session State Initialization ---
if "user_name" not in st.session_state:
    st.session_state.user_name = "test_user"

# --- Sidebar Global Header: System & User Configuration ---
with st.sidebar:
    try:
        display_container_name()
    except Exception as e:
        logger.warning(f"Container info display error: {e}")

    with st.expander("System info & database", expanded=False):
        config_data = get_system_config()
        if config_data and isinstance(config_data, dict):
            st.markdown(f"**Model:** `{config_data.get('ollama_model', 'N/A')}`")
            st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
            st.markdown(f"**Database Container:** `{config_data.get('container_name', 'N/A')}`")
            st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
        else:
            st.caption("Backend offline or unreachable.")

    st.subheader("Active profile", help="Switch between user workspaces")
    existing_users = fetch_all_users()
    if not existing_users:
        existing_users = ["test_user"]

    user_options = existing_users + ["+ Create new user"]
    default_idx = existing_users.index(st.session_state.user_name) if st.session_state.user_name in existing_users else 0

    selected_user = st.selectbox(
        "Select user profile",
        options=user_options,
        index=default_idx,
        label_visibility="collapsed",
    )

    if selected_user == "+ Create new user":
        new_username = st.text_input("Enter new username", placeholder="e.g. alice, bob", key="new_user_input").strip()
        if new_username:
            st.session_state.user_name = new_username
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            st.rerun()
    else:
        if st.session_state.user_name != selected_user:
            st.session_state.user_name = selected_user
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            st.rerun()

    if selected_user != "+ Create new user" and st.session_state.user_name:
        with st.popover("Delete user profile", icon=":material/person_remove:"):
            st.caption(f"Permanently delete user '{st.session_state.user_name}' and all associated chats?")
            if st.button("Confirm delete user", key="conf_del_usr_btn", type="primary"):
                delete_user_api(st.session_state.user_name)
                st.session_state.user_name = "test_user"
                st.session_state.chats = {}
                st.session_state.active_chat_id = None
                st.rerun()

    st.divider()

# --- Modern Multi-Page Routing with st.navigation ---
pages = {
    "AI Assistant": [
        st.Page("app_pages/chat.py", title="Chat & Assistant", icon=":material/smart_toy:", default=True),
    ],
    "Knowledge Base": [
        st.Page("app_pages/doc_ingestion.py", title="Document Ingestion", icon=":material/folder_open:"),
        st.Page("app_pages/dashboard.py", title="StackExchange Data", icon=":material/analytics:"),
    ],
    "Graph Visualization": [
        st.Page("app_pages/neo4j_explorer.py", title="Neo4j Explorer", icon=":material/hub:"),
    ],
}

pg = st.navigation(pages)
pg.run()
