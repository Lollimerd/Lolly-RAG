import gc
import os
import time
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit.logger import get_logger

from utils.ui_utils import (
    BACKEND_URL,
    delete_import_log_api,
    display_container_name,
    get_database_summary,
    get_import_history,
    update_import_log_api,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="Dashboard & Importer — Lolly-RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.google.com",
        "Report a bug": "https://www.google.com",
        "About": "# Lolly-RAG StackExchange Dashboard and Data Importer",
    },
)

# ---------------------------------------------------------------------------
# StackExchange Loader Logic
# ---------------------------------------------------------------------------
so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"


def load_so_data(tag: str, page: int, site: str) -> dict:
    """
    Load Stack Overflow data and handle potential errors gracefully.
    Returns a dictionary indicating the result.
    """
    try:
        api_key = os.getenv("STACKEXCHANGE_API_KEY")
        key_param = f"&key={api_key}" if api_key else ""
        site = "stackoverflow"
        parameters = f"""?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}&site={site}&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb{key_param}"""

        # Retry logic for network flakiness
        max_retries = 3
        data = None
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = requests.get(so_api_base_url + parameters, stream=False)
                response.raise_for_status()
                data = response.json()
                break  # Success
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt  # 1s, 2s, 4s...
                    time.sleep(sleep_time)
                    continue
                else:
                    raise last_exception or e

        if not data:
            raise last_exception or Exception("Failed to retrieve data after retries")

        if "items" in data and data["items"]:
            # Handle API backoff requests
            if "backoff" in data:
                time.sleep(data["backoff"])
            elif "error_name" in data:
                backoff_time = min(300, 2 ** (page % 8))  # Max 300 seconds
                time.sleep(backoff_time)
            insert_so_data(data)
            return {
                "status": "success",
                "tag": tag,
                "page": page,
                "count": len(data["items"]),
            }
        else:
            return {"status": "empty", "tag": tag, "page": page}

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "tag": tag,
            "page": page,
            "error": f"Network error: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "tag": tag,
            "page": page,
            "error": f"An unexpected error occurred: {e}",
        }


def insert_so_data(data: dict) -> None:
    """Insert StackOverflow data into Neo4j via Backend API."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest", json={"data": data["items"]}
        )
        response.raise_for_status()
        res_json = response.json()
        if res_json["status"] != "success":
            logger.error(f"Ingest failed: {res_json.get('message')}")
            st.error(f"Ingestion failed for a page: {res_json.get('message')}")
    except Exception as e:
        logger.error(f"Error posting ingestion data: {e}")
        st.error(f"Failed to send data to backend: {e}")


# ---------------------------------------------------------------------------
# Tab 1: Dashboard & Analytics Renderer
# ---------------------------------------------------------------------------
def _render_dashboard_tab():
    st.caption("Track your StackExchange data imports and database statistics")

    # Get database summary
    try:
        summary = get_database_summary() or {}
        total_questions = summary.get("total_questions") or 0
        total_tags = summary.get("total_tags") or 0
        total_answers = summary.get("total_answers") or 0
        total_users = summary.get("total_users") or 0
        total_imports = summary.get("total_imports") or 0

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📝 Total Questions",
                value=f"{total_questions:,}",
                help="Total questions imported from StackExchange",
            )

        with col2:
            st.metric(
                label="🏷️ Total Tags",
                value=f"{total_tags:,}",
                help="Unique tags in the database",
            )

        with col3:
            st.metric(
                label="💬 Total Answers",
                value=f"{total_answers:,}",
                help="Total answers imported",
            )

        with col4:
            st.metric(
                label="👥 Total Users",
                value=f"{total_users:,}",
                help="Unique users in the database",
            )

        # Additional metrics row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="📦 Import Sessions",
                value=f"{total_imports:,}",
                help="Total import sessions recorded",
            )

        with col2:
            last_import = summary.get("last_import")
            if last_import:
                if hasattr(last_import, "strftime"):
                    formatted_date = last_import.strftime("%Y-%m-%d %H:%M")
                else:
                    formatted_date = str(last_import)[:16]
                st.metric(
                    label="🕒 Last Import",
                    value=formatted_date,
                    help="Date of the most recent import session",
                )
            else:
                st.metric(
                    label="🕒 Last Import",
                    value="Never",
                    help="No import sessions recorded yet",
                )

        with col3:
            if total_imports > 0:
                avg_questions = total_questions / total_imports
                st.metric(
                    label="📈 Avg Questions/Import",
                    value=f"{avg_questions:.1f}",
                    help="Average questions imported per session",
                )
            else:
                st.metric(
                    label="📈 Avg Questions/Import",
                    value="N/A",
                    help="No import sessions yet",
                )

    except Exception as e:
        st.error(f"Could not fetch database summary: {e}")
        return

    st.divider()

    # Import History Section
    st.subheader("📋 Import History")

    try:
        history = get_import_history(limit=50)

        if history:
            df = pd.DataFrame(history)

            if "timestamp" in df.columns:
                df["formatted_time"] = df["timestamp"].apply(
                    lambda x: (
                        x.strftime("%Y-%m-%d %H:%M")
                        if hasattr(x, "strftime")
                        else str(x)[:16]
                    )
                )

            edited_df = st.data_editor(
                df[
                    ["id", "formatted_time", "site", "questions", "tags", "pages", "tags_list"]
                ].rename(
                    columns={
                        "formatted_time": "Date",
                        "site": "Site",
                        "questions": "Questions",
                        "tags": "Tags",
                        "pages": "Pages",
                        "tags_list": "Tag List",
                    }
                ),
                key="import_history_editor",
                hide_index=True,
                column_config={
                    "id": None,
                    "Date": st.column_config.DatetimeColumn(
                        "Date",
                        disabled=True,
                    ),
                    "Site": st.column_config.TextColumn(
                        "Site",
                        help="StackExchange site imported from",
                        disabled=True,
                    ),
                    "Tag List": st.column_config.ListColumn(
                        "Tag List",
                        help="List of tags imported",
                    ),
                },
                disabled=[
                    "Date",
                    "Site",
                    "Tags",
                ],
                num_rows="dynamic",
            )

            # Handle changes
            if st.session_state.get("import_history_editor"):
                changes = st.session_state["import_history_editor"]

                for index, updates in changes.get("edited_rows", {}).items():
                    row_id = df.iloc[index]["id"]
                    current_row = df.iloc[index].to_dict()

                    col_map_inv = {
                        "Questions": "total_questions",
                        "Tags": "total_tags",
                        "Pages": "total_pages",
                        "Tag List": "tags_list",
                    }

                    update_data = {}
                    for col, val in updates.items():
                        if col in col_map_inv:
                            update_data[col_map_inv[col]] = val

                    full_payload = {
                        "total_questions": update_data.get(
                            "total_questions", current_row.get("questions")
                        ),
                        "tags_list": update_data.get(
                            "tags_list", current_row.get("tags_list")
                        ),
                        "total_pages": update_data.get(
                            "total_pages", current_row.get("pages")
                        ),
                        "site": current_row.get("site", "stackoverflow"),
                    }

                    if update_import_log_api(row_id, full_payload):
                        st.success(f"Updated row {index + 1}")
                        st.rerun()

                for index in changes.get("deleted_rows", []):
                    row_id = df.iloc[index]["id"]
                    if delete_import_log_api(row_id):
                        st.success(f"Deleted row {index + 1}")
                        st.rerun()

            # Tag Import Summary
            st.divider()
            st.subheader("📦 Tag Import Summary")
            st.caption("Total pages imported across all sessions, classified by tag.")

            tag_stats = {}
            for _, row in edited_df.iterrows():
                tags = row.get("Tag List", [])
                pages = row.get("Pages", 0)

                if isinstance(tags, list):
                    for tag in tags:
                        if tag not in tag_stats:
                            tag_stats[tag] = {"Total Pages": 0, "Import Sessions": 0}
                        tag_stats[tag]["Total Pages"] += pages if pages else 0
                        tag_stats[tag]["Import Sessions"] += 1

            if tag_stats:
                summary_df = pd.DataFrame.from_dict(tag_stats, orient="index")
                summary_df.index.name = "Tag"
                summary_df = summary_df.reset_index().sort_values(
                    by="Total Pages", ascending=False
                )

                st.dataframe(
                    summary_df,
                    hide_index=True,
                    column_config={
                        "Tag": st.column_config.TextColumn("Tag", width="medium"),
                        "Total Pages": st.column_config.NumberColumn(
                            "Total Pages Imported", format="%d"
                        ),
                        "Import Sessions": st.column_config.NumberColumn(
                            "Total Sessions", format="%d"
                        ),
                    },
                )

                fig_tag = px.bar(
                    summary_df,
                    x="Total Pages",
                    y="Tag",
                    orientation="h",
                    title="Pages Imported per Tag",
                    labels={"Total Pages": "Total Pages", "Tag": "Tag"},
                    color="Total Pages",
                    color_continuous_scale="Viridis",
                )
                fig_tag.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_tag)
            else:
                st.info("No tag information available to summarize.")
        else:
            st.info(
                "No import history found. Use the 'StackExchange Importer' tab to load data."
            )

    except Exception as e:
        st.error(f"Could not fetch import history: {e}")


# ---------------------------------------------------------------------------
# Tab 2: StackExchange Importer Renderer
# ---------------------------------------------------------------------------
def _render_loader_tab():
    st.caption("Choose StackExchange tags to load into Neo4j graph database.")
    st.caption("Go to http://localhost:7474/ to explore the database directly.")

    input_text = st.text_input("Enter tags separated by commas", value="python", key="loader_tags")
    tags_to_import = [tag.strip() for tag in input_text.split(",") if tag.strip()]

    site = st.text_input("Enter Stack Exchange site", value="stackoverflow", key="loader_site").strip()

    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages (100 questions per page)", step=1, min_value=1, value=1, key="loader_num_pages"
        )
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1, value=1, key="loader_start_page")
    st.caption("Only questions with answers will be imported.")

    if st.button("📥 Start Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            info_placeholder = st.empty()
            error_placeholder = st.container()

            tasks_to_complete = len(tags_to_import) * int(num_pages)
            completed_tasks = 0
            total_imported_count = 0

            for tag in tags_to_import:
                for i in range(int(num_pages)):
                    result = load_so_data(tag, int(start_page) + i, site)
                    completed_tasks += 1
                    progress = (completed_tasks / tasks_to_complete) * 100

                    with info_placeholder:
                        if result["status"] == "success":
                            total_imported_count += result["count"]
                            st.info(
                                f"({progress:.2f}%) ✅ Success: Imported page {result['page']} for tag '{result['tag']}' ({result['count']} items)."
                            )
                        elif result["status"] == "empty":
                            st.info(
                                f"({progress:.2f}%) 🟡 Skipped: No items on page {result['page']} for tag '{result['tag']}'."
                            )
                    with error_placeholder:
                        if result["status"] == "error":
                            st.error(
                                f"({progress:.2f}%) ❌ Failed: Page {result['page']} for tag '{result['tag']}'. Reason: {result['error']}"
                            )

                    del result
                    gc.collect()
                    time.sleep(0.5)

            st.success(
                f"Import complete! Successfully imported {total_imported_count} questions.",
                icon="✅",
            )

            # Record the import session in Neo4j
            try:
                payload = {
                    "total_questions": total_imported_count,
                    "tags_list": tags_to_import,
                    "total_pages": int(num_pages),
                    "site": site,
                }
                rec_resp = requests.post(
                    f"{BACKEND_URL}/ingest/record", json=payload
                )
                rec_resp.raise_for_status()

                if rec_resp.json().get("status") == "success":
                    st.info("📊 Import session recorded in dashboard history")
                else:
                    st.warning(
                        f"Could not record import session: {rec_resp.json().get('message')}"
                    )
            except Exception as e:
                st.warning(f"Could not record import session: {e}")


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def render_page():
    st.header("📊 StackExchange Management")

    with st.sidebar:
        display_container_name()

    tab_dashboard, tab_loader = st.tabs(["📊 Analytics & Dashboard", "📥 StackExchange Importer"])

    with tab_dashboard:
        _render_dashboard_tab()

    with tab_loader:
        _render_loader_tab()


render_page()
