import gc
import logging
import os
import time
import pandas as pd
import requests
import streamlit as st

from utils.ui_utils import (
    BACKEND_URL,
    clear_api_cache,
    delete_import_log_api,
    get_database_summary,
    get_import_history,
    update_import_log_api,
)

logger = logging.getLogger(__name__)

so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"


def load_so_data(tag: str, page: int, site: str) -> dict:
    """Load Stack Overflow data and handle potential errors gracefully."""
    try:
        api_key = os.getenv("STACKEXCHANGE_API_KEY")
        key_param = f"&key={api_key}" if api_key else ""
        site = site or "stackoverflow"
        parameters = f"?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}&site={site}&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb{key_param}"

        max_retries = 3
        data = None
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = requests.get(so_api_base_url + parameters, timeout=15)
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    raise last_exception or e

        if not data:
            raise last_exception or Exception("Failed to retrieve data after retries")

        if "items" in data and data["items"]:
            if "backoff" in data:
                time.sleep(data["backoff"])
            elif "error_name" in data:
                time.sleep(min(300, 2 ** (page % 8)))
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
            f"{BACKEND_URL}/ingest", json={"data": data["items"]}, timeout=60
        )
        response.raise_for_status()
        res_json = response.json()
        if res_json.get("status") != "success":
            logger.error(f"Ingest failed: {res_json.get('message')}")
            st.error(f"Ingestion failed for a page: {res_json.get('message')}")
        else:
            clear_api_cache()
    except Exception as e:
        logger.error(f"Error posting ingestion data: {e}")
        st.error(f"Failed to send data to backend: {e}")


# ---------------------------------------------------------------------------
# Tab 1: Dashboard & Analytics Renderer
# ---------------------------------------------------------------------------
def _render_dashboard_tab():
    st.caption("Track your StackExchange data imports and database statistics.")

    summary = get_database_summary() or {}
    total_questions = summary.get("total_questions") or 0
    total_tags = summary.get("total_tags") or 0
    total_answers = summary.get("total_answers") or 0
    total_users = summary.get("total_users") or 0
    total_imports = summary.get("total_imports") or 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric(label="Total questions", value=f"{total_questions:,}", help="Total questions imported from StackExchange")
    with col2:
        with st.container(border=True):
            st.metric(label="Total tags", value=f"{total_tags:,}", help="Unique tags in the database")
    with col3:
        with st.container(border=True):
            st.metric(label="Total answers", value=f"{total_answers:,}", help="Total answers imported")
    with col4:
        with st.container(border=True):
            st.metric(label="Total users", value=f"{total_users:,}", help="Unique users in the database")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(label="Import sessions", value=f"{total_imports:,}", help="Total import sessions recorded")
    with col2:
        with st.container(border=True):
            last_import = summary.get("last_import")
            formatted_date = str(last_import)[:16] if last_import else "Never"
            st.metric(label="Last import", value=formatted_date, help="Date of the most recent import session")
    with col3:
        with st.container(border=True):
            avg_questions = (total_questions / total_imports) if total_imports > 0 else 0
            st.metric(label="Avg questions / session", value=f"{avg_questions:.1f}" if total_imports > 0 else "N/A")

    st.divider()

    st.subheader("Import history")
    history = get_import_history(limit=50)

    if history:
        df = pd.DataFrame(history)
        if "timestamp" in df.columns:
            df["formatted_time"] = df["timestamp"].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)[:16]
            )

        edited_df = st.data_editor(
            df[["id", "formatted_time", "site", "questions", "tags", "pages", "tags_list"]].rename(
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
                "Date": st.column_config.DatetimeColumn("Date", disabled=True),
                "Site": st.column_config.TextColumn("Site", help="StackExchange site", disabled=True),
                "Tag List": st.column_config.ListColumn("Tag List", help="List of tags imported"),
            },
            disabled=["Date", "Site", "Tags"],
            num_rows="dynamic",
        )

        if st.session_state.get("import_history_editor"):
            changes = st.session_state["import_history_editor"]
            for index, updates in changes.get("edited_rows", {}).items():
                row_id = df.iloc[index]["id"]
                current_row = df.iloc[index].to_dict()
                col_map_inv = {"Questions": "total_questions", "Tags": "total_tags", "Pages": "total_pages", "Tag List": "tags_list"}
                update_data = {col_map_inv[col]: val for col, val in updates.items() if col in col_map_inv}

                full_payload = {
                    "total_questions": update_data.get("total_questions", current_row.get("questions")),
                    "tags_list": update_data.get("tags_list", current_row.get("tags_list")),
                    "total_pages": update_data.get("total_pages", current_row.get("pages")),
                    "site": current_row.get("site", "stackoverflow"),
                }

                if update_import_log_api(row_id, full_payload):
                    st.success(f"Updated row {index + 1}.")
                    st.rerun()

            for index in changes.get("deleted_rows", []):
                row_id = df.iloc[index]["id"]
                if delete_import_log_api(row_id):
                    st.success(f"Deleted row {index + 1}.")
                    st.rerun()

        st.divider()
        st.subheader("Tag import summary")
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
            summary_df = summary_df.reset_index().sort_values(by="Total Pages", ascending=False)

            c_chart, c_table = st.columns([0.6, 0.4])
            with c_chart:
                st.bar_chart(
                    summary_df,
                    x="Tag",
                    y="Total Pages",
                    horizontal=True,
                    color="#6366F1",
                )
            with c_table:
                st.dataframe(
                    summary_df,
                    hide_index=True,
                    column_config={
                        "Tag": st.column_config.TextColumn("Tag", width="medium"),
                        "Total Pages": st.column_config.NumberColumn("Pages imported", format="%d"),
                        "Import Sessions": st.column_config.NumberColumn("Sessions", format="%d"),
                    },
                )
        else:
            st.info("No tag information available to summarize.")
    else:
        st.info("No import history found. Use the 'StackExchange importer' tab to load data.")


# ---------------------------------------------------------------------------
# Tab 2: StackExchange Importer Renderer
# ---------------------------------------------------------------------------
def _render_loader_tab():
    st.caption("Choose StackExchange tags to load into the Neo4j developer knowledge graph.")

    with st.form("stackexchange_import_form"):
        input_text = st.text_input("Enter tags separated by commas", value="python", key="loader_tags")
        tags_to_import = [tag.strip() for tag in input_text.split(",") if tag.strip()]

        site = st.text_input("Stack Exchange site", value="stackoverflow", key="loader_site").strip()

        col1, col2 = st.columns(2)
        with col1:
            num_pages = st.number_input(
                "Number of pages (100 questions per page)", step=1, min_value=1, value=1, key="loader_num_pages"
            )
        with col2:
            start_page = st.number_input("Start page", step=1, min_value=1, value=1, key="loader_start_page")

        st.caption("Only questions with accepted or high-scoring answers will be ingested into the graph.")
        submit_import = st.form_submit_button("Start import", icon=":material/download:", type="primary")

    if submit_import and tags_to_import:
        with st.spinner("Importing questions and answers into Neo4j..."):
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
                                f"({progress:.1f}%) Success: Imported page {result['page']} for tag '{result['tag']}' ({result['count']} items).",
                                icon=":material/check_circle:",
                            )
                        elif result["status"] == "empty":
                            st.info(
                                f"({progress:.1f}%) Skipped: No items on page {result['page']} for tag '{result['tag']}'.",
                                icon=":material/info:",
                            )
                    with error_placeholder:
                        if result["status"] == "error":
                            st.error(
                                f"({progress:.1f}%) Failed: Page {result['page']} for tag '{result['tag']}'. Reason: {result['error']}",
                                icon=":material/error:",
                            )

                    del result
                    gc.collect()
                    time.sleep(0.3)

            st.success(
                f"Import complete! Successfully imported {total_imported_count} questions into Neo4j.",
                icon=":material/check_circle:",
            )

            try:
                payload = {
                    "total_questions": total_imported_count,
                    "tags_list": tags_to_import,
                    "total_pages": int(num_pages),
                    "site": site,
                }
                rec_resp = requests.post(f"{BACKEND_URL}/ingest/record", json=payload, timeout=10)
                rec_resp.raise_for_status()
                clear_api_cache()
            except Exception as e:
                logger.warning(f"Could not record import session: {e}")


# ---------------------------------------------------------------------------
# Page Entry
# ---------------------------------------------------------------------------
st.header("StackExchange Developer Knowledge")

tab_dashboard, tab_loader = st.tabs(["Analytics & dashboard", "StackExchange importer"])

with tab_dashboard:
    _render_dashboard_tab()

with tab_loader:
    _render_loader_tab()
