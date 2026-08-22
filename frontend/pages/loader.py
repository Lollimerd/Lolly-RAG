import os, time, requests, gc
from dotenv import load_dotenv
import streamlit as st
from streamlit.logger import get_logger
from utils.ui_utils import BACKEND_URL

logger = get_logger(__name__)

st.set_page_config(
    page_title="StackExchange Import",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

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
                    # After all retries, raise the last exception to be handled by the outer block
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


def load_high_score_so_data(site: str) -> None:
    """load stackoverflow data with a high score"""
    parameters = f"""?fromdate=1664150400&order=desc&sort=votes&site={site}&filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII"""
    data = requests.get(so_api_base_url + parameters).json()
    if "items" in data and data["items"]:
        if "error_name" in data:
            # backoff_time = min(300, 2 ** (page % 8))  # Max 300 seconds
            backoff_time = 10  # Fixed backoff time for high score data
            st.warning(f"API requested a backoff of {backoff_time} seconds.")
            time.sleep(backoff_time)
        insert_so_data(data)
    else:
        st.warning("No highly ranked items found. Skipping.")


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


# --- Streamlit ---
def get_tags() -> list[str]:
    """Gets a comma-separated string of tags and returns a clean list."""
    input_text = st.text_input("Enter tags separated by commas", value="python")
    return [tag.strip() for tag in input_text.split(",") if tag.strip()]


def get_site() -> str:
    """Gets the Stack Exchange site to import from."""
    site = st.text_input("Enter Stack Exchange site", value="stackoverflow")
    return site.strip()


def get_pages():
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages (100 questions per page)", step=1, min_value=1
        )
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1)
    st.caption("Only questions with answers will be imported.")
    return (int(num_pages), int(start_page))


# --- Main Page Rendering (Modified Logic) ---
def render_page():
    st.header("StackExchange Loader")
    st.subheader("Choose StackExchange tags to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the database.")

    site = get_site()
    tags_to_import = get_tags()
    num_pages, start_page = get_pages()

    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            info_placeholder = st.empty()
            error_placeholder = st.container()  # A container to log errors

            tasks_to_complete = len(tags_to_import) * num_pages
            completed_tasks = 0
            total_imported_count = 0

            # Sequential processing to minimize memory usage
            for tag in tags_to_import:
                for i in range(num_pages):
                    result = load_so_data(tag, start_page + i, site)
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

                    # Free memory between pages
                    del result
                    gc.collect()
                    time.sleep(0.5)

            st.success(
                f"Import complete! Successfully imported {total_imported_count} questions.",
                icon="✅",
            )

            # Record the import session in Neo4j
            try:
                # record_import_session call refactored to API
                payload = {
                    "total_questions": total_imported_count,
                    "tags_list": tags_to_import,
                    "total_pages": num_pages,
                    "site": site,
                }
                rec_resp = requests.post(
                    f"{BACKEND_URL}/ingest/record", json=payload
                )
                rec_resp.raise_for_status()

                if rec_resp.json().get("status") == "success":
                    st.info("📊 Import session recorded in dashboard")
                else:
                    st.warning(
                        f"Could not record import session: {rec_resp.json().get('message')}"
                    )
            except Exception as e:
                st.warning(f"Could not record import session: {e}")

render_page()
