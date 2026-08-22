import streamlit as st
import pandas as pd
import plotly.express as px

from utils.ui_utils import (
    get_database_summary,
    get_import_history,
    display_container_name,
    update_import_log_api,
    delete_import_log_api,
)

st.set_page_config(
    page_title="StackExchange Import Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)


def render_page():
    st.header("📊 StackExchange Import Dashboard")
    st.caption("Track your StackExchange data imports and database statistics")

    # Display container status in sidebar
    with st.sidebar:
        display_container_name()

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
                # Format the datetime for display
                if hasattr(last_import, "strftime"):
                    formatted_date = last_import.strftime("%Y-%m-%d %H:%M")
                else:
                    formatted_date = str(last_import)[:16]  # Truncate if it's a string
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
            # Calculate average questions per import
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
        # Get import history
        history = get_import_history(limit=50)

        if history:
            # Convert to DataFrame for better display
            df = pd.DataFrame(history)

            # Format timestamp for display
            if "timestamp" in df.columns:
                df["formatted_time"] = df["timestamp"].apply(
                    lambda x: (
                        x.strftime("%Y-%m-%d %H:%M")
                        if hasattr(x, "strftime")
                        else str(x)[:16]
                    )
                )

            # Display as interactive editor
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
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": None,  # Hide ID column
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
                ],  # Disable editing of date, site and calculated tags count
                num_rows="dynamic",  # Allow adding/deleting rows (we only handle deletion)
            )

            # Handle changes
            if st.session_state.get("import_history_editor"):
                changes = st.session_state["import_history_editor"]

                # Handle updates
                for index, updates in changes.get("edited_rows", {}).items():
                    # Get the ID from the original dataframe
                    # Note: index is the row index in the displayed dataframe
                    row_id = df.iloc[index]["id"]

                    # Synthesize new data
                    current_row = df.iloc[index].to_dict()

                    # Map column names back to internal names for updates
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

            # --- 📦 Tag Import Summary Section ---
            st.divider()
            st.subheader("📦 Tag Import Summary")
            st.caption("Total pages imported across all sessions, classified by tag.")

            # Aggregate data based on Tag List and Pages
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
                # Convert to DataFrame for display
                summary_df = pd.DataFrame.from_dict(tag_stats, orient="index")
                summary_df.index.name = "Tag"
                summary_df = summary_df.reset_index().sort_values(
                    by="Total Pages", ascending=False
                )

                # Display table
                st.dataframe(
                    summary_df,
                    use_container_width=True,
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

                # Visualization: Horizontal Bar Chart of Pages per Tag
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
                st.plotly_chart(fig_tag, use_container_width=True)
            else:
                st.info("No tag information available to summarize.")
        else:
            st.info(
                "No import history found. Start importing data to see statistics here."
            )

    except Exception as e:
        st.error(f"Could not fetch import history: {e}")

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📥 Go to Loader", use_container_width=True):
            st.switch_page("pages/loader.py")


render_page()
