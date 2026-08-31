"""
doc_ingestion.py — Unstructured Document File Explorer & Ingestion
-------------------------------------------------------------------
Streamlit page featuring a File Explorer interface for uploading, browsing,
previewing, and managing unstructured documents (PDF, DOCX, TXT, Markdown)
stored in the Neo4j knowledge graph with customizable destination folders.
"""

import streamlit as st
from streamlit_markdown import st_markdown
from utils.doc_utils import (
    FILE_TYPE_INFO,
    SUPPORTED_TYPES,
    delete_all_in_folder,
    delete_document,
    fetch_document_chunks,
    fetch_documents,
    get_all_folders,
    parse_folder,
    update_document_metadata,
    upload_file,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Explorer — Lolly-RAG",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Upload Section with Customizable Folders
# ---------------------------------------------------------------------------
def render_upload_modal(user_id: str, docs: list[dict]) -> None:
    st.subheader("Upload new documents", help="Ingest documents into Neo4j knowledge graph")
    st.caption("Upload documents to chunk, embed, and index into your Neo4j knowledge graph.")

    # Display flash notification from recent upload
    if "upload_flash" in st.session_state:
        flash = st.session_state.pop("upload_flash")
        f_type = flash.get("type", "info")
        f_msg = flash.get("message", "")
        if f_type == "success":
            st.success(f_msg, icon=":material/check_circle:")
        elif f_type == "warning":
            st.warning(f_msg, icon=":material/warning:")
        elif f_type == "info":
            st.info(f_msg, icon=":material/info:")
        elif f_type == "error":
            st.error(f_msg, icon=":material/error:")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    all_folders = get_all_folders(docs)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        uploaded_files = st.file_uploader(
            "Select files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, Markdown, CSV, Excel (.xlsx, .xls)",
            key=f"file_uploader_{st.session_state['uploader_key']}",
        )

        existing_filenames = {d.get("filename", "") for d in docs}
        if uploaded_files:
            dup_files = [f.name for f in uploaded_files if f.name in existing_filenames]
            if dup_files:
                st.info(
                    f"**{len(dup_files)} duplicate file(s) detected:** `{', '.join(dup_files)}`. "
                    "Duplicates will be skipped automatically unless overwrite is enabled.",
                    icon=":material/info:",
                )

    with col2:
        folder_options = all_folders + ["+ Create New Folder..."]
        selected_folder_opt = st.selectbox(
            "Destination folder",
            options=folder_options,
            index=0,
            key="upload_folder_selector",
        )

        target_folder = selected_folder_opt
        if selected_folder_opt == "+ Create New Folder...":
            new_folder_input = st.text_input(
                "New folder name",
                placeholder="e.g. Architecture Specs, Client Proposals",
                key="upload_new_folder_input",
            ).strip()
            if new_folder_input:
                target_folder = new_folder_input
                if target_folder not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(target_folder)
            else:
                target_folder = "Root"

        description = st.text_input(
            "Document description (optional)",
            placeholder="e.g. Q3 API design spec",
            key="upload_doc_desc",
        )

        overwrite = st.checkbox(
            "Overwrite if already ingested",
            value=False,
            help="If checked, existing documents with the same name will be replaced and re-embedded.",
            key="upload_overwrite_chk",
        )

        has_csv = bool(uploaded_files and any(f.name.lower().endswith(".csv") for f in uploaded_files))
        has_xlsx = bool(uploaded_files and any(f.name.lower().endswith((".xlsx", ".xls")) for f in uploaded_files))

        selected_engine = "pandas"
        if has_csv:
            engine_choice = st.radio(
                "CSV Ingestion engine",
                options=["Pandas (Structured RAG)", "APOC (Database Batch)"],
                horizontal=True,
                index=0,
                help="Pandas creates structured markdown tables with preserved headers (recommended for RAG). APOC runs database-side batch loading via apoc.load.csv.",
                key="upload_engine_choice",
            )
            selected_engine = "apoc" if engine_choice and "APOC" in engine_choice else "pandas"
        elif has_xlsx:
            st.caption("📈 **Excel detected**: Using Pandas engine for multi-sheet structured ingestion.")

        engine_badge = selected_engine.upper() if has_csv else "PANDAS"
        st.caption(f"Destination: :material/folder: **`{target_folder}`** &nbsp;|&nbsp; Engine: **`{engine_badge}`**")

    if uploaded_files and st.button("Ingest selected files", icon=":material/upload:", type="primary"):
        total = len(uploaded_files)
        progress_bar = st.progress(0.0, text="Starting document ingestion...")
        succeeded, skipped, failed = 0, 0, 0

        for idx, file in enumerate(uploaded_files):
            ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
            file_info = FILE_TYPE_INFO.get(ext, {"icon": "📄"})
            icon = file_info["icon"]

            progress_bar.progress(idx / total, text=f"{icon} Processing `{file.name}`...")

            # Use APOC only for CSV if selected; XLSX and other files always use Pandas
            file_engine = selected_engine if ext == "csv" else "pandas"

            with st.status(f"Ingesting `{file.name}` into `{target_folder}`...", expanded=False) as status:
                try:
                    file_bytes = file.read()
                    result = upload_file(
                        file_bytes=file_bytes,
                        filename=file.name,
                        user_id=user_id,
                        description=description,
                        folder=target_folder or "Root",
                        force=overwrite,
                        engine=file_engine,
                    )
                    status_code = result.get("status")
                    if status_code == "success":
                        chunk_count = result.get("chunk_count", 0)
                        status.update(
                            label=f"`{file.name}` ({chunk_count} chunks stored in `{target_folder}`)",
                            state="complete",
                        )
                        succeeded += 1
                    elif status_code == "skipped":
                        status.update(
                            label=f"`{file.name}` (Already ingested — duplicate skipped)",
                            state="complete",
                        )
                        skipped += 1
                    else:
                        status.update(label=f"`{file.name}` — {result.get('message')}", state="error")
                        failed += 1
                except Exception as exc:
                    status.update(label=f"`{file.name}` — {str(exc)[:150]}", state="error")
                    failed += 1

            progress_bar.progress((idx + 1) / total)

        progress_bar.empty()

        if succeeded == total:
            flash = {"type": "success", "message": f"Successfully ingested {total} file(s) into folder `{target_folder}`."}
        elif skipped == total:
            flash = {"type": "info", "message": f"All {total} file(s) were already ingested. Duplicate ingestion skipped."}
        elif failed == 0:
            flash = {"type": "success", "message": f"Processed {total} file(s): {succeeded} newly ingested, {skipped} duplicate(s) skipped."}
        elif succeeded > 0 or skipped > 0:
            flash = {"type": "warning", "message": f"Processed with issues: {succeeded} ingested, {skipped} skipped, {failed} failed."}
        else:
            flash = {"type": "error", "message": f"File ingestion failed for all {total} file(s). Check backend logs."}

        st.session_state["upload_flash"] = flash
        st.session_state["uploader_key"] += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Manage Custom Folders Section
# ---------------------------------------------------------------------------
def render_manage_folders_section(docs: list[dict]) -> None:
    st.subheader("Manage destination folders")
    st.caption("Inspect, create, or remove folders for organizing your documents.")

    all_folders = get_all_folders(docs)

    fcol1, fcol2 = st.columns([0.6, 0.4])

    with fcol1:
        st_markdown("#### Existing folders")
        for folder in all_folders:
            folder_docs = [d for d in docs if parse_folder(d.get("description", ""))[0] == folder]
            count = len(folder_docs)
            chunks = sum(d.get("chunk_count", 0) for d in folder_docs)

            with st.container(border=True):
                rcol1, rcol2 = st.columns([0.68, 0.32])
                with rcol1:
                    st_markdown(f":material/folder: **`{folder}`** &nbsp;·&nbsp; `{count} files` &nbsp;·&nbsp; `{chunks} chunks`")
                with rcol2:
                    if count > 0:
                        with st.popover(f"Clear ({count})", icon=":material/delete:", help=f"Permanently delete all {count} files in '{folder}'"):
                            st_markdown(f"**Clear all {count} files in `{folder}`?**")
                            st.caption("This will remove all documents and their vector embeddings in Neo4j.")
                            if st.button("Confirm clear", key=f"conf_clear_mgr_{folder}", type="primary"):
                                succ, _ = delete_all_in_folder(folder_docs)
                                st.success(f"Deleted {succ} file(s) from `{folder}`")
                                st.rerun()
                    elif folder != "Root":
                        if st.button("Remove", icon=":material/delete:", key=f"del_empty_folder_{folder}", help="Delete this empty folder"):
                            if folder in st.session_state["custom_folders"]:
                                st.session_state["custom_folders"].remove(folder)
                                st.success(f"Removed empty folder `{folder}`")
                                st.rerun()
                    else:
                        st.caption("Default root folder")

    with fcol2:
        st_markdown("#### Add new folder")
        with st.form("create_folder_form", clear_on_submit=True):
            new_f_name = st.text_input("Folder name", placeholder="e.g. Legal, Specifications")
            submit_btn = st.form_submit_button("Create folder", icon=":material/create_new_folder:", type="primary")

            if submit_btn and new_f_name.strip():
                clean_name = new_f_name.strip()
                if clean_name not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(clean_name)
                    st.success(f"Folder `📁 {clean_name}` created!")
                    st.rerun()
                else:
                    st.info(f"Folder `{clean_name}` already exists.", icon=":material/info:")


# ---------------------------------------------------------------------------
# File Inspector Drawer with Move Folder capability
# ---------------------------------------------------------------------------
def render_file_inspector(doc: dict, folder: str, clean_desc: str, all_folders: list[str]) -> None:
    filename = doc.get("filename", "Unknown")
    doc_id = doc.get("id", "")
    ext = (doc.get("file_type") or "").lower()
    file_info = FILE_TYPE_INFO.get(ext, {"icon": "📄", "label": ext.upper(), "color": "#64748B"})
    upload_date = (doc.get("upload_date") or "")[:19].replace("T", " ")
    chunk_count = doc.get("chunk_count", 0)
    user_id = doc.get("user_id") or "default"

    with st.container(border=True):
        col_title, col_close = st.columns([0.85, 0.15])
        with col_title:
            st_markdown(f"### {file_info['icon']} {filename}")
            st.caption(f":material/folder: Folder: **`{folder}`** &nbsp;|&nbsp; ID: `{doc_id}`")
        with col_close:
            if st.button("Close", icon=":material/close:", key=f"close_inspect_{doc_id}"):
                st.session_state["selected_doc_id"] = None
                st.rerun()

        # Metadata metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            with st.container(border=True):
                st.metric("Format", file_info["label"])
        with mcol2:
            with st.container(border=True):
                st.metric("Total chunks", f"{chunk_count}")
        with mcol3:
            with st.container(border=True):
                st.metric("Uploaded by", user_id)
        with mcol4:
            with st.container(border=True):
                st.metric("Uploaded at", upload_date or "N/A")

        # Edit Folder / Move Location & Description
        with st.expander("Edit folder & description", icon=":material/edit:", expanded=False):
            move_col1, move_col2 = st.columns([0.5, 0.5])
            with move_col1:
                cur_idx = all_folders.index(folder) if folder in all_folders else 0
                new_folder_choice = st.selectbox(
                    "Move to folder",
                    options=all_folders + ["+ Create New Folder..."],
                    index=cur_idx,
                    key=f"move_folder_sel_{doc_id}",
                )
                if new_folder_choice == "+ Create New Folder...":
                    inline_folder = st.text_input("New folder name", key=f"inline_f_{doc_id}").strip()
                    target_move_folder = inline_folder if inline_folder else folder
                else:
                    target_move_folder = new_folder_choice

            with move_col2:
                new_desc_input = st.text_input(
                    "Description",
                    value=clean_desc,
                    key=f"edit_desc_{doc_id}",
                )

            if st.button("Save changes", icon=":material/save:", key=f"save_meta_{doc_id}"):
                if update_document_metadata(doc_id, target_move_folder, new_desc_input):
                    st.success(f"Updated `{filename}` to folder `{target_move_folder}`!")
                    if target_move_folder not in st.session_state.get("custom_folders", []):
                        st.session_state["custom_folders"].append(target_move_folder)
                    st.rerun()

        if clean_desc:
            st.info(f"**Description:** {clean_desc}", icon=":material/description:")

        # Chunk Preview Section
        st_markdown("#### Indexed document chunks")
        chunks = fetch_document_chunks(doc_id)

        if chunks:
            st.caption(f"Showing {len(chunks)} chunks retrieved from Neo4j vector store:")
            for chunk in chunks:
                c_idx = chunk.get("chunk_index", 0)
                c_content = chunk.get("content", "")
                c_source = chunk.get("source", "")
                expander_label = f"Chunk #{c_idx + 1} ({len(c_content)} chars)"
                if c_source and c_source != filename:
                    expander_label += f" — {c_source}"
                with st.expander(expander_label, icon=":material/segment:", expanded=(c_idx == 0)):
                    tab_formatted, tab_raw = st.tabs(["Formatted View", "Raw Content"])
                    with tab_formatted:
                        st_markdown(c_content, mermaid_theme="dark", theme_color="blue")
                    with tab_raw:
                        st.text_area(
                            label=f"Raw Chunk Content {c_idx + 1}",
                            value=c_content,
                            height=120,
                            disabled=True,
                            key=f"chunk_raw_{doc_id}_{c_idx}",
                            label_visibility="collapsed",
                        )
        else:
            st.caption("No individual chunk records found or vector embedding in progress.")

        with st.popover("Delete document", icon=":material/delete:", help="Permanently delete this document"):
            st_markdown(f"**Delete `{filename}`?**")
            st.caption("This will remove the file and all its vector embeddings from Neo4j.")
            if st.button("Confirm deletion", key=f"del_inspect_{doc_id}", type="primary"):
                if delete_document(doc_id):
                    st.success(f"Deleted `{filename}`")
                    st.session_state["selected_doc_id"] = None
                    st.rerun()


# ---------------------------------------------------------------------------
# File Explorer View
# ---------------------------------------------------------------------------
def render_explorer_view(docs: list[dict], all_folders: list[str]) -> None:
    if not docs:
        st.info("Your document library is currently empty. Upload files in the 'Upload documents' tab.", icon=":material/folder_open:")
        return

    enriched_docs = []
    for doc in docs:
        folder, clean_desc = parse_folder(doc.get("description", ""))
        enriched_docs.append({**doc, "_folder": folder, "_clean_desc": clean_desc})

    # Toolbar: Search, Grouping, View Mode, Refresh
    tcol1, tcol2, tcol3, tcol4 = st.columns([0.42, 0.25, 0.2, 0.13], vertical_alignment="center")

    with tcol1:
        search_term = st.text_input(
            "Search documents",
            placeholder="Search by name, folder, or description...",
            label_visibility="collapsed",
        ).strip().lower()

    with tcol2:
        group_by = st.selectbox(
            "Group by",
            options=["Folder / Category", "File Extension", "Uploaded By"],
            label_visibility="collapsed",
        )

    with tcol3:
        view_mode = st.segmented_control(
            "View mode",
            options=["Table", "Grid"],
            default="Table",
            label_visibility="collapsed",
        ) or "Table"

    with tcol4:
        if st.button("Refresh", icon=":material/refresh:", width="stretch"):
            st.rerun()

    # Filter documents based on search term
    filtered_docs = enriched_docs
    if search_term:
        filtered_docs = [
            d for d in enriched_docs
            if search_term in d.get("filename", "").lower()
            or search_term in d.get("_clean_desc", "").lower()
            or search_term in d.get("_folder", "").lower()
        ]
        st.caption(f"Showing {len(filtered_docs)} of {len(enriched_docs)} documents matching '`{search_term}`'")

    # Inspector view if a doc is selected
    selected_doc_id = st.session_state.get("selected_doc_id")
    selected_doc_match = next((d for d in enriched_docs if d.get("id") == selected_doc_id), None)

    if selected_doc_match:
        render_file_inspector(
            selected_doc_match,
            selected_doc_match.get("_folder", "Root"),
            selected_doc_match.get("_clean_desc", ""),
            all_folders,
        )

    # Group documents
    groups: dict[str, list[dict]] = {}
    for doc in filtered_docs:
        if group_by == "Folder / Category":
            key = f"📁 {doc.get('_folder', 'Root')}"
        elif group_by == "File Extension":
            ext = (doc.get("file_type") or "other").upper()
            icon = FILE_TYPE_INFO.get(doc.get("file_type", "").lower(), {}).get("icon", "📄")
            key = f"{icon} {ext} Files"
        else:
            key = f"👤 {doc.get('user_id') or 'default'}"

        groups.setdefault(key, []).append(doc)

    # Render Groups (Folders)
    for folder_name, folder_files in sorted(groups.items()):
        total_chunks_in_folder = sum(f.get("chunk_count", 0) for f in folder_files)
        folder_header = f"{folder_name} &nbsp;·&nbsp; `{len(folder_files)} file(s)` &nbsp;·&nbsp; `{total_chunks_in_folder} chunk(s)`"

        with st.expander(folder_header, icon=":material/folder:", expanded=True):
            f_col_info, f_col_del = st.columns([0.72, 0.28], vertical_alignment="center")
            with f_col_info:
                st.caption(f"**{len(folder_files)} file(s)** inside **{folder_name}**")
            with f_col_del:
                with st.popover(f"Clear folder ({len(folder_files)})", icon=":material/delete:", help=f"Permanently delete all {len(folder_files)} documents in {folder_name}"):
                    st_markdown(f"**Delete all files in `{folder_name}`?**")
                    st.caption("This will permanently remove all files and vector embeddings in Neo4j.")
                    if st.button("Confirm delete all", key=f"conf_del_all_{folder_name}", type="primary"):
                        succ, _ = delete_all_in_folder(folder_files)
                        st.success(f"Deleted {succ} file(s) from `{folder_name}`")
                        st.rerun()

            if view_mode == "Grid":
                cols = st.columns(3)
                for idx, doc in enumerate(folder_files):
                    col = cols[idx % 3]
                    filename = doc.get("filename", "Unknown")
                    doc_id = doc.get("id", "")
                    ext = (doc.get("file_type") or "").lower()
                    file_info = FILE_TYPE_INFO.get(ext, {"icon": "📄", "label": ext.upper()})
                    chunk_count = doc.get("chunk_count", 0)
                    upload_date = (doc.get("upload_date") or "")[:10]
                    clean_desc = doc.get("_clean_desc", "")

                    with col:
                        with st.container(border=True):
                            st_markdown(f"#### {file_info['icon']} {filename}")
                            st.caption(f"**Type:** `{ext}` &nbsp;|&nbsp; **Chunks:** `{chunk_count}`")
                            if clean_desc:
                                st.caption(f"📝 {clean_desc[:55]}..." if len(clean_desc) > 55 else f"📝 {clean_desc}")
                            if upload_date:
                                st.caption(f"📅 {upload_date}")

                            btn_col1, btn_col2 = st.columns([0.55, 0.45])
                            with btn_col1:
                                if st.button("Inspect", icon=":material/visibility:", key=f"inspect_{doc_id}", width="stretch"):
                                    st.session_state["selected_doc_id"] = doc_id
                                    st.rerun()
                            with btn_col2:
                                with st.popover("Delete", icon=":material/delete:", key=f"pop_del_{doc_id}"):
                                    st.caption(f"Delete `{filename}`?")
                                    if st.button("Confirm", key=f"del_grid_{doc_id}", type="primary"):
                                        if delete_document(doc_id):
                                            st.success(f"Deleted `{filename}`")
                                            if st.session_state.get("selected_doc_id") == doc_id:
                                                st.session_state["selected_doc_id"] = None
                                            st.rerun()

            else:
                table_data = []
                for doc in folder_files:
                    ext = (doc.get("file_type") or "").lower()
                    icon = FILE_TYPE_INFO.get(ext, {}).get("icon", "📄")
                    table_data.append({
                        "id": doc.get("id"),
                        "Icon": icon,
                        "Filename": doc.get("filename"),
                        "Type": ext.upper(),
                        "Folder": doc.get("_folder", "Root"),
                        "Chunks": doc.get("chunk_count", 0),
                        "Uploaded By": doc.get("user_id", "default"),
                        "Date": (doc.get("upload_date") or "")[:19].replace("T", " "),
                        "Description": doc.get("_clean_desc", ""),
                    })

                st.dataframe(
                    table_data,
                    hide_index=True,
                    column_config={
                        "id": None,
                        "Icon": st.column_config.TextColumn("", width="small"),
                        "Filename": st.column_config.TextColumn("File Name", width="medium"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Folder": st.column_config.TextColumn("Folder", width="small"),
                        "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
                        "Uploaded By": st.column_config.TextColumn("Author"),
                        "Date": st.column_config.TextColumn("Upload Date"),
                        "Description": st.column_config.TextColumn("Description", width="large"),
                    },
                )

                selected_row = st.selectbox(
                    f"Select file from {folder_name} to inspect or manage",
                    options=[d["Filename"] for d in table_data],
                    key=f"select_table_{folder_name}",
                )
                if selected_row:
                    matched = next((d for d in folder_files if d.get("filename") == selected_row), None)
                    if matched:
                        act_col1, act_col2 = st.columns([0.3, 0.7])
                        with act_col1:
                            if st.button("Inspect file", icon=":material/visibility:", key=f"inspect_tbl_{matched['id']}"):
                                st.session_state["selected_doc_id"] = matched["id"]
                                st.rerun()
                        with act_col2:
                            with st.popover("Delete file", icon=":material/delete:", key=f"pop_tbl_{matched['id']}"):
                                st.caption(f"Permanently delete `{selected_row}`?")
                                if st.button("Confirm delete", key=f"del_tbl_{matched['id']}", type="primary"):
                                    if delete_document(matched["id"]):
                                        st.success(f"Deleted `{selected_row}`")
                                        st.rerun()


# ---------------------------------------------------------------------------
# Main Page Execution (Direct Script Execution per Streamlit Best Practices)
# ---------------------------------------------------------------------------
st.html(
    """
    <div style="text-align: center; padding: 12px 0 8px;">
        <h1 style="
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        ">Document File Explorer</h1>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
            Organize, browse, and inspect your unstructured document library and vector embeddings in Neo4j.
        </p>
    </div>
    """
)

user_id = st.session_state.get("user_name", "default")
docs = fetch_documents()
all_folders = get_all_folders(docs)

# Summary KPI cards
total_docs = len(docs)
total_chunks = sum(d.get("chunk_count", 0) for d in docs)
pdf_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "pdf")
docx_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "docx")
txt_count = sum(1 for d in docs if (d.get("file_type") or "").lower() in ("txt", "md"))

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    with st.container(border=True):
        st.metric("Total files", f"{total_docs}")
with kpi2:
    with st.container(border=True):
        st.metric("Total chunks", f"{total_chunks:,}")
with kpi3:
    with st.container(border=True):
        st.metric("Folders", f"{len(all_folders)}")
with kpi4:
    with st.container(border=True):
        st.metric("PDF files", f"{pdf_count}")
with kpi5:
    with st.container(border=True):
        st.metric("Word & text", f"{docx_count + txt_count}")

# Main Tabs
tab_explorer, tab_upload, tab_folders = st.tabs([
    ":material/folder: Browse files",
    ":material/upload: Upload documents",
    ":material/folder_managed: Manage folders",
])

with tab_explorer:
    render_explorer_view(docs, all_folders)

with tab_upload:
    render_upload_modal(user_id=user_id, docs=docs)

with tab_folders:
    render_manage_folders_section(docs)
