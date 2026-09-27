"""
doc_ingestion.py — Unstructured Document File Explorer & Ingestion
-------------------------------------------------------------------
Streamlit page featuring a File Explorer interface for uploading, browsing,
previewing, and managing unstructured documents (PDF, DOCX, TXT, Markdown)
stored in the Neo4j knowledge graph with customizable destination folders.
"""

from concurrent.futures import ThreadPoolExecutor
import queue
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
    upload_file_stream,
)

st.set_page_config(
    page_title="Document Explorer — Lolly-RAG",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _batch_upload(prepared_files: list, user_id: str, description: str, target_folder: str, overwrite: bool) -> tuple[int, int, int]:
    """Args: prepared_files: File data tuples, user_id: User identifier, description: Document description, target_folder: Destination folder, overwrite: Overwrite flag."""
    total = len(prepared_files)
    progress_bar = st.progress(0.0, text=f"Starting ingestion ({total} files)...")
    succeeded, skipped, failed = 0, 0, 0
    progress_queue: queue.Queue[dict] = queue.Queue()

    def _worker(fname: str, fb: bytes, fengine: str) -> None:
        try:
            res = upload_file_stream(
                file_bytes=fb, filename=fname, user_id=user_id, description=description,
                folder=target_folder or "Root", force=overwrite, engine=fengine,
                progress_callback=lambda ev: progress_queue.put({"type": "progress", "filename": fname, "event": ev}),
            )
            progress_queue.put({"type": "done", "filename": fname, "result": res})
        except Exception as exc:
            progress_queue.put({"type": "done", "filename": fname, "result": {"status": "error", "message": str(exc)}})

    file_progress = {fname: 0.0 for fname, _, _ in prepared_files}
    completed_count = 0
    status_container = st.container()

    with ThreadPoolExecutor(max_workers=min(4, total)) as executor:
        futures = [executor.submit(_worker, fn, fb, fe) for fn, fb, fe in prepared_files]
        while completed_count < total:
            try:
                msg = progress_queue.get(timeout=0.08)
            except queue.Empty:
                if all(f.done() for f in futures):
                    break
                continue
            fname, mtype = msg.get("filename", ""), msg.get("type")
            if mtype == "progress":
                prog = float(msg.get("event", {}).get("progress", 0.0))
                if prog > file_progress.get(fname, 0.0):
                    file_progress[fname] = prog
                pct = int(min(0.99, sum(file_progress.values()) / total) * 100)
                msg_txt = msg.get("event", {}).get("message", "Processing...")
                txt = f"Ingesting ({pct}%): {msg_txt}" if total == 1 else f"Ingesting ({pct}%) [{completed_count}/{total}]: {fname} — {msg_txt}"
                progress_bar.progress(min(0.99, sum(file_progress.values()) / total), text=txt)
            elif mtype == "done":
                completed_count += 1
                file_progress[fname] = 1.0
                res = msg.get("result", {})
                scode = res.get("status")
                progress_bar.progress(min(1.0, sum(file_progress.values()) / total), text=f"Finished {fname} ({completed_count}/{total})")
                if scode == "success":
                    status_container.success(f"`{fname}` ({res.get('chunk_count', 0)} chunks stored in `{target_folder}`)", icon=":material/check_circle:")
                    succeeded += 1
                elif scode == "skipped":
                    status_container.info(f"`{fname}` (Duplicate skipped)", icon=":material/info:")
                    skipped += 1
                else:
                    status_container.error(f"`{fname}` — {res.get('message', 'Failed')}", icon=":material/error:")
                    failed += 1
    progress_bar.empty()
    return succeeded, skipped, failed

def render_upload_modal(user_id: str, docs: list[dict]) -> None:
    """Args: user_id: User identifier, docs: Ingested document records."""
    st.subheader("Upload new documents", help="Ingest documents into Neo4j knowledge graph")
    st.caption("Upload documents to chunk, embed, and index into your Neo4j knowledge graph.")

    if "upload_flash" in st.session_state:
        flash = st.session_state.pop("upload_flash")
        ftype, fmsg = flash.get("type", "info"), flash.get("message", "")
        fn = getattr(st, ftype, st.info)
        fn(fmsg)

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    all_folders = get_all_folders(docs)
    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        uploaded_files = st.file_uploader(
            "Select files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="Supported formats: PDF, Word, PowerPoint, Text, Markdown, CSV, Excel, Images (OCR)",
            key=f"file_uploader_{st.session_state['uploader_key']}",
        )
        existing_filenames = {d.get("filename", "") for d in docs}
        if uploaded_files:
            dup_files = [f.name for f in uploaded_files if f.name in existing_filenames]
            if dup_files:
                st.info(f"**{len(dup_files)} duplicate(s) detected:** `{', '.join(dup_files)}`. Duplicates skipped unless overwrite enabled.", icon=":material/info:")

    with col2:
        selected_folder_opt = st.selectbox("Destination folder", options=all_folders + ["+ Create New Folder..."], index=0, key="upload_folder_selector")
        target_folder = selected_folder_opt
        if selected_folder_opt == "+ Create New Folder...":
            new_input = st.text_input("New folder name", placeholder="e.g. Architecture Specs", key="upload_new_folder_input").strip()
            if new_input:
                target_folder = new_input
                if target_folder not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(target_folder)
            else:
                target_folder = "Root"

        description = st.text_input("Document description (optional)", placeholder="e.g. Q3 API design spec", key="upload_doc_desc")
        overwrite = st.checkbox("Overwrite if already ingested", value=False, key="upload_overwrite_chk")
        has_csv = bool(uploaded_files and any(f.name.lower().endswith(".csv") for f in uploaded_files))
        has_xlsx = bool(uploaded_files and any(f.name.lower().endswith((".xlsx", ".xls")) for f in uploaded_files))

        if has_csv and has_xlsx:
            st.caption("⚡ **Tabular files detected**: CSV routed to **APOC**; Excel to **Pandas**.")
        elif has_csv:
            st.caption("⚡ **CSV detected**: Ingested via **APOC** database batch engine.")
        elif has_xlsx:
            st.caption("📈 **Excel detected**: Ingested via **Pandas** multi-sheet structured engine.")
        st.caption(f"Destination: 📁 **`{target_folder}`**")

    if uploaded_files and st.button("Ingest selected files", icon=":material/upload:", type="primary"):
        total = len(uploaded_files)
        prepared_files = [
            (f.name, f.getvalue() if hasattr(f, "getvalue") else f.read(), "apoc" if f.name.lower().endswith(".csv") else "pandas")
            for f in uploaded_files
        ]
        succeeded, skipped, failed = _batch_upload(prepared_files, user_id, description, target_folder, overwrite)

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

def render_manage_folders_section(docs: list[dict]) -> None:
    """Args: docs: Ingested document records."""
    st.subheader("Manage destination folders")
    st.caption("Inspect, create, or remove folders for organizing your documents.")
    all_folders = get_all_folders(docs)
    fcol1, fcol2 = st.columns([0.6, 0.4])

    with fcol1:
        st_markdown("#### Existing folders")
        for folder in all_folders:
            folder_docs = [d for d in docs if parse_folder(d.get("description", ""))[0] == folder]
            count, chunks = len(folder_docs), sum(d.get("chunk_count", 0) for d in folder_docs)
            with st.container(border=True):
                r1, r2 = st.columns([0.68, 0.32])
                with r1:
                    st_markdown(f"📁 **`{folder}`** · `{count} files` · `{chunks} chunks`")
                with r2:
                    if count > 0:
                        with st.popover(f"Clear ({count})", icon=":material/delete:", help=f"Clear files in {folder}"):
                            st_markdown(f"**Clear all {count} files in `{folder}`?**")
                            st.caption("This will remove all documents and their vector embeddings in Neo4j.")
                            if st.button("Confirm clear", key=f"conf_clear_{folder}", type="primary"):
                                succ, _ = delete_all_in_folder(folder_docs)
                                st.success(f"Deleted {succ} file(s) from `{folder}`")
                                st.rerun()
                    elif folder != "Root":
                        if st.button("Remove", icon=":material/delete:", key=f"del_f_{folder}"):
                            if folder in st.session_state.get("custom_folders", []):
                                st.session_state["custom_folders"].remove(folder)
                                st.rerun()
                    else:
                        st.caption("Default root")

    with fcol2:
        st_markdown("#### Add new folder")
        with st.form("create_folder_form", clear_on_submit=True):
            new_name = st.text_input("Folder name", placeholder="e.g. Legal, Specifications")
            if st.form_submit_button("Create folder", icon=":material/create_new_folder:", type="primary") and new_name.strip():
                clean_name = new_name.strip()
                if clean_name not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(clean_name)
                    st.success(f"Folder `📁 {clean_name}` created!")
                    st.rerun()
                else:
                    st.info(f"Folder `{clean_name}` already exists.", icon=":material/info:")

def render_file_inspector(doc: dict, folder: str, clean_desc: str, all_folders: list[str]) -> None:
    """Args: doc: Document record, folder: Folder name, clean_desc: Cleaned description, all_folders: Known folders."""
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
            st.caption(f"📁 Folder: **`{folder}`** | ID: `{doc_id}`")
        with col_close:
            if st.button("Close", icon=":material/close:", key=f"close_inspect_{doc_id}"):
                st.session_state["selected_doc_id"] = None
                st.rerun()

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        for col, (label, val) in zip((mcol1, mcol2, mcol3, mcol4), [
            ("Format", file_info["label"]),
            ("Total chunks", f"{chunk_count}"),
            ("Uploaded by", user_id),
            ("Uploaded at", upload_date or "N/A"),
        ]):
            with col:
                with st.container(border=True):
                    st.metric(label, val)

        with st.expander("Edit folder & description", icon=":material/edit:", expanded=False):
            move_col1, move_col2 = st.columns([0.5, 0.5])
            with move_col1:
                cur_idx = all_folders.index(folder) if folder in all_folders else 0
                new_choice = st.selectbox("Move to folder", options=all_folders + ["+ Create New Folder..."], index=cur_idx, key=f"move_folder_sel_{doc_id}")
                if new_choice == "+ Create New Folder...":
                    inline_folder = st.text_input("New folder name", key=f"inline_f_{doc_id}").strip()
                    target_move_folder = inline_folder or folder
                else:
                    target_move_folder = new_choice
            with move_col2:
                new_desc_input = st.text_input("Description", value=clean_desc, key=f"edit_desc_{doc_id}")
            if st.button("Save changes", icon=":material/save:", key=f"save_meta_{doc_id}"):
                if update_document_metadata(doc_id, target_move_folder, new_desc_input):
                    st.success(f"Updated `{filename}` to folder `{target_move_folder}`!")
                    if target_move_folder not in st.session_state.get("custom_folders", []):
                        st.session_state["custom_folders"].append(target_move_folder)
                    st.rerun()

        if clean_desc:
            st.info(f"**Description:** {clean_desc}", icon=":material/description:")

        st_markdown("#### Indexed document chunks")
        chunks = fetch_document_chunks(doc_id)
        if chunks:
            st.caption(f"Showing {len(chunks)} chunks retrieved from Neo4j vector store:")
            for chunk in chunks:
                c_idx = chunk.get("chunk_index", 0)
                c_content = chunk.get("content", "")
                c_source = chunk.get("source", "")
                label = f"Chunk #{c_idx + 1} ({len(c_content)} chars)" + (f" — {c_source}" if c_source and c_source != filename else "")
                with st.expander(label, icon=":material/segment:", expanded=(c_idx == 0)):
                    tab_fmt, tab_raw = st.tabs(["Formatted View", "Raw Content"])
                    with tab_fmt:
                        st_markdown(c_content)
                    with tab_raw:
                        st.text_area(label=f"Raw Chunk {c_idx + 1}", value=c_content, height=120, disabled=True, key=f"chunk_raw_{doc_id}_{c_idx}", label_visibility="collapsed")
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

def render_explorer_view(docs: list[dict], all_folders: list[str]) -> None:
    """Args: docs: Ingested document records, all_folders: Known folder names."""
    if not docs:
        st.info("Your document library is currently empty. Upload files in the 'Upload documents' tab.", icon=":material/folder_open:")
        return

    enriched_docs = [{**doc, "_folder": parse_folder(doc.get("description", ""))[0], "_clean_desc": parse_folder(doc.get("description", ""))[1]} for doc in docs]
    tcol1, tcol2, tcol3, tcol4 = st.columns([0.42, 0.25, 0.2, 0.13], vertical_alignment="center")

    with tcol1:
        search_term = st.text_input("Search documents", placeholder="Search by name, folder, or description...", label_visibility="collapsed").strip().lower()
    with tcol2:
        group_by = st.selectbox("Group by", options=["Folder / Category", "File Extension", "Uploaded By"], label_visibility="collapsed")
    with tcol3:
        view_mode = st.segmented_control("View mode", options=["Table", "Grid"], default="Table", label_visibility="collapsed") or "Table"
    with tcol4:
        if st.button("Refresh", icon=":material/refresh:", width="stretch"):
            st.rerun()

    filtered_docs = enriched_docs
    if search_term:
        filtered_docs = [
            d for d in enriched_docs
            if search_term in d.get("filename", "").lower()
            or search_term in d.get("_clean_desc", "").lower()
            or search_term in d.get("_folder", "").lower()
        ]
        st.caption(f"Showing {len(filtered_docs)} of {len(enriched_docs)} documents matching '`{search_term}`'")

    selected_doc_id = st.session_state.get("selected_doc_id")
    selected_doc = next((d for d in enriched_docs if d.get("id") == selected_doc_id), None)
    if selected_doc:
        render_file_inspector(selected_doc, selected_doc.get("_folder", "Root"), selected_doc.get("_clean_desc", ""), all_folders)

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

    for folder_name, folder_files in sorted(groups.items()):
        total_chunks = sum(f.get("chunk_count", 0) for f in folder_files)
        with st.expander(f"{folder_name} · `{len(folder_files)} file(s)` · `{total_chunks} chunk(s)`", icon=":material/folder:", expanded=True):
            f_col_info, f_col_del = st.columns([0.72, 0.28], vertical_alignment="center")
            with f_col_info:
                st.caption(f"**{len(folder_files)} file(s)** inside **{folder_name}**")
            with f_col_del:
                with st.popover(f"Clear folder ({len(folder_files)})", icon=":material/delete:", help=f"Delete all {len(folder_files)} docs in {folder_name}"):
                    st_markdown(f"**Delete all files in `{folder_name}`?**")
                    st.caption("This will permanently remove all files and vector embeddings in Neo4j.")
                    if st.button("Confirm delete all", key=f"conf_del_all_{folder_name}", type="primary"):
                        succ, _ = delete_all_in_folder(folder_files)
                        st.success(f"Deleted {succ} file(s) from `{folder_name}`")
                        st.rerun()

            if view_mode == "Grid":
                cols = st.columns(3)
                for idx, doc in enumerate(folder_files):
                    filename, doc_id = doc.get("filename", "Unknown"), doc.get("id", "")
                    ext = (doc.get("file_type") or "").lower()
                    file_info = FILE_TYPE_INFO.get(ext, {"icon": "📄", "label": ext.upper()})
                    clean_desc, upload_date = doc.get("_clean_desc", ""), (doc.get("upload_date") or "")[:10]
                    with cols[idx % 3]:
                        with st.container(border=True):
                            st_markdown(f"#### {file_info['icon']} {filename}")
                            st.caption(f"**Type:** `{ext}` | **Chunks:** `{doc.get('chunk_count', 0)}`")
                            if clean_desc:
                                st.caption(f"📝 {clean_desc[:55]}..." if len(clean_desc) > 55 else f"📝 {clean_desc}")
                            if upload_date:
                                st.caption(f"📅 {upload_date}")
                            bcol1, bcol2 = st.columns([0.55, 0.45])
                            with bcol1:
                                if st.button("Inspect", icon=":material/visibility:", key=f"inspect_{doc_id}", width="stretch"):
                                    st.session_state["selected_doc_id"] = doc_id
                                    st.rerun()
                            with bcol2:
                                with st.popover("Delete", icon=":material/delete:", key=f"pop_del_{doc_id}"):
                                    st.caption(f"Delete `{filename}`?")
                                    if st.button("Confirm", key=f"del_grid_{doc_id}", type="primary") and delete_document(doc_id):
                                        st.success(f"Deleted `{filename}`")
                                        if st.session_state.get("selected_doc_id") == doc_id:
                                            st.session_state["selected_doc_id"] = None
                                        st.rerun()
            else:
                table_data = [
                    {
                        "id": doc.get("id"),
                        "Icon": FILE_TYPE_INFO.get((doc.get("file_type") or "").lower(), {}).get("icon", "📄"),
                        "Filename": doc.get("filename"),
                        "Type": (doc.get("file_type") or "").upper(),
                        "Folder": doc.get("_folder", "Root"),
                        "Chunks": doc.get("chunk_count", 0),
                        "Uploaded By": doc.get("user_id", "default"),
                        "Date": (doc.get("upload_date") or "")[:19].replace("T", " "),
                        "Description": doc.get("_clean_desc", ""),
                    }
                    for doc in folder_files
                ]
                st.dataframe(
                    table_data, hide_index=True,
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
                selected_row = st.selectbox(f"Select file from {folder_name} to inspect or manage", options=[d["Filename"] for d in table_data], key=f"select_table_{folder_name}")
                if selected_row:
                    matched = next((d for d in folder_files if d.get("filename") == selected_row), None)
                    if matched:
                        act1, act2 = st.columns([0.3, 0.7])
                        with act1:
                            if st.button("Inspect file", icon=":material/visibility:", key=f"inspect_tbl_{matched['id']}"):
                                st.session_state["selected_doc_id"] = matched["id"]
                                st.rerun()
                        with act2:
                            with st.popover("Delete file", icon=":material/delete:", key=f"pop_tbl_{matched['id']}"):
                                st.caption(f"Permanently delete `{selected_row}`?")
                                if st.button("Confirm delete", key=f"del_tbl_{matched['id']}", type="primary") and delete_document(matched["id"]):
                                    st.success(f"Deleted `{selected_row}`")
                                    st.rerun()

st.html(
    """
    <div style="text-align: center; padding: 12px 0 8px;">
        <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;">
            Document File Explorer
        </h1>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
            Organize, browse, and inspect your unstructured document library and vector embeddings in Neo4j.
        </p>
    </div>
    """
)

user_id = st.session_state.get("user_name", "default")
docs = fetch_documents()
all_folders = get_all_folders(docs)

total_docs = len(docs)
total_chunks = sum(d.get("chunk_count", 0) for d in docs)
pdf_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "pdf")
docx_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "docx")
txt_count = sum(1 for d in docs if (d.get("file_type") or "").lower() in ("txt", "md"))

kpi_cols = st.columns(5)
for col, (lbl, val) in zip(kpi_cols, [
    ("Total files", f"{total_docs}"),
    ("Total chunks", f"{total_chunks:,}"),
    ("Folders", f"{len(all_folders)}"),
    ("PDF files", f"{pdf_count}"),
    ("Word & text", f"{docx_count + txt_count}"),
]):
    with col:
        with st.container(border=True):
            st.metric(lbl, val)

tab_explorer, tab_upload, tab_folders = st.tabs(["📁 Browse files", "📤 Upload documents", "📂 Manage folders"])
with tab_explorer:
    render_explorer_view(docs, all_folders)
with tab_upload:
    render_upload_modal(user_id=user_id, docs=docs)
with tab_folders:
    render_manage_folders_section(docs)
