"""
doc_ingestion.py — Unstructured Document File Explorer & Ingestion
-------------------------------------------------------------------
Streamlit page featuring a File Explorer interface for uploading, browsing,
previewing, and managing unstructured documents (PDF, DOCX, TXT, Markdown)
stored in the Neo4j knowledge graph with customizable destination folders.
"""

import requests
import streamlit as st
from utils.ui_utils import BACKEND_URL, clear_api_cache

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------
INGEST_DOC_URL = f"{BACKEND_URL}/ingest/documents"
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]

FILE_TYPE_INFO = {
    "pdf": {"icon": ":material/picture_as_pdf:", "label": "PDF document", "color": "#EF4444"},
    "docx": {"icon": ":material/description:", "label": "Word document", "color": "#3B82F6"},
    "txt": {"icon": ":material/article:", "label": "Text file", "color": "#10B981"},
    "md": {"icon": ":material/markdown:", "label": "Markdown file", "color": "#8B5CF6"},
}

INITIAL_FOLDERS = ["Root", "Specifications", "Guides & manuals", "Research", "Notes", "General"]


def _fetch_documents() -> list[dict]:
    """Fetch the list of all ingested documents from the backend."""
    try:
        resp = requests.get(INGEST_DOC_URL, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception as exc:
        st.error(f"Failed to fetch document list: {exc}", icon=":material/error:")
        return []


def _fetch_document_chunks(doc_id: str) -> list[dict]:
    """Fetch chunks for a specific document."""
    try:
        resp = requests.get(f"{INGEST_DOC_URL}/{doc_id}/chunks", timeout=10)
        resp.raise_for_status()
        return resp.json().get("chunks", [])
    except Exception:
        return []


def _upload_file(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    description: str,
    folder: str,
    force: bool = False,
) -> dict:
    """POST a file to backend with folder metadata encoded in description."""
    folder_prefix = f"[{folder}] " if folder and folder != "Root" else ""
    full_description = f"{folder_prefix}{description}".strip()

    resp = requests.post(
        INGEST_DOC_URL,
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={
            "user_id": user_id,
            "description": full_description,
            "force": str(force).lower(),
        },
        timeout=120,
    )
    resp.raise_for_status()
    clear_api_cache()
    return resp.json()


def _update_document_metadata(doc_id: str, folder: str, description: str) -> bool:
    """Update folder/description for a document."""
    folder_prefix = f"[{folder}] " if folder and folder != "Root" else ""
    full_description = f"{folder_prefix}{description}".strip()
    try:
        resp = requests.put(
            f"{INGEST_DOC_URL}/{doc_id}",
            json={"description": full_description},
            timeout=10,
        )
        resp.raise_for_status()
        clear_api_cache()
        return resp.json().get("status") == "success"
    except Exception as exc:
        st.error(f"Failed to update document: {exc}", icon=":material/error:")
        return False


def _delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks from Neo4j."""
    try:
        resp = requests.delete(f"{INGEST_DOC_URL}/{doc_id}", timeout=10)
        resp.raise_for_status()
        clear_api_cache()
        return resp.json().get("status") == "success"
    except Exception as exc:
        st.error(f"Failed to delete document: {exc}", icon=":material/error:")
        return False


def _delete_all_in_folder(files: list[dict]) -> tuple[int, int]:
    """Delete all documents in a given list of files. Returns (succeeded, failed)."""
    succeeded, failed = 0, 0
    for doc in files:
        doc_id = doc.get("id")
        if doc_id:
            if _delete_document(doc_id):
                succeeded += 1
            else:
                failed += 1
    clear_api_cache()
    return succeeded, failed


def _parse_folder(description: str) -> tuple[str, str]:
    """Extract folder name from description if prefixed like '[Folder] rest of desc'."""
    desc = (description or "").strip()
    if desc.startswith("[") and "]" in desc:
        end_idx = desc.index("]")
        folder = desc[1:end_idx].strip()
        clean_desc = desc[end_idx + 1 :].strip()
        return folder or "Root", clean_desc
    return "Root", desc


def _get_all_folders(docs: list[dict]) -> list[str]:
    """Return sorted unique list of all folders (discovered + custom added)."""
    if "custom_folders" not in st.session_state:
        st.session_state["custom_folders"] = list(INITIAL_FOLDERS)

    discovered = {f for doc in docs for f, _ in [_parse_folder(doc.get("description", ""))]}
    all_f = set(st.session_state["custom_folders"]).union(discovered)
    if "Root" not in all_f:
        all_f.add("Root")

    others = sorted([f for f in all_f if f != "Root"])
    return ["Root"] + others


# ---------------------------------------------------------------------------
# Upload Section
# ---------------------------------------------------------------------------
def _render_upload_section(user_id: str, docs: list[dict]) -> None:
    st.subheader("Upload new documents")
    st.caption("Upload documents to chunk, embed, and index into your Neo4j knowledge graph.")

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

    all_folders = _get_all_folders(docs)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        uploaded_files = st.file_uploader(
            "Select files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, Markdown",
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
        folder_options = all_folders + ["+ Create new folder..."]
        selected_folder_opt = st.selectbox(
            "Destination folder",
            options=folder_options,
            index=0,
            key="upload_folder_selector",
        )

        target_folder = selected_folder_opt
        if selected_folder_opt == "+ Create new folder...":
            new_folder_input = st.text_input(
                "Name of new folder",
                placeholder="e.g. Architecture specs, Client proposals",
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

        st.caption(f"Files will be saved under: **`{target_folder}`**")

    if uploaded_files and st.button("Ingest selected files", icon=":material/upload:", type="primary"):
        total = len(uploaded_files)
        progress_bar = st.progress(0.0, text="Starting document ingestion...")
        succeeded, skipped, failed = 0, 0, 0

        for idx, file in enumerate(uploaded_files):
            ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
            file_info = FILE_TYPE_INFO.get(ext, {"icon": ":material/article:"})
            icon = file_info["icon"]

            progress_bar.progress(idx / total, text=f"Processing `{file.name}`...")

            with st.status(f"Ingesting `{file.name}` into folder `{target_folder}`...", expanded=False) as status:
                try:
                    file_bytes = file.read()
                    result = _upload_file(
                        file_bytes=file_bytes,
                        filename=file.name,
                        user_id=user_id,
                        description=description,
                        folder=target_folder or "Root",
                        force=overwrite,
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
            flash = {"type": "success", "message": f"Ingested {total} file(s) into folder `{target_folder}`."}
        elif skipped == total:
            flash = {"type": "info", "message": f"All {total} file(s) were already ingested. Duplicate ingestion skipped."}
        elif failed == 0:
            flash = {"type": "success", "message": f"Processed {total} file(s): {succeeded} ingested, {skipped} duplicate(s) skipped."}
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
def _render_manage_folders_section(docs: list[dict]) -> None:
    st.subheader("Manage destination folders")
    st.caption("Add, inspect, or remove custom destination folders for organizing your documents.")

    all_folders = _get_all_folders(docs)
    fcol1, fcol2 = st.columns([0.6, 0.4])

    with fcol1:
        st.markdown("#### Existing folders")
        for folder in all_folders:
            folder_docs = [d for d in docs if _parse_folder(d.get("description", ""))[0] == folder]
            count = len(folder_docs)
            chunks = sum(d.get("chunk_count", 0) for d in folder_docs)

            with st.container(border=True):
                rcol1, rcol2 = st.columns([0.65, 0.35])
                with rcol1:
                    st.markdown(f"**`{folder}`** &nbsp;·&nbsp; `{count} file(s)` &nbsp;·&nbsp; `{chunks} chunk(s)`")
                with rcol2:
                    if count > 0:
                        with st.popover(f"Clear ({count})", icon=":material/delete:", help=f"Permanently delete all {count} files in '{folder}'"):
                            st.markdown(f"**Clear all {count} files in `{folder}`?**")
                            st.caption("This will remove all documents and their vector embeddings.")
                            if st.button("Confirm clear", key=f"conf_clear_mgr_{folder}", type="primary"):
                                succ, _ = _delete_all_in_folder(folder_docs)
                                st.success(f"Deleted {succ} file(s) from `{folder}`")
                                st.rerun()
                    elif folder != "Root":
                        if st.button("Remove folder", icon=":material/delete:", key=f"del_empty_folder_{folder}"):
                            if folder in st.session_state["custom_folders"]:
                                st.session_state["custom_folders"].remove(folder)
                                st.success(f"Removed empty folder `{folder}`")
                                st.rerun()
                    else:
                        st.caption("Default root")

    with fcol2:
        st.markdown("#### Add new folder")
        with st.form("create_folder_form", clear_on_submit=True):
            new_f_name = st.text_input("Folder name", placeholder="e.g. Legal, Project Alpha")
            submit_btn = st.form_submit_button("Create folder", icon=":material/create_new_folder:", type="primary")

            if submit_btn and new_f_name.strip():
                clean_name = new_f_name.strip()
                if clean_name not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(clean_name)
                    st.success(f"Folder `{clean_name}` created.")
                    st.rerun()
                else:
                    st.info(f"Folder `{clean_name}` already exists.")


# ---------------------------------------------------------------------------
# File Inspector Drawer
# ---------------------------------------------------------------------------
def _render_file_inspector(doc: dict, folder: str, clean_desc: str, all_folders: list[str]) -> None:
    filename = doc.get("filename", "Unknown")
    doc_id = doc.get("id", "")
    ext = (doc.get("file_type") or "").lower()
    file_info = FILE_TYPE_INFO.get(ext, {"icon": ":material/article:", "label": ext.upper(), "color": "#64748B"})
    upload_date = (doc.get("upload_date") or "")[:19].replace("T", " ")
    chunk_count = doc.get("chunk_count", 0)
    user_id = doc.get("user_id") or "default"

    with st.container(border=True):
        col_title, col_close = st.columns([0.85, 0.15])
        with col_title:
            st.markdown(f"### {filename}")
            st.caption(f"Current folder: **`{folder}`** &nbsp;|&nbsp; ID: `{doc_id}`")
        with col_close:
            if st.button("Close", icon=":material/close:", key=f"close_inspect_{doc_id}"):
                st.session_state["selected_doc_id"] = None
                st.rerun()

        st.divider()

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Format", file_info["label"])
        with mcol2:
            st.metric("Total chunks", f"{chunk_count}")
        with mcol3:
            st.metric("Uploaded by", user_id)
        with mcol4:
            st.metric("Uploaded at", upload_date or "N/A")

        with st.expander("Edit folder & description", expanded=False):
            move_col1, move_col2 = st.columns([0.5, 0.5])
            with move_col1:
                cur_idx = all_folders.index(folder) if folder in all_folders else 0
                new_folder_choice = st.selectbox(
                    "Move to folder",
                    options=all_folders + ["+ Create new folder..."],
                    index=cur_idx,
                    key=f"move_folder_sel_{doc_id}",
                )
                if new_folder_choice == "+ Create new folder...":
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
                if _update_document_metadata(doc_id, target_move_folder, new_desc_input):
                    st.success(f"Updated `{filename}` to folder `{target_move_folder}`.")
                    if target_move_folder not in st.session_state.get("custom_folders", []):
                        st.session_state["custom_folders"].append(target_move_folder)
                    st.rerun()

        if clean_desc:
            st.info(f"**Description:** {clean_desc}", icon=":material/description:")

        st.markdown("#### Indexed document chunks")
        chunks = _fetch_document_chunks(doc_id)

        if chunks:
            st.caption(f"Showing {len(chunks)} chunks retrieved from Neo4j vector store:")
            for chunk in chunks:
                c_idx = chunk.get("chunk_index", 0)
                c_content = chunk.get("content", "")
                with st.expander(f"Chunk #{c_idx + 1} ({len(c_content)} chars)", expanded=(c_idx == 0)):
                    st.text_area(
                        label=f"Chunk Content {c_idx + 1}",
                        value=c_content,
                        height=120,
                        disabled=True,
                        key=f"chunk_text_{doc_id}_{c_idx}",
                        label_visibility="collapsed",
                    )
        else:
            st.caption("No individual chunk records found.")

        st.divider()
        with st.popover("Delete document & chunks", icon=":material/delete:"):
            st.caption(f"Permanently delete '{filename}' and all its embeddings?")
            if st.button("Confirm deletion", key=f"del_inspect_{doc_id}", type="primary"):
                if _delete_document(doc_id):
                    st.success(f"Deleted `{filename}`.")
                    st.session_state["selected_doc_id"] = None
                    st.rerun()


# ---------------------------------------------------------------------------
# File Explorer View
# ---------------------------------------------------------------------------
def _render_explorer_view(docs: list[dict], all_folders: list[str]) -> None:
    if not docs:
        st.info("Your document library is currently empty. Upload files in the 'Upload documents' tab.", icon=":material/folder_open:")
        return

    enriched_docs = []
    for doc in docs:
        folder, clean_desc = _parse_folder(doc.get("description", ""))
        enriched_docs.append({**doc, "_folder": folder, "_clean_desc": clean_desc})

    tcol1, tcol2, tcol3, tcol4 = st.columns([0.4, 0.25, 0.2, 0.15])

    with tcol1:
        search_term = st.text_input(
            "Search documents",
            placeholder="Search by name, folder, or description...",
            label_visibility="collapsed",
        ).strip().lower()

    with tcol2:
        group_by = st.selectbox(
            "Group by",
            options=["Folder / Category", "File extension", "Uploaded by"],
            label_visibility="collapsed",
        )

    with tcol3:
        view_mode = st.segmented_control(
            "View mode",
            options=["Grid", "Table"],
            default="Grid",
            label_visibility="collapsed",
        )

    with tcol4:
        if st.button("Refresh", icon=":material/refresh:"):
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
    selected_doc_match = next((d for d in enriched_docs if d.get("id") == selected_doc_id), None)

    if selected_doc_match:
        _render_file_inspector(
            selected_doc_match,
            selected_doc_match.get("_folder", "Root"),
            selected_doc_match.get("_clean_desc", ""),
            all_folders,
        )
        st.divider()

    groups: dict[str, list[dict]] = {}
    for doc in filtered_docs:
        if group_by == "Folder / Category":
            key = f"Folder: {doc.get('_folder', 'Root')}"
        elif group_by == "File extension":
            ext = (doc.get("file_type") or "other").upper()
            key = f"{ext} files"
        else:
            key = f"Uploaded by: {doc.get('user_id') or 'default'}"

        groups.setdefault(key, []).append(doc)

    for folder_name, folder_files in sorted(groups.items()):
        total_chunks_in_folder = sum(f.get("chunk_count", 0) for f in folder_files)
        folder_header = f"{folder_name} · {len(folder_files)} file(s) · {total_chunks_in_folder} chunk(s)"

        with st.expander(folder_header, expanded=True):
            f_col_info, f_col_del = st.columns([0.7, 0.3])
            with f_col_info:
                st.caption(f"Showing **{len(folder_files)} file(s)** inside **{folder_name}**")
            with f_col_del:
                with st.popover(f"Clear folder ({len(folder_files)})", icon=":material/delete:"):
                    st.markdown(f"**Delete all content in `{folder_name}`?**")
                    st.caption("This will permanently remove all files and their vector embeddings in Neo4j.")
                    if st.button("Confirm delete all", key=f"conf_del_all_{folder_name}", type="primary"):
                        succ, _ = _delete_all_in_folder(folder_files)
                        st.success(f"Deleted {succ} file(s) from `{folder_name}`.")
                        st.rerun()

            st.markdown("---")

            if view_mode == "Grid" or not view_mode:
                cols = st.columns(3)
                for idx, doc in enumerate(folder_files):
                    col = cols[idx % 3]
                    filename = doc.get("filename", "Unknown")
                    doc_id = doc.get("id", "")
                    ext = (doc.get("file_type") or "").lower()
                    chunk_count = doc.get("chunk_count", 0)
                    upload_date = (doc.get("upload_date") or "")[:10]
                    clean_desc = doc.get("_clean_desc", "")

                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### {filename}")
                            st.caption(f"**Type:** `{ext}` | **Chunks:** `{chunk_count}`")
                            if clean_desc:
                                st.caption(f"{clean_desc[:60]}..." if len(clean_desc) > 60 else f"{clean_desc}")
                            if upload_date:
                                st.caption(f"Date: {upload_date}")

                            btn_col1, btn_col2 = st.columns([0.6, 0.4])
                            with btn_col1:
                                if st.button("Inspect", icon=":material/search:", key=f"inspect_{doc_id}"):
                                    st.session_state["selected_doc_id"] = doc_id
                                    st.rerun()
                            with btn_col2:
                                with st.popover("", icon=":material/delete:", help="Delete document"):
                                    st.caption(f"Delete '{filename}'?")
                                    if st.button("Confirm", key=f"del_grid_{doc_id}", type="primary"):
                                        if _delete_document(doc_id):
                                            st.success(f"Deleted `{filename}`.")
                                            if st.session_state.get("selected_doc_id") == doc_id:
                                                st.session_state["selected_doc_id"] = None
                                            st.rerun()

            else:
                table_data = []
                for doc in folder_files:
                    ext = (doc.get("file_type") or "").lower()
                    table_data.append({
                        "id": doc.get("id"),
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
                        "Filename": st.column_config.TextColumn("File name", width="medium"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Folder": st.column_config.TextColumn("Folder", width="small"),
                        "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
                        "Uploaded By": st.column_config.TextColumn("Author"),
                        "Date": st.column_config.TextColumn("Upload date"),
                        "Description": st.column_config.TextColumn("Description", width="large"),
                    },
                )


# ---------------------------------------------------------------------------
# Page Entry
# ---------------------------------------------------------------------------
st.header("Document Knowledge Base")
st.caption("Organize, browse, and inspect your unstructured document library and vector embeddings in Neo4j.")

user_id = st.session_state.get("user_name", "default")
docs = _fetch_documents()
all_folders = _get_all_folders(docs)

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
        st.metric("PDFs", f"{pdf_count}")
with kpi5:
    with st.container(border=True):
        st.metric("Word & text", f"{docx_count + txt_count}")

st.divider()

tab_explorer, tab_upload, tab_folders = st.tabs([
    "Browse files & folders",
    "Upload documents",
    "Manage folders",
])

with tab_explorer:
    _render_explorer_view(docs, all_folders)

with tab_upload:
    _render_upload_section(user_id=user_id, docs=docs)

with tab_folders:
    _render_manage_folders_section(docs)
