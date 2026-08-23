"""
doc_injestion.py — Unstructured Document File Explorer & Ingestion
-------------------------------------------------------------------
Streamlit page featuring a File Explorer interface for uploading, browsing,
previewing, and managing unstructured documents (PDF, DOCX, TXT, Markdown)
stored in the Neo4j knowledge graph with customizable destination folders.
"""

import requests
import streamlit as st
from utils.ui_utils import BACKEND_URL

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
# Constants & Helpers
# ---------------------------------------------------------------------------
INGEST_DOC_URL = f"{BACKEND_URL}/ingest/documents"
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]

FILE_TYPE_INFO = {
    "pdf": {"icon": "📕", "label": "PDF Document", "color": "#EF4444"},
    "docx": {"icon": "📘", "label": "Word Document", "color": "#3B82F6"},
    "txt": {"icon": "📄", "label": "Text File", "color": "#10B981"},
    "md": {"icon": "📝", "label": "Markdown File", "color": "#8B5CF6"},
}

INITIAL_FOLDERS = ["Root", "Specifications", "Guides & Manuals", "Research", "Notes", "General"]


def _fetch_documents() -> list[dict]:
    """Fetch the list of all ingested documents from the backend."""
    try:
        resp = requests.get(INGEST_DOC_URL, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception as exc:
        st.error(f"Failed to fetch document list: {exc}")
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
) -> dict:
    """POST a file to backend with folder metadata encoded in description."""
    folder_prefix = f"[{folder}] " if folder and folder != "Root" else ""
    full_description = f"{folder_prefix}{description}".strip()

    resp = requests.post(
        INGEST_DOC_URL,
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={"user_id": user_id, "description": full_description},
        timeout=120,
    )
    resp.raise_for_status()
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
        return resp.json().get("status") == "success"
    except Exception as exc:
        st.error(f"Failed to update document: {exc}")
        return False


def _delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks from Neo4j."""
    try:
        resp = requests.delete(f"{INGEST_DOC_URL}/{doc_id}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as exc:
        st.error(f"Failed to delete document: {exc}")
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

    # Collect folders discovered from existing documents
    discovered = {f for doc in docs for f, _ in [_parse_folder(doc.get("description", ""))]}

    # Union with session custom folders
    all_f = set(st.session_state["custom_folders"]).union(discovered)
    if "Root" not in all_f:
        all_f.add("Root")

    # Return with 'Root' first, then alphabetical
    others = sorted([f for f in all_f if f != "Root"])
    return ["Root"] + others


# ---------------------------------------------------------------------------
# Upload Section with Customizable Folders
# ---------------------------------------------------------------------------
def _render_upload_modal(user_id: str, docs: list[dict]) -> None:
    st.subheader("📤 Upload New Documents")
    st.caption("Upload documents to chunk, embed, and index into your Neo4j knowledge graph.")

    all_folders = _get_all_folders(docs)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        uploaded_files = st.file_uploader(
            "Select files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, Markdown",
        )

    with col2:
        folder_options = all_folders + ["➕ Create New Folder..."]
        selected_folder_opt = st.selectbox(
            "📁 Destination Folder",
            options=folder_options,
            index=0,
            key="upload_folder_selector",
        )

        target_folder = selected_folder_opt
        if selected_folder_opt == "➕ Create New Folder...":
            new_folder_input = st.text_input(
                "Name of New Folder",
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
            "📝 Document Description (optional)",
            placeholder="e.g. Q3 API design spec",
            key="upload_doc_desc",
        )

        st.caption(f"Files will be saved under: 📂 **`{target_folder}`**")

    if uploaded_files and st.button("⬆️ Ingest Selected Files", type="primary"):
        total = len(uploaded_files)
        progress_bar = st.progress(0.0, text="Starting document ingestion...")
        succeeded, failed = 0, 0

        for idx, file in enumerate(uploaded_files):
            ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
            file_info = FILE_TYPE_INFO.get(ext, {"icon": "📄"})
            icon = file_info["icon"]

            progress_bar.progress(idx / total, text=f"{icon} Processing `{file.name}`...")

            with st.status(f"{icon} Ingesting `{file.name}` into folder `{target_folder}`...", expanded=False) as status:
                try:
                    file_bytes = file.read()
                    result = _upload_file(
                        file_bytes=file_bytes,
                        filename=file.name,
                        user_id=user_id,
                        description=description,
                        folder=target_folder or "Root",
                    )
                    if result.get("status") == "success":
                        chunk_count = result.get("chunk_count", 0)
                        status.update(
                            label=f"✅ `{file.name}` ({chunk_count} chunks stored in `{target_folder}`)",
                            state="complete",
                        )
                        succeeded += 1
                    else:
                        status.update(label=f"❌ `{file.name}` — {result.get('message')}", state="error")
                        failed += 1
                except Exception as exc:
                    status.update(label=f"❌ `{file.name}` — {str(exc)[:150]}", state="error")
                    failed += 1

            progress_bar.progress((idx + 1) / total)

        progress_bar.empty()

        if succeeded == total:
            st.success(f"🎉 Ingested {total} file(s) into folder `{target_folder}`!", icon="✅")
        elif succeeded > 0:
            st.warning(f"⚠️ {succeeded}/{total} files succeeded, {failed} failed.")
        else:
            st.error("❌ File ingestion failed. Check backend logs.")
        st.rerun()


# ---------------------------------------------------------------------------
# Manage Custom Folders Section
# ---------------------------------------------------------------------------
def _render_manage_folders_section(docs: list[dict]) -> None:
    st.subheader("📁 Manage Custom Destination Folders")
    st.caption("Add, inspect, or remove custom destination folders for organizing your documents.")

    all_folders = _get_all_folders(docs)

    fcol1, fcol2 = st.columns([0.6, 0.4])

    with fcol1:
        st.markdown("#### 📂 Existing Folders")
        for folder in all_folders:
            folder_docs = [d for d in docs if _parse_folder(d.get("description", ""))[0] == folder]
            count = len(folder_docs)
            chunks = sum(d.get("chunk_count", 0) for d in folder_docs)

            with st.container(border=True):
                rcol1, rcol2 = st.columns([0.65, 0.35])
                with rcol1:
                    st.markdown(f"**📁 `{folder}`** &nbsp;·&nbsp; `{count} file(s)` &nbsp;·&nbsp; `{chunks} chunk(s)`")
                with rcol2:
                    if count > 0:
                        if hasattr(st, "popover"):
                            with st.popover(f"🗑️ Clear ({count})", help=f"Permanently delete all {count} files in '{folder}'"):
                                st.markdown(f"**Clear all {count} files in `{folder}`?**")
                                st.caption("This will remove all documents and their vector embeddings.")
                                if st.button("⚠️ Confirm Clear", key=f"conf_clear_mgr_{folder}", type="primary"):
                                    succ, _ = _delete_all_in_folder(folder_docs)
                                    st.success(f"Deleted {succ} file(s) from `{folder}`")
                                    st.rerun()
                        else:
                            if st.button(f"🗑️ Clear ({count})", key=f"clear_mgr_{folder}"):
                                succ, _ = _delete_all_in_folder(folder_docs)
                                st.success(f"Deleted {succ} file(s) from `{folder}`")
                                st.rerun()
                    elif folder != "Root":
                        if st.button("🗑️ Remove", key=f"del_empty_folder_{folder}", help="Delete this empty folder"):
                            if folder in st.session_state["custom_folders"]:
                                st.session_state["custom_folders"].remove(folder)
                                st.success(f"Removed empty folder `{folder}`")
                                st.rerun()
                    else:
                        st.caption("Default root")

    with fcol2:
        st.markdown("#### ➕ Add New Folder")
        with st.form("create_folder_form", clear_on_submit=True):
            new_f_name = st.text_input("Folder Name", placeholder="e.g. Legal, Project Alpha")
            submit_btn = st.form_submit_button("Create Folder", type="primary")

            if submit_btn and new_f_name.strip():
                clean_name = new_f_name.strip()
                if clean_name not in st.session_state["custom_folders"]:
                    st.session_state["custom_folders"].append(clean_name)
                    st.success(f"Folder `📁 {clean_name}` created!")
                    st.rerun()
                else:
                    st.info(f"Folder `{clean_name}` already exists.")


# ---------------------------------------------------------------------------
# File Inspector Drawer with Move Folder capability
# ---------------------------------------------------------------------------
def _render_file_inspector(doc: dict, folder: str, clean_desc: str, all_folders: list[str]) -> None:
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
            st.markdown(f"### {file_info['icon']} {filename}")
            st.caption(f"📂 Current Folder: **`{folder}`** &nbsp;|&nbsp; ID: `{doc_id}`")
        with col_close:
            if st.button("✖ Close", key=f"close_inspect_{doc_id}"):
                st.session_state["selected_doc_id"] = None
                st.rerun()

        st.divider()

        # Metadata metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Format", file_info["label"])
        with mcol2:
            st.metric("Total Chunks", f"{chunk_count}")
        with mcol3:
            st.metric("Uploaded By", user_id)
        with mcol4:
            st.metric("Uploaded At", upload_date or "N/A")

        # Edit Folder / Move Location & Description
        with st.expander("✏️ Edit Folder & Description", expanded=False):
            move_col1, move_col2 = st.columns([0.5, 0.5])
            with move_col1:
                cur_idx = all_folders.index(folder) if folder in all_folders else 0
                new_folder_choice = st.selectbox(
                    "Move to Folder",
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

            if st.button("💾 Save Changes", key=f"save_meta_{doc_id}"):
                if _update_document_metadata(doc_id, target_move_folder, new_desc_input):
                    st.success(f"Updated `{filename}` to folder `{target_move_folder}`!")
                    if target_move_folder not in st.session_state.get("custom_folders", []):
                        st.session_state["custom_folders"].append(target_move_folder)
                    st.rerun()

        if clean_desc:
            st.info(f"**Description:** {clean_desc}", icon="📝")

        # Chunk Preview Section
        st.markdown("#### 🧩 Indexed Document Chunks")
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
            st.caption("No individual chunk records found or vector embedding in progress.")

        st.divider()
        if st.button("🗑️ Delete Document & Chunks", key=f"del_inspect_{doc_id}", type="primary"):
            if _delete_document(doc_id):
                st.success(f"Deleted `{filename}`")
                st.session_state["selected_doc_id"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# File Explorer View
# ---------------------------------------------------------------------------
def _render_explorer_view(docs: list[dict], all_folders: list[str]) -> None:
    if not docs:
        st.info("📂 Your document library is currently empty. Upload files in the 'Upload Documents' tab.", icon="📭")
        return

    enriched_docs = []
    for doc in docs:
        folder, clean_desc = _parse_folder(doc.get("description", ""))
        enriched_docs.append({**doc, "_folder": folder, "_clean_desc": clean_desc})

    # Toolbar: Search, Grouping, View Mode, Refresh
    tcol1, tcol2, tcol3, tcol4 = st.columns([0.4, 0.25, 0.2, 0.15])

    with tcol1:
        search_term = st.text_input(
            "Search documents",
            placeholder="🔍 Search by name, folder, or description...",
            label_visibility="collapsed",
        ).strip().lower()

    with tcol2:
        group_by = st.selectbox(
            "Group by",
            options=["📁 Folder / Category", "🏷️ File Extension", "👤 Uploaded By"],
            label_visibility="collapsed",
        )

    with tcol3:
        view_mode = st.radio(
            "View Mode",
            options=["🗂️ Grid", "📋 Table"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with tcol4:
        if st.button("🔄 Refresh"):
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

    # Inspector check
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

    # Group documents
    groups: dict[str, list[dict]] = {}
    for doc in filtered_docs:
        if group_by == "📁 Folder / Category":
            key = f"📁 {doc.get('_folder', 'Root')}"
        elif group_by == "🏷️ File Extension":
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

        with st.expander(folder_header, expanded=True):
            # Folder Action Bar
            f_col_info, f_col_del = st.columns([0.7, 0.3])
            with f_col_info:
                st.caption(f"Showing **{len(folder_files)} file(s)** inside **{folder_name}**")
            with f_col_del:
                if hasattr(st, "popover"):
                    with st.popover(f"🗑️ Clear Folder ({len(folder_files)})", help=f"Permanently delete all {len(folder_files)} documents in {folder_name}"):
                        st.markdown(f"**Delete all content in `{folder_name}`?**")
                        st.caption("This will permanently remove all files and their vector embeddings in Neo4j.")
                        if st.button("⚠️ Confirm Delete All", key=f"conf_del_all_{folder_name}", type="primary"):
                            succ, _ = _delete_all_in_folder(folder_files)
                            st.success(f"Deleted {succ} file(s) from `{folder_name}`")
                            st.rerun()
                else:
                    if st.button(f"🗑️ Clear Folder ({len(folder_files)})", key=f"del_all_{folder_name}"):
                        succ, _ = _delete_all_in_folder(folder_files)
                        st.success(f"Deleted {succ} file(s) from `{folder_name}`")
                        st.rerun()

            st.markdown("---")

            if view_mode == "🗂️ Grid":
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
                            st.markdown(f"#### {file_info['icon']} {filename}")
                            st.caption(f"**Type:** `{ext}` &nbsp;|&nbsp; **Chunks:** `{chunk_count}`")
                            if clean_desc:
                                st.caption(f"📝 {clean_desc[:60]}..." if len(clean_desc) > 60 else f"📝 {clean_desc}")
                            if upload_date:
                                st.caption(f"📅 {upload_date}")

                            btn_col1, btn_col2 = st.columns([0.6, 0.4])
                            with btn_col1:
                                if st.button("🔍 Inspect", key=f"inspect_{doc_id}"):
                                    st.session_state["selected_doc_id"] = doc_id
                                    st.rerun()
                            with btn_col2:
                                if st.button("🗑️ Delete", key=f"del_grid_{doc_id}"):
                                    if _delete_document(doc_id):
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
                    f"Select file from {folder_name} to inspect/delete",
                    options=[d["Filename"] for d in table_data],
                    key=f"select_table_{folder_name}",
                )
                if selected_row:
                    matched = next((d for d in folder_files if d.get("filename") == selected_row), None)
                    if matched:
                        act_col1, act_col2 = st.columns([0.2, 0.8])
                        with act_col1:
                            if st.button("🔍 Inspect File", key=f"inspect_tbl_{matched['id']}"):
                                st.session_state["selected_doc_id"] = matched["id"]
                                st.rerun()
                        with act_col2:
                            if st.button("🗑️ Delete File", key=f"del_tbl_{matched['id']}"):
                                if _delete_document(matched["id"]):
                                    st.success(f"Deleted `{selected_row}`")
                                    st.rerun()


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
def render_page() -> None:
    st.markdown(
        """
        <div style="text-align: center; padding: 18px 0 8px;">
            <h1 style="
                font-size: 2.2rem;
                background: linear-gradient(to right, #60A5FA, #A78BFA);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 4px;
            ">🗂️ Document File Explorer</h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
                Organize, browse, and inspect your unstructured document library and vector embeddings in Neo4j.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    user_id = st.session_state.get("user_name", "default")
    docs = _fetch_documents()
    all_folders = _get_all_folders(docs)

    # Summary KPI banner
    total_docs = len(docs)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    pdf_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "pdf")
    docx_count = sum(1 for d in docs if (d.get("file_type") or "").lower() == "docx")
    txt_count = sum(1 for d in docs if (d.get("file_type") or "").lower() in ("txt", "md"))

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("📁 Total Files", f"{total_docs}")
    with kpi2:
        st.metric("🧩 Total Chunks", f"{total_chunks:,}")
    with kpi3:
        st.metric("📂 Folders", f"{len(all_folders)}")
    with kpi4:
        st.metric("📕 PDFs", f"{pdf_count}")
    with kpi5:
        st.metric("📘 Word & Text", f"{docx_count + txt_count}")

    st.markdown("---")

    # Main Tabs: Explorer, Upload, Manage Folders
    tab_explorer, tab_upload, tab_folders = st.tabs([
        "🗂️ Browse Files & Folders",
        "⬆️ Upload Documents",
        "⚙️ Manage Folders",
    ])

    with tab_explorer:
        _render_explorer_view(docs, all_folders)

    with tab_upload:
        _render_upload_modal(user_id=user_id, docs=docs)

    with tab_folders:
        _render_manage_folders_section(docs)


render_page()
