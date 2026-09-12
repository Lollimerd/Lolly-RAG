"""
document_search.py
------------------
LangChain tool for multi-index hybrid search over uploaded DocumentChunk and Document nodes.

Makes full use of:
1. Vector Index: DocumentChunk_index (embedding similarity on chunk content)
2. Fulltext Keyword Indexes:
   - DocumentChunk_keyword_index (on chunk content and source)
   - Document_keyword_index (on document filename and description)
3. Text Indexes & Metadata matching:
   - DocumentChunk_source_text_index (on source path)
   - Document text indexes (filename, description, file_type, source)
4. Comprehensive properties & metadata extraction:
   - chunk_id, doc_id, filename, file_type, upload_date, user_id, chunk_count,
     description, chunk_index, source, file_hash, score, match_types
5. Resilient Community ID filtering:
   - Supports single integer/string community IDs and list-based intermediate community IDs
     from hierarchical graph clustering (e.g., Leiden / Louvain).
   - Applied across retrieved index candidates before cross-encoder reranking.
"""

from .queries import FALLBACK_DOCUMENT_SEARCH_QUERY, HYBRID_DOCUMENT_SEARCH_QUERY
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document

from setup.init_config import embedding_model, get_graph_instance, reranker_model
from utils.utils import (
    format_docs_with_metadata,
    check_retrieval_hard_stop,
    increment_tool_call_count,
    escape_lucene_chars,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 1000        # candidates fetched across hybrid index search branches
RERANKER_TOP_N = 25        # documents passed to the LLM after reranking
MAX_CONTENT_CHARS = 2500   # truncation for page_content fed to cross-encoder/LLM
TABULAR_EXTENSIONS = {"csv", "xlsx", "xls"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
PRESENTATION_EXTENSIONS = {"pptx", "ppt"}


def _get_compressor() -> CrossEncoderReranker:
    """Build (or return cached) CrossEncoderReranker."""
    _compressor = CrossEncoderReranker(
        model=reranker_model(),
        top_n=RERANKER_TOP_N,
    )
    logger.info("DocumentSearch: CrossEncoderReranker initialised (top_n=%d).", RERANKER_TOP_N)
    return _compressor


def _build_lucene_query(question: str) -> str:
    """
    Builds a sanitized Lucene query string from the user's question,
    escaping special Lucene operators to prevent syntax errors.
    """
    if not question or not question.strip():
        return ""
    # Extract alphanumeric and path words
    tokens = re.findall(r"[\w\.\-]+", question)
    if not tokens:
        return escape_lucene_chars(question.strip())
    # Escape each token and join with OR for comprehensive keyword recall
    clean_tokens = [escape_lucene_chars(t) for t in tokens if len(t) > 0]
    return " OR ".join(clean_tokens)


def _normalize_file_types(file_type: Optional[str]) -> Optional[List[str]]:
    """Normalize user or agent supplied file_type parameter to a list of extensions."""
    if not file_type:
        return None
    ft = file_type.strip().lower().lstrip(".")
    if ft in ("tabular", "table", "tables", "spreadsheet", "spreadsheets"):
        return ["csv", "xlsx", "xls"]
    elif ft in ("excel", "workbook", "sheets"):
        return ["xlsx", "xls"]
    elif ft == "word":
        return ["docx"]
    elif ft in ("text", "txt"):
        return ["txt"]
    elif ft in ("markdown", "md"):
        return ["md"]
    elif ft in ("image", "images", "img", "imgs", "picture", "pictures", "photo", "photos", "screenshot", "screenshots", "diagram", "diagrams", "ocr"):
        return ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
    elif ft in ("presentation", "presentations", "slides", "powerpoint", "ppt", "pptx"):
        return ["pptx", "ppt"]
    elif ft in ("csv", "xlsx", "xls", "pdf", "docx", "pptx", "ppt", "png", "jpg", "jpeg", "webp", "bmp", "tiff"):
        return [ft]
    return [ft]


def _auto_detect_tabular_filters(
    question: str,
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """
    Auto-detect referenced filenames (e.g. *.csv, *.xlsx, *.png, *.jpg), sheets, images,
    or tabular requests from question text if not explicitly passed.
    """
    target_types = _normalize_file_types(file_type)
    target_fname = (filename or "").strip() or None
    target_sname = (sheet_name or "").strip() or None

    # Auto-detect filename like movies.csv, quarterly_sales.xlsx, chart.png, 9 (12).png
    if not target_fname:
        fn_match = re.search(
            r"([\w\-\.\(\) ]+\.(?:csv|xlsx|xls|pdf|docx|txt|md|png|jpg|jpeg|webp|bmp|tiff|pptx|ppt))\b",
            question,
            re.IGNORECASE,
        )
        if fn_match:
            target_fname = fn_match.group(1).strip()
            # If a specific filename was found, infer its file_type if not set
            if not target_types and "." in target_fname:
                ext = target_fname.rsplit(".", 1)[-1].lower()
                target_types = [ext]

    # Auto-detect sheet reference like 'sheet: Q3_Summary', 'sheet = Sheet1', 'in sheet Sheet1'
    if not target_sname:
        sheet_match = re.search(
            r"\b(?:sheet|tab)\b\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?|\b(?:in|from)\s+sheet\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            question,
            re.IGNORECASE,
        )
        if sheet_match:
            target_sname = (sheet_match.group(1) or sheet_match.group(2) or "").strip()
            if not target_types:
                target_types = ["xlsx", "xls"]

    # If searching specifically for images, presentations, or text files, ignore any spurious sheet filters
    if target_types and not any(t in ("xlsx", "xls") for t in target_types):
        target_sname = None

    # Auto-detect image / OCR intent if user asks about images, photos, diagrams, screenshots
    if not target_types:
        lower_q = question.lower()
        image_keywords = [
            "image", "images", "photo", "photos", "picture", "pictures",
            "screenshot", "screenshots", "diagram", "diagrams", "figure",
            "ocr", "chart", "graphic", "illustration", "drawing", "visual"
        ]
        if any(re.search(rf"\b{re.escape(w)}\b", lower_q) for w in image_keywords):
            target_types = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]

    # Auto-detect general tabular intent if user mentions csv or spreadsheet keywords
    if not target_types:
        lower_q = question.lower()
        tabular_keywords = [
            " csv", "csv ", ".csv", "spreadsheet", "spreadsheets",
            "excel ", " excel", ".xlsx", ".xls", "table row", "tabular",
            "table schema", "column names", "dataset schema", "data columns"
        ]
        if any(w in lower_q for w in tabular_keywords):
            if "csv" in lower_q and not any(x in lower_q for x in ["excel", "xlsx", "xls"]):
                target_types = ["csv"]
            elif any(x in lower_q for x in ["excel", "xlsx", "xls", "sheet"]) and "csv" not in lower_q:
                target_types = ["xlsx", "xls"]

    # Auto-detect presentation intent
    if not target_types:
        lower_q = question.lower()
        pres_keywords = ["slide", "slides", "presentation", "powerpoint", "deck", "slide deck"]
        if any(re.search(rf"\b{re.escape(w)}\b", lower_q) for w in pres_keywords):
            target_types = ["pptx", "ppt"]

    return target_types, target_fname, target_sname


def _search_document_chunks(
    question: str, 
    community_ids: Optional[List[Any]] = None,
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed *question* and run multi-index hybrid search over DocumentChunk and Document nodes.
    Applies community ID and tabular metadata filtering (file_type, filename, sheet_name)
    over the retrieved index candidates.
    """
    graph = get_graph_instance()
    embedder = embedding_model()

    query_embedding = embedder.embed_query(question)
    fulltext_query = _build_lucene_query(question)
    question_clean = question.strip()[:200]

    target_file_types, target_filename, target_sheet_name = _auto_detect_tabular_filters(
        question=question,
        file_type=file_type,
        filename=filename,
        sheet_name=sheet_name,
    )

    # Normalize community_ids to strings for robust comparison
    str_community_ids = [str(cid).strip() for cid in (community_ids or []) if cid is not None and str(cid).strip()]

    params = {
        "question": question,
        "question_clean": question_clean,
        "fulltext_query": fulltext_query,
        "query_embedding": query_embedding,
        "top_k": VECTOR_TOP_K,
        "str_community_ids": str_community_ids,
        "target_file_types": target_file_types or [],
        "target_filename": target_filename or "",
        "target_sheet_name": target_sheet_name or "",
    }

    try:
        records = graph.query(HYBRID_DOCUMENT_SEARCH_QUERY, params=params)
    except Exception as exc:
        logger.warning(
            "Primary hybrid document search failed (%s); attempting fallback query without Document_keyword_index.",
            exc,
        )
        try:
            records = graph.query(FALLBACK_DOCUMENT_SEARCH_QUERY, params=params)
        except Exception as fallback_exc:
            logger.error("Fallback document search failed: %s", fallback_exc)
            raise fallback_exc

    logger.info(
        "DocumentSearch: multi-index search returned %d records (file_types=%s, filename=%s, sheet=%s, community_ids=%s).", 
        len(records),
        target_file_types if target_file_types else "All",
        target_filename if target_filename else "All",
        target_sheet_name if target_sheet_name else "All",
        str_community_ids if str_community_ids else "None",
    )
    return [dict(r) for r in records]


def _format_tabular_chunk_content(raw_content: str, file_type: str) -> str:
    """
    Formats raw chunk content into clean, readable tabular text.
    - If content is JSON string (APOC CSV engine), converts to clean Markdown Key-Value list.
    - If content is hierarchical Schema Summary or Semantic Records, preserves formatting.
    - If content is Markdown table, preserves headers and formatting.
    """
    if not raw_content:
        return ""

    stripped = raw_content.strip()
    # Check if this is an APOC JSON-formatted row
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            row_dict = json.loads(stripped)
            if isinstance(row_dict, dict):
                kv_lines = []
                for k, v in row_dict.items():
                    val_str = str(v).strip() if v is not None else ""
                    if val_str:
                        kv_lines.append(f"- **{k}**: {val_str}")
                if kv_lines:
                    return "### Tabular Row Record:\n" + "\n".join(kv_lines)
        except Exception:
            pass

    return raw_content


def _records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert raw Neo4j DocumentChunk records to LangChain Document objects with
    rich document metadata, tabular formatting, and structured context headers.
    """
    docs: List[Document] = []
    for r in records:
        raw_content = r.get("content") or ""
        filename = r.get("filename") or "unknown"
        chunk_idx = r.get("chunk_index")
        chunk_count = r.get("chunk_count")
        file_type = (r.get("file_type") or "").lower()
        description = r.get("description") or ""
        upload_date = r.get("upload_date") or ""
        source = r.get("source") or ""
        community_id = r.get("community_id")
        match_types = r.get("match_types") or []
        score = r.get("score")

        # Format tabular data (Hierarchical Schema Summary, Semantic Records, or APOC JSON)
        is_tabular = file_type in TABULAR_EXTENSIONS or "Row " in source or "Sheet: " in source or "Dataset Overview" in source
        is_schema_overview = (
            "Dataset Overview" in source
            or "Tabular Dataset Overview" in raw_content
            or (is_tabular and chunk_idx == 0)
        )
        is_image = file_type in IMAGE_EXTENSIONS or "Image" in source or "(Image)" in source
        is_presentation = file_type in PRESENTATION_EXTENSIONS or "Slide " in source or "Presentation" in source

        formatted_content = _format_tabular_chunk_content(raw_content, file_type)
        content = formatted_content[:MAX_CONTENT_CHARS]

        # Extract Sheet Name and Row Numbers if present in source
        sheet_match = re.search(r"Sheet:\s*['\"]?([^'\",\)]+?)['\"]?(?:,|\)|$)", source)
        sheet_name = sheet_match.group(1).strip() if sheet_match else ""

        row_match = re.search(r"Rows?\s*([0-9]+(?:\s*-\s*[0-9]+)?)", source)
        row_info = row_match.group(1).strip() if row_match else ""

        # Build structured context header
        header_parts = [f"[Document: {filename}]"]
        if file_type:
            header_parts.append(f"[Type: {file_type.upper()}]")
        if is_schema_overview:
            header_parts.append("[Dataset: Tabular Schema & Overview]")
        elif is_tabular:
            header_parts.append("[Dataset: Tabular Records]")
        elif is_image:
            header_parts.append("[Media: Image & OCR Text]")
        elif is_presentation:
            header_parts.append("[Media: Slide Presentation]")

        if sheet_name:
            header_parts.append(f"[Sheet: {sheet_name}]")
        if row_info:
            header_parts.append(f"[Rows: {row_info}]")
        if chunk_idx is not None:
            total_str = f"/{chunk_count}" if chunk_count else ""
            header_parts.append(f"[Chunk: {chunk_idx}{total_str}]")
        if description:
            header_parts.append(f"[Description: {description}]")
        if source and source != filename and not (sheet_name and row_info) and not is_schema_overview:
            header_parts.append(f"[Source: {source}]")
        if upload_date:
            header_parts.append(f"[Uploaded: {upload_date}]")
        if community_id is not None:
            header_parts.append(f"[Community ID: {community_id}]")
        if match_types:
            header_parts.append(f"[Matched By: {', '.join(match_types)}]")

        header_str = " ".join(header_parts)
        page_content = f"{header_str}\n\n{content}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "chunk_id":            r.get("chunk_id"),
                    "doc_id":              r.get("doc_id"),
                    "filename":            filename,
                    "file_type":           file_type,
                    "is_tabular":          is_tabular,
                    "is_table_summary":    is_schema_overview,
                    "is_image":            is_image,
                    "is_presentation":     is_presentation,
                    "sheet_name":          sheet_name,
                    "row_info":            row_info,
                    "upload_date":         upload_date,
                    "user_id":             r.get("user_id"),
                    "chunk_count":         chunk_count,
                    "description":         description,
                    "file_hash":           r.get("file_hash"),
                    "chunk_index":         chunk_idx,
                    "source":              source,
                    "community_id":        community_id,
                    "score":               score,
                    "match_types":         match_types,
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Public LangChain tool
# ---------------------------------------------------------------------------
@tool
def document_search_tool(
    question: str, 
    community_ids: Optional[List[str]] = None,
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> str:
    """
    Search through all user-uploaded documents, spreadsheets, images, and presentations:
    - Images (.png, .jpg, .jpeg, .webp, .bmp, .tiff) transcribed via Nemotron OCR v2 with extracted text, numbers, labels, layout, and visual metadata
    - Presentations (.pptx, .ppt) with slide contents, tables, speaker notes, and embedded slide image OCR
    - Excel workbooks (.xlsx, .xls) and multi-sheet spreadsheets
    - CSV data files (.csv) and structured data tables
    - PDF documents (.pdf), Word files (.docx), Text files (.txt), and Markdown (.md)

    Leverages multi-index hybrid search across vector embeddings, fulltext keyword indexes (content, source, filename, description), and text indexes, extracting rich metadata (sheet names, row indices, table headers, column attributes, and community IDs).

    Use this tool whenever:
    - The user asks about an image, uploaded photo, screenshot, diagram, chart, or visual document (e.g. 'what is in this image?', 'describe the screenshot', 'what can you see?').
    - An image or file is attached to the chat (e.g. [Attached File(s): ...]).
    - The user asks about content from uploaded files, spreadsheets, tables, CSV rows, or documents.
    - The user asks for specific columns, metrics, aggregations, or records in CSV or Excel datasets (e.g. movies.csv, sales.xlsx).
    - The user references specific documents, sheets, data tables, specs, manuals, project files, or reports.

    Args:
        question: The search query, keyword, visual description lookup, or question to find matching chunks for.
        community_ids: Optional list of community IDs to filter the search results by.
        file_type: Optional filter for file type (e.g. 'image', 'png', 'jpg', 'csv', 'xlsx', 'excel', 'tabular', 'pdf', 'docx', 'pptx').
        filename: Optional filename filter (e.g. 'chart.png', 'movies.csv', 'sales.xlsx').
        sheet_name: Optional Excel sheet name filter (e.g. 'Sheet1', 'Q3_Financials').

    Returns:
        A formatted string containing the most relevant document passages, OCR image text, tabular rows, and table excerpts with
        filename, metadata, and extracted content, or a message if no documents match.
    """
    logger.info(
        "document_search_tool invoked: %r (community_ids=%s, file_type=%s, filename=%s, sheet_name=%s)", 
        question[:120], 
        community_ids, 
        file_type, 
        filename, 
        sheet_name,
    )

    should_stop, stop_reason = check_retrieval_hard_stop()
    if should_stop:
        logger.warning("document_search_tool hard stopped: %s", stop_reason)
        return stop_reason

    increment_tool_call_count()

    try:
        # 1. Multi-index hybrid similarity search & community ID / tabular filtering
        raw_records = _search_document_chunks(
            question, 
            community_ids=community_ids,
            file_type=file_type,
            filename=filename,
            sheet_name=sheet_name,
        )

        if not raw_records:
            filter_desc = []
            if filename:
                filter_desc.append(f"file '{filename}'")
            if file_type:
                filter_desc.append(f"type '{file_type}'")
            if sheet_name:
                filter_desc.append(f"sheet '{sheet_name}'")
            
            filter_str = f" matching {', '.join(filter_desc)}" if filter_desc else ""
            return (
                f"No relevant information found in the uploaded document library{filter_str}. "
                "No documents may have been ingested yet, or none match this query."
            )

        # 2. Convert to Documents with full metadata & tabular formatting
        docs = _records_to_documents(raw_records)

        # 3. Rerank with CrossEncoder
        compressor = _get_compressor()
        try:
            reranked_docs = list(compressor.compress_documents(documents=docs, query=question))
        except Exception as exc:
            logger.warning("Reranking failed, returning unranked docs: %s", exc)
            reranked_docs = docs

        if not reranked_docs:
            return "Document search returned no results after reranking."

        logger.info("document_search_tool: returning %d reranked chunks.", len(reranked_docs))
        return format_docs_with_metadata(reranked_docs)

    except Exception as exc:
        logger.error("document_search_tool error: %s", exc, exc_info=True)
        return f"Document search failed: {exc}."

