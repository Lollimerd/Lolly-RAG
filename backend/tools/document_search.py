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

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.documents import Document

from .queries import FALLBACK_DOCUMENT_SEARCH_QUERY, HYBRID_DOCUMENT_SEARCH_QUERY
from setup.init_config import embedding_model, get_graph_instance, reranker_model
from utils.utils import check_retrieval_hard_stop, escape_lucene_chars, format_docs_with_metadata, increment_tool_call_count

logger = logging.getLogger(__name__)

VECTOR_TOP_K = 1000
RERANKER_TOP_N = 25
MAX_CONTENT_CHARS = 2500
TABULAR_EXTENSIONS = {"csv", "xlsx", "xls"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
PRESENTATION_EXTENSIONS = {"pptx", "ppt"}

_TYPE_MAP = {
    "tabular": ["csv", "xlsx", "xls"], 
    "table": ["csv", "xlsx", "xls"], 
    "tables": ["csv", "xlsx", "xls"],
    "spreadsheet": ["csv", "xlsx", "xls"], 
    "spreadsheets": ["csv", "xlsx", "xls"],
    "excel": ["xlsx", "xls"], 
    "workbook": ["xlsx", "xls"], 
    "sheets": ["xlsx", "xls"],
    "word": ["docx"], 
    "text": ["txt"], 
    "txt": ["txt"], 
    "markdown": ["md"], 
    "md": ["md"],
    "presentation": ["pptx", "ppt"], 
    "presentations": ["pptx", "ppt"], 
    "slides": ["pptx", "ppt"], 
    "powerpoint": ["pptx", "ppt"], 
    "ppt": ["pptx", "ppt"], 
    "pptx": ["pptx", "ppt"],
    "image": list(IMAGE_EXTENSIONS), 
    "images": list(IMAGE_EXTENSIONS), 
    "img": list(IMAGE_EXTENSIONS), 
    "imgs": list(IMAGE_EXTENSIONS),
    "photo": list(IMAGE_EXTENSIONS), "photos": list(IMAGE_EXTENSIONS), "screenshot": list(IMAGE_EXTENSIONS), "screenshots": list(IMAGE_EXTENSIONS),
    "diagram": list(IMAGE_EXTENSIONS), "diagrams": list(IMAGE_EXTENSIONS), "ocr": list(IMAGE_EXTENSIONS),
}

def _get_compressor() -> CrossEncoderReranker:
    """Args: None."""
    return CrossEncoderReranker(model=reranker_model(), top_n=RERANKER_TOP_N)

def _build_lucene_query(question: str) -> str:
    """Args: question: Query text."""
    if not question or not question.strip(): return ""
    tokens = re.findall(r"[\w\.\-]+", question)
    return " OR ".join(escape_lucene_chars(t) for t in tokens if t) if tokens else escape_lucene_chars(question.strip())

def _normalize_file_types(file_type: Optional[str]) -> Optional[List[str]]:
    """Args: file_type: Raw file type input."""
    if not file_type: return None
    ft = file_type.strip().lower().lstrip(".")
    return _TYPE_MAP.get(ft, [ft])

def _auto_detect_tabular_filters(
    question: str, file_type: Optional[str] = None, filename: Optional[str] = None, sheet_name: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """Args: question: Query string, file_type: Optional file type, filename: Optional filename, sheet_name: Optional sheet."""
    target_types = _normalize_file_types(file_type)
    target_fname = (filename or "").strip() or None
    target_sname = (sheet_name or "").strip() or None

    if not target_fname and (m := re.search(r"([\w\-\.\(\) ]+\.(?:csv|xlsx|xls|pdf|docx|txt|md|png|jpg|jpeg|webp|bmp|tiff|pptx|ppt))\b", question, re.I)):
        target_fname = m.group(1).strip()
        if not target_types and "." in target_fname:
            target_types = [target_fname.rsplit(".", 1)[-1].lower()]

    if not target_sname and (m := re.search(r"\b(?:sheet|tab)\b\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?|\b(?:in|from)\s+sheet\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", question, re.I)):
        target_sname = (m.group(1) or m.group(2) or "").strip()
        if not target_types: target_types = ["xlsx", "xls"]

    if target_types and not any(t in ("xlsx", "xls") for t in target_types):
        target_sname = None

    if not target_types:
        lq = question.lower()
        if any(re.search(rf"\b{re.escape(w)}\b", lq) for w in ["image", "images", "photo", "photos", "screenshot", "screenshots", "diagram", "diagrams", "ocr", "chart", "figure"]):
            target_types = list(IMAGE_EXTENSIONS)
        elif "csv" in lq and not any(x in lq for x in ["excel", "xlsx", "xls"]):
            target_types = ["csv"]
        elif any(x in lq for x in ["excel", "xlsx", "xls", "sheet"]) and "csv" not in lq:
            target_types = ["xlsx", "xls"]
        elif any(re.search(rf"\b{re.escape(w)}\b", lq) for w in ["slide", "slides", "presentation", "powerpoint", "deck"]):
            target_types = ["pptx", "ppt"]

    return target_types, target_fname, target_sname

def _search_document_chunks(
    question: str, community_ids: Optional[List[Any]] = None, file_type: Optional[str] = None,
    filename: Optional[str] = None, sheet_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Args: question: Search query, community_ids: Community ID filters, file_type: File type filter, filename: File name filter, sheet_name: Sheet filter."""
    graph, embedder = get_graph_instance(), embedding_model()
    target_file_types, target_filename, target_sheet_name = _auto_detect_tabular_filters(question, file_type, filename, sheet_name)
    str_cids = [str(cid).strip() for cid in (community_ids or []) if cid is not None and str(cid).strip()]

    params = {
        "question": question, 
        "question_clean": question.strip()[:200],
        "fulltext_query": _build_lucene_query(question), 
        "query_embedding": embedder.embed_query(question),
        "top_k": VECTOR_TOP_K, "str_community_ids": str_cids,
        "target_file_types": target_file_types or [], 
        "target_filename": target_filename or "", 
        "target_sheet_name": target_sheet_name or "",
    }
    try: records = graph.query(HYBRID_DOCUMENT_SEARCH_QUERY, params=params)
    except Exception as exc:
        logger.warning("Primary hybrid search failed (%s); running fallback.", exc)
        records = graph.query(FALLBACK_DOCUMENT_SEARCH_QUERY, params=params)
    logger.info("DocumentSearch: returned %d records.", len(records))
    return [dict(r) for r in records]

def _format_tabular_chunk_content(raw_content: str, file_type: str) -> str:
    """Args: raw_content: Raw chunk content, file_type: Document file type."""
    if not raw_content: return ""
    s = raw_content.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            row = json.loads(s)
            if isinstance(row, dict):
                kv = [f"- **{k}**: {str(v).strip()}" for k, v in row.items() if v is not None and str(v).strip()]
                if kv: return "### Tabular Row Record:\n" + "\n".join(kv)
        except Exception: pass
    return raw_content

def _records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """Args: records: Neo4j chunk records."""
    docs: List[Document] = []
    for r in records:
        raw_content, filename, chunk_idx, chunk_count = r.get("content") or "", r.get("filename") or "unknown", r.get("chunk_index"), r.get("chunk_count")
        file_type, description, upload_date, source = (r.get("file_type") or "").lower(), r.get("description") or "", r.get("upload_date") or "", r.get("source") or ""
        comm_id, match_types, score = r.get("community_id"), r.get("match_types") or [], r.get("score")

        is_tabular = file_type in TABULAR_EXTENSIONS or "Row " in source or "Sheet: " in source or "Dataset Overview" in source
        is_schema = "Dataset Overview" in source or "Tabular Dataset Overview" in raw_content or (is_tabular and chunk_idx == 0)
        is_img = file_type in IMAGE_EXTENSIONS or "Image" in source
        is_pres = file_type in PRESENTATION_EXTENSIONS or "Slide " in source or "Presentation" in source

        sm = re.search(r"Sheet:\s*['\"]?([^'\",\)]+?)['\"]?(?:,|\)|$)", source)
        rm = re.search(r"Rows?\s*([0-9]+(?:\s*-\s*[0-9]+)?)", source)
        sheet_name, row_info = sm.group(1).strip() if sm else "", rm.group(1).strip() if rm else ""

        tags = [f"[Document: {filename}]"]
        if file_type: tags.append(f"[Type: {file_type.upper()}]")
        tags.append("[Dataset: Tabular Schema]" if is_schema else "[Dataset: Tabular Records]" if is_tabular else "[Media: Image/OCR]" if is_img else "[Media: Slide Presentation]" if is_pres else "")
        if sheet_name: tags.append(f"[Sheet: {sheet_name}]")
        if row_info: tags.append(f"[Rows: {row_info}]")
        if chunk_idx is not None: tags.append(f"[Chunk: {chunk_idx}{f'/{chunk_count}' if chunk_count else ''}]")
        if description: tags.append(f"[Description: {description}]")
        if source and source != filename and not (sheet_name and row_info) and not is_schema: tags.append(f"[Source: {source}]")
        if upload_date: tags.append(f"[Uploaded: {upload_date}]")
        if comm_id is not None: tags.append(f"[Community ID: {comm_id}]")
        if match_types: tags.append(f"[Matched By: {', '.join(match_types)}]")

        header = " ".join(t for t in tags if t)
        docs.append(Document(
            page_content=f"{header}\n\n{_format_tabular_chunk_content(raw_content, file_type)[:MAX_CONTENT_CHARS]}",
            metadata={"chunk_id": r.get("chunk_id"), "doc_id": r.get("doc_id"), "filename": filename, "file_type": file_type, "is_tabular": is_tabular, "is_table_summary": is_schema, "is_image": is_img, "is_presentation": is_pres, "sheet_name": sheet_name, "row_info": row_info, "upload_date": upload_date, "user_id": r.get("user_id"), "chunk_count": chunk_count, "description": description, "file_hash": r.get("file_hash"), "chunk_index": chunk_idx, "source": source, "community_id": comm_id, "score": score, "match_types": match_types},
        ))
    return docs

@tool
def document_search_tool(
    question: str, community_ids: Optional[List[str]] = None, file_type: Optional[str] = None,
    filename: Optional[str] = None, sheet_name: Optional[str] = None,
) -> str:
    """Search uploaded documents, spreadsheets, images, presentations, and PDFs.

    Args:
        question: Search query or visual description lookup.
        community_ids: Optional community IDs to filter by.
        file_type: Optional file type filter ('image', 'csv', 'xlsx', 'pdf', 'docx', 'pptx').
        filename: Optional filename filter (e.g. 'chart.png', 'movies.csv').
        sheet_name: Optional Excel sheet name filter (e.g. 'Sheet1').
    """
    logger.info("document_search_tool: %r", question[:100])
    should_stop, stop_reason = check_retrieval_hard_stop()
    if should_stop: return stop_reason
    increment_tool_call_count()

    try:
        raw_records = _search_document_chunks(question, community_ids, file_type, filename, sheet_name)
        if not raw_records: return "No relevant information found in the uploaded document library matching this query."
        docs = _records_to_documents(raw_records)
        try: reranked = list(_get_compressor().compress_documents(documents=docs, query=question))
        except Exception as exc: logger.warning("Reranking failed: %s", exc); reranked = docs
        return format_docs_with_metadata(reranked) if reranked else "Document search returned no results after reranking."
    except Exception as exc:
        logger.error("document_search_tool error: %s", exc, exc_info=True)
        return f"Document search failed: {exc}."
