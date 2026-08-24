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

from __future__ import annotations
from .queries import FALLBACK_DOCUMENT_SEARCH_QUERY, HYBRID_DOCUMENT_SEARCH_QUERY
import logging
import re
from typing import Any, Dict, List, Optional

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
VECTOR_TOP_K = 100          # candidates fetched across hybrid index search branches
RERANKER_TOP_N = 20         # documents passed to the LLM after reranking
MAX_CONTENT_CHARS = 3500   # truncation for page_content fed to cross-encoder/LLM

# ---------------------------------------------------------------------------
# Lazy-initialised singletons
# ---------------------------------------------------------------------------
_compressor: Optional[CrossEncoderReranker] = None

def _get_compressor() -> CrossEncoderReranker:
    """Build (or return cached) CrossEncoderReranker."""
    global _compressor
    if _compressor is None:
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


def _search_document_chunks(
    question: str, 
    community_ids: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Embed *question* and run multi-index hybrid search over DocumentChunk and Document nodes.
    Applies community ID filtering over the retrieved index candidates.
    Returns raw Neo4j records with full chunk and document properties/metadata.
    """
    graph = get_graph_instance()
    embedder = embedding_model()

    query_embedding = embedder.embed_query(question)
    fulltext_query = _build_lucene_query(question)
    question_clean = question.strip()[:200]

    # Normalize community_ids to strings for robust comparison (handles int and str lists)
    str_community_ids = [str(cid).strip() for cid in (community_ids or []) if cid is not None and str(cid).strip()]

    params = {
        "question": question,
        "question_clean": question_clean,
        "fulltext_query": fulltext_query,
        "query_embedding": query_embedding,
        "top_k": VECTOR_TOP_K,
        "str_community_ids": str_community_ids,
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
        "DocumentSearch: multi-index search returned %d records (filtered by community_ids=%s).", 
        len(records),
        str_community_ids if str_community_ids else "None",
    )
    return [dict(r) for r in records]


def _records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert raw Neo4j DocumentChunk records to LangChain Document objects with
    rich document metadata and structured context headers.
    """
    docs: List[Document] = []
    for r in records:
        content = (r.get("content") or "")[:MAX_CONTENT_CHARS]
        filename = r.get("filename") or "unknown"
        chunk_idx = r.get("chunk_index")
        chunk_count = r.get("chunk_count")
        file_type = r.get("file_type") or ""
        description = r.get("description") or ""
        upload_date = r.get("upload_date") or ""
        source = r.get("source") or ""
        community_id = r.get("community_id")
        match_types = r.get("match_types") or []
        score = r.get("score")

        # Build structured, highly informative context header for reranker and LLM
        header_parts = [f"[Document: {filename}]"]
        if file_type:
            header_parts.append(f"[Type: {file_type.upper()}]")
        if chunk_idx is not None:
            total_str = f"/{chunk_count}" if chunk_count else ""
            header_parts.append(f"[Chunk: {chunk_idx}{total_str}]")
        if description:
            header_parts.append(f"[Description: {description}]")
        if source and source != filename:
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
                    "chunk_id":     r.get("chunk_id"),
                    "doc_id":       r.get("doc_id"),
                    "filename":     filename,
                    "file_type":    file_type,
                    "upload_date":  upload_date,
                    "user_id":      r.get("user_id"),
                    "chunk_count":  chunk_count,
                    "description":  description,
                    "file_hash":    r.get("file_hash"),
                    "chunk_index":  chunk_idx,
                    "source":       source,
                    "community_id": community_id,
                    "score":        score,
                    "match_types":  match_types,
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Public LangChain tool
# ---------------------------------------------------------------------------
@tool
def document_search_tool(question: str, community_ids: Optional[List[str]] = None) -> str:
    """
    Search through all user-uploaded unstructured documents (such as PDF files, Word .docx documents, text .txt files, Markdown .md files, technical specifications, guides, reports, and internal notes).

    Leverages multi-index hybrid search across vector embeddings, fulltext keyword indexes (content, source, filename, description), and text indexes, extracting rich document/chunk metadata and filtering by community IDs when provided.

    Use this tool whenever:
    - The user asks about content from uploaded files or documents.
    - The user references specific documents, specs, manuals, project files, or reports.
    - The user asks domain-specific or private project questions that would be in their document library.

    Args:
        question: The search query or question to find matching document chunks for.
        community_ids: Optional list of community IDs to filter the search results by.

    Returns:
        A formatted string containing the most relevant document passages with
        filename, chunk, community, and metadata, or a message if no documents are available.
    """
    logger.info("document_search_tool invoked: %r (community_ids=%s)", question[:120], community_ids)

    should_stop, stop_reason = check_retrieval_hard_stop()
    if should_stop:
        logger.warning("document_search_tool hard stopped: %s", stop_reason)
        return stop_reason

    increment_tool_call_count()

    try:
        # 1. Multi-index hybrid similarity search & community ID filtering
        raw_records = _search_document_chunks(question, community_ids=community_ids)

        if not raw_records:
            return (
                "No relevant information found in the uploaded document library. "
                "No documents may have been ingested yet, or none match this query."
            )

        # 2. Convert to Documents with full metadata
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
