"""
document_search.py
------------------
LangChain tool for vector-similarity search over uploaded DocumentChunk nodes.

Works alongside graph_rag_tool (StackExchange) to let the agent retrieve
information from both the structured Q&A knowledge graph AND any user-uploaded
documents (PDF, DOCX, TXT).
"""

from __future__ import annotations

import logging
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
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 20          # candidates fetched from vector index
RERANKER_TOP_N = 8         # documents passed to the LLM after reranking
MAX_CONTENT_CHARS = 2000   # truncation for page_content fed to cross-encoder

# Cypher: hybrid search (vector, fulltext, text) over DocumentChunk embeddings
_VECTOR_SEARCH_QUERY = """
CALL {
    // 1. Vector Search
    CALL db.index.vector.queryNodes('DocumentChunk_index', $top_k, $query_embedding)
    YIELD node, score
    RETURN node, score
    UNION
    // 2. Fulltext Search
    CALL db.index.fulltext.queryNodes('DocumentChunk_keyword_index', $question, {limit: $top_k})
    YIELD node, score
    RETURN node, score
    UNION
    // 3. Text Index Search (Exact/Contains on source)
    MATCH (node:DocumentChunk)
    WHERE node.source CONTAINS $question
    RETURN node, 1.0 AS score
}
WITH node AS chunk, max(score) AS max_score
WHERE ($community_ids IS NULL OR size($community_ids) = 0 OR chunk.communityId IN $community_ids)
MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
RETURN
    chunk.id          AS chunk_id,
    chunk.content     AS content,
    chunk.chunk_index AS chunk_index,
    chunk.source      AS source,
    chunk.communityId AS community_id,
    d.id              AS doc_id,
    d.filename        AS filename,
    d.file_type       AS file_type,
    d.upload_date     AS upload_date,
    d.description     AS description,
    max_score         AS score
ORDER BY score DESC
LIMIT $top_k
"""

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


def _search_document_chunks(question: str, community_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Embed *question* and run hybrid similarity search over DocumentChunk nodes.
    Returns raw Neo4j records.
    """
    graph = get_graph_instance()
    embedder = embedding_model()

    query_embedding = embedder.embed_query(question)
    
    if community_ids is None:
        community_ids = []
        
    records = graph.query(
        _VECTOR_SEARCH_QUERY,
        params={
            "question": question,
            "query_embedding": query_embedding,
            "top_k": VECTOR_TOP_K,
            "community_ids": community_ids,
        },
    )
    logger.info("DocumentSearch: hybrid search returned %d records.", len(records))
    return [dict(r) for r in records]


def _records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw Neo4j DocumentChunk records to LangChain Document objects."""
    docs: List[Document] = []
    for r in records:
        content = (r.get("content") or "")[:MAX_CONTENT_CHARS]
        page_content = (
            f"[Source: {r.get('filename', 'unknown')}]\n"
            f"[Chunk {r.get('chunk_index', '?')}]\n\n"
            f"{content}"
        )
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "chunk_id":    r.get("chunk_id"),
                    "doc_id":      r.get("doc_id"),
                    "filename":    r.get("filename"),
                    "file_type":   r.get("file_type"),
                    "upload_date": r.get("upload_date"),
                    "description": r.get("description"),
                    "chunk_index": r.get("chunk_index"),
                    "source":      r.get("source"),
                    "community_id": r.get("community_id"),
                    "score":       r.get("score"),
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

    Use this tool whenever:
    - The user asks about content from uploaded files or documents.
    - The user references specific documents, specs, manuals, project files, or reports.
    - The user asks domain-specific or private project questions that would be in their document library.

    Args:
        question: The search query or question to find matching document chunks for.
        community_ids: Optional list of community IDs to filter the search results by.

    Returns:
        A formatted string containing the most relevant document passages with
        filename and chunk metadata, or a message if no documents are available.
    """
    logger.info("document_search_tool invoked: %r", question[:120])

    should_stop, stop_reason = check_retrieval_hard_stop()
    if should_stop:
        logger.warning("document_search_tool hard stopped: %s", stop_reason)
        return stop_reason

    increment_tool_call_count()

    try:
        # 1. Hybrid similarity search
        raw_records = _search_document_chunks(question, community_ids=community_ids)

        if not raw_records:
            return (
                "No relevant information found in the uploaded document library. "
                "No documents may have been ingested yet, or none match this query."
            )

        # 2. Convert to Documents
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
