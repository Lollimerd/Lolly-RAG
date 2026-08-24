"""
doc_processor.py
----------------
Unstructured document ingestion pipeline for lolly-rag.

Inspired by CORTEX-AI-SUPER-RAG's doc_handler.py (github.com/SaiAkhil066/CORTEX-AI-SUPER-RAG)
but adapted to the Neo4j-centric architecture of this codebase.

Flow:
    uploaded file → load_document() → chunk_documents() → embed_and_store_chunks()
                                                           ↓
                                              Neo4j: (Document)-[:HAS_CHUNK]->(DocumentChunk)
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match CORTEX-AI-SUPER-RAG defaults)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_SEPARATOR = "\n"
EMBED_BATCH_SIZE = 32  # texts per embedding call (memory-safe)
WRITE_BATCH_SIZE = 50  # chunks per Neo4j write

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

# Upsert a Document node
_UPSERT_DOCUMENT_QUERY = """
MERGE (d:Document {id: $doc_id})
ON CREATE SET
    d.filename    = $filename,
    d.file_type   = $file_type,
    d.upload_date = $upload_date,
    d.user_id     = $user_id,
    d.chunk_count = $chunk_count,
    d.description = $description,
    d.file_hash   = $file_hash
ON MATCH SET
    d.upload_date = $upload_date,
    d.chunk_count = $chunk_count,
    d.description = $description,
    d.file_hash   = $file_hash
RETURN d.id AS id
"""

# Batch-create DocumentChunk nodes and link to their parent Document
_INSERT_CHUNKS_QUERY = """
UNWIND $chunks AS c
MERGE (chunk:DocumentChunk {id: c.id})
ON CREATE SET
    chunk.content     = c.content,
    chunk.chunk_index = c.chunk_index,
    chunk.source      = c.source,
    chunk.embedding   = c.embedding
WITH chunk, c
MATCH (d:Document {id: c.doc_id})
MERGE (d)-[:HAS_CHUNK]->(chunk)
"""

# List all documents (metadata only, no embeddings)
_LIST_DOCUMENTS_QUERY = """
MATCH (d:Document)
RETURN d.id          AS id,
       d.filename    AS filename,
       d.file_type   AS file_type,
       d.upload_date AS upload_date,
       d.user_id     AS user_id,
       d.chunk_count AS chunk_count,
       d.description AS description,
       d.file_hash   AS file_hash
ORDER BY d.upload_date DESC
"""

# Delete a document and all its chunks
_DELETE_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
DETACH DELETE d, c
"""

# Check if a document exists by ID
_GET_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
RETURN d.id AS id, d.filename AS filename
"""

# Query to find duplicate document by filename or file_hash
_FIND_EXISTING_DOCUMENT_QUERY = """
MATCH (d:Document)
WHERE d.filename = $filename OR (d.file_hash IS NOT NULL AND d.file_hash <> '' AND d.file_hash = $file_hash)
RETURN d.id AS id, d.filename AS filename, d.chunk_count AS chunk_count, d.file_hash AS file_hash
LIMIT 1
"""

# Get all chunks for a document
_GET_DOCUMENT_CHUNKS_QUERY = """
MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:DocumentChunk)
RETURN c.id AS id, c.chunk_index AS chunk_index, c.content AS content, c.source AS source
ORDER BY c.chunk_index ASC
"""


def find_existing_document(
    filename: str,
    file_hash: Optional[str],
    graph: Any,
) -> Optional[Dict[str, Any]]:
    """Check if a document with the same filename or content hash already exists in Neo4j."""
    results = graph.query(
        _FIND_EXISTING_DOCUMENT_QUERY,
        params={"filename": filename, "file_hash": file_hash or ""},
    )
    if results and len(results) > 0:
        return dict(results[0])
    return None


def get_document_chunks(doc_id: str, graph: Any) -> List[Dict[str, Any]]:
    """Return all chunks for a given document (ordered by chunk_index)."""
    results = graph.query(_GET_DOCUMENT_CHUNKS_QUERY, params={"doc_id": doc_id})
    return [dict(r) for r in results]


# Update document metadata (such as folder / description)
_UPDATE_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
SET d.description = $description
RETURN d.id AS id
"""


def update_document(doc_id: str, description: str, graph: Any) -> bool:
    """Update description/folder metadata of a Document node."""
    graph.query(_UPDATE_DOCUMENT_QUERY, params={"doc_id": doc_id, "description": description})
    return True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_document(file_path: str, filename: str) -> List[Document]:
    """
    Load a single file into LangChain Document objects.

    Supports .pdf, .docx, .txt, and .md files.
    Raises ValueError for unsupported extensions.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)  # type: ignore[abstract]
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    docs = loader.load()
    logger.info("Loaded %d pages/sections from '%s'", len(docs), filename)
    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split loaded documents into fixed-size overlapping chunks.

    Returns a list of Document objects where page_content is the chunk text
    and metadata carries the original source information.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=DEFAULT_CHUNK_SEPARATOR,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d docs → %d chunks (size=%d, overlap=%d)",
                len(documents), len(chunks), chunk_size, chunk_overlap)
    return chunks


def embed_and_store_chunks(
    chunks: List[Document],
    doc_id: str,
    filename: str,
    user_id: str,
    description: str,
    graph: Any,
    embedder: Any,
    file_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute embeddings for all chunks (in micro-batches) and write them to Neo4j.

    Creates:
        (d:Document {id, filename, file_type, upload_date, user_id, chunk_count, description, file_hash})
        (c:DocumentChunk {id, content, chunk_index, source, embedding})
        (d)-[:HAS_CHUNK]->(c)

    Returns a summary dict with doc_id and chunk_count.
    """
    file_ext = os.path.splitext(filename)[1].lower().lstrip(".")
    upload_date = datetime.now(timezone.utc).isoformat()
    chunk_count = len(chunks)

    # 1. Upsert the parent Document node first
    graph.query(
        _UPSERT_DOCUMENT_QUERY,
        params={
            "doc_id":      doc_id,
            "filename":    filename,
            "file_type":   file_ext,
            "upload_date": upload_date,
            "user_id":     user_id,
            "chunk_count": chunk_count,
            "description": description,
            "file_hash":   file_hash or "",
        },
    )
    logger.info("Document node created/updated: id=%s, filename=%s", doc_id, filename)

    # 2. Compute embeddings in micro-batches to limit peak memory
    texts = [c.page_content for c in chunks]
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        del batch, batch_embeddings
    logger.info("Computed %d embeddings for '%s'", len(all_embeddings), filename)

    # 3. Build chunk dicts and write to Neo4j in batches
    chunk_records = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        chunk_records.append(
            {
                "id":          str(uuid.uuid4()),
                "doc_id":      doc_id,
                "content":     chunk.page_content,
                "chunk_index": idx,
                "source":      chunk.metadata.get("source", filename),
                "embedding":   embedding,
            }
        )

    for i in range(0, len(chunk_records), WRITE_BATCH_SIZE):
        batch = chunk_records[i : i + WRITE_BATCH_SIZE]
        graph.query(_INSERT_CHUNKS_QUERY, params={"chunks": batch})
        del batch

    del all_embeddings, chunk_records
    gc.collect()

    logger.info("Stored %d chunks for document '%s' (id=%s)", chunk_count, filename, doc_id)
    return {"doc_id": doc_id, "filename": filename, "chunk_count": chunk_count}


def list_documents(graph: Any) -> List[Dict[str, Any]]:
    """Return a list of all Document nodes (no embeddings)."""
    results = graph.query(_LIST_DOCUMENTS_QUERY)
    return [dict(r) for r in results]


def delete_document(doc_id: str, graph: Any) -> bool:
    """
    Delete a Document and all its HAS_CHUNK → DocumentChunk nodes.
    Returns True if the document existed and was deleted, False otherwise.
    """
    exists = graph.query(_GET_DOCUMENT_QUERY, params={"doc_id": doc_id})
    if not exists:
        return False
    graph.query(_DELETE_DOCUMENT_QUERY, params={"doc_id": doc_id})
    logger.info("Deleted document %s and all its chunks.", doc_id)
    return True


def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    description: str,
    graph: Any,
    embedder: Any,
    temp_dir: str = "/tmp/lolly_rag_uploads",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end pipeline for a single uploaded file:
        bytes → duplicate check → temp file → load → chunk → embed → store → cleanup

    If the document already exists in Neo4j (by filename or file hash) and force=False,
    skips processing and returns status='skipped'.
    """
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for existing document
    existing = find_existing_document(filename=filename, file_hash=file_hash, graph=graph)
    if existing and not force:
        logger.info(
            "Document '%s' (hash=%s) already exists with id=%s. Skipping duplicate ingestion.",
            filename,
            file_hash[:8],
            existing["id"],
        )
        return {
            "status": "skipped",
            "doc_id": existing["id"],
            "filename": filename,
            "chunk_count": existing.get("chunk_count", 0),
            "message": f"Document '{filename}' was already ingested. Skipping duplicate ingestion.",
        }

    if existing and force:
        logger.info("Force re-ingesting document '%s' (deleting existing id=%s).", filename, existing["id"])
        delete_document(existing["id"], graph)

    os.makedirs(temp_dir, exist_ok=True)
    doc_id = str(uuid.uuid4())
    temp_path = os.path.join(temp_dir, f"{doc_id}_{filename}")

    try:
        # Write bytes to temp file
        with open(temp_path, "wb") as fh:
            fh.write(file_bytes)

        # Load, chunk, embed and store
        documents = load_document(temp_path, filename)
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        result = embed_and_store_chunks(
            chunks=chunks,
            doc_id=doc_id,
            filename=filename,
            user_id=user_id,
            description=description,
            graph=graph,
            embedder=embedder,
            file_hash=file_hash,
        )
        result["status"] = "success"
        result["message"] = f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks."
        return result

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
