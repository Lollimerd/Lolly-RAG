"""
doc_processor.py
----------------
Unified document ingestion pipeline and Neo4j graph store orchestrator.

Coordinates specialized document processors:
- Text documents (.docx, .pdf, .txt, .md) via utils.text_processor
- Tabular data (.csv, .xlsx, .xls) via utils.tabular_processor
- Presentations (.pptx, .ppt) and Images (.png, .jpg, OCR) via utils.media_processor

Handles text splitting/chunking, batch vector embedding computation,
and transactional persistence of Document and DocumentChunk nodes in Neo4j.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# Import specialized processors
from utils.media_processor import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    PRESENTATION_EXTENSIONS,
    load_media_document,
)
from utils.tabular_processor import (
    TABULAR_EXTENSIONS,
    ingest_csv_with_apoc,
    load_tabular_document,
)
from utils.text_processor import (
    TEXT_EXTENSIONS,
    load_text_document,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_SEPARATOR = "\n"
EMBED_BATCH_SIZE = 32    # Micro-batch size per embedding API call
MAX_EMBED_WORKERS = int(os.getenv("MAX_EMBED_WORKERS", "4"))  # Concurrency worker threads
WRITE_BATCH_SIZE = 50    # Chunks per Neo4j transaction
DEFAULT_TEMP_DIR = "/tmp/lolly_rag_uploads"

# Combined supported file extensions
SUPPORTED_EXTENSIONS = set.union(TEXT_EXTENSIONS, TABULAR_EXTENSIONS, MEDIA_EXTENSIONS)

# ---------------------------------------------------------------------------
# Cypher Queries
# ---------------------------------------------------------------------------

# Upsert a Document node with metadata and content hash
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

# Batch-create DocumentChunk nodes and connect to parent Document
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

# Delete a document and all associated chunks
_DELETE_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
DETACH DELETE d, c
"""

# Check document existence by ID
_GET_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
RETURN d.id AS id, d.filename AS filename
"""

# Find existing document by filename or content hash
_FIND_EXISTING_DOCUMENT_QUERY = """
MATCH (d:Document)
WHERE d.filename = $filename OR (d.file_hash IS NOT NULL AND d.file_hash <> '' AND d.file_hash = $file_hash)
RETURN d.id AS id, d.filename AS filename, d.chunk_count AS chunk_count, d.file_hash AS file_hash
LIMIT 1
"""

# Fetch all chunks for a document ordered by index
_GET_DOCUMENT_CHUNKS_QUERY = """
MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:DocumentChunk)
RETURN c.id AS id, c.chunk_index AS chunk_index, c.content AS content, c.source AS source
ORDER BY c.chunk_index ASC
"""

# Update document description metadata
_UPDATE_DOCUMENT_QUERY = """
MATCH (d:Document {id: $doc_id})
SET d.description = $description
RETURN d.id AS id
"""

# ---------------------------------------------------------------------------
# Database Inspection & Metadata Helpers
# ---------------------------------------------------------------------------
def find_existing_document(
    filename: str,
    file_hash: Optional[str],
    graph: Any,
) -> Optional[Dict[str, Any]]:
    """Check if a document with matching filename or content hash already exists."""
    results = graph.query(
        _FIND_EXISTING_DOCUMENT_QUERY,
        params={"filename": filename, "file_hash": file_hash or ""},
    )
    return dict(results[0]) if results else None


def get_document_chunks(doc_id: str, graph: Any) -> List[Dict[str, Any]]:
    """Return all chunks for a given document ordered by chunk_index."""
    results = graph.query(_GET_DOCUMENT_CHUNKS_QUERY, params={"doc_id": doc_id})
    return [dict(r) for r in (results or [])]


def list_documents(graph: Any) -> List[Dict[str, Any]]:
    """Return a list of all Document nodes (metadata only, no embeddings)."""
    results = graph.query(_LIST_DOCUMENTS_QUERY)
    return [dict(r) for r in (results or [])]


def update_document(doc_id: str, description: str, graph: Any) -> bool:
    """Update description metadata of a Document node."""
    graph.query(_UPDATE_DOCUMENT_QUERY, params={"doc_id": doc_id, "description": description})
    return True


def delete_document(doc_id: str, graph: Any) -> bool:
    """
    Delete a Document and all its connected DocumentChunk nodes.
    Returns True if deleted, False if document was not found.
    """
    exists = graph.query(_GET_DOCUMENT_QUERY, params={"doc_id": doc_id})
    if not exists:
        return False
    graph.query(_DELETE_DOCUMENT_QUERY, params={"doc_id": doc_id})
    logger.info("Deleted document %s and all its chunks.", doc_id)
    return True


def is_apoc_available(graph: Any) -> bool:
    """Check if APOC procedures are available in the connected Neo4j instance."""
    try:
        results = graph.query("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(name) AS c")
        if results and results[0].get("c", 0) > 0:
            return True
    except Exception as e:
        logger.debug("Could not verify APOC availability: %s", e)
    return False


# ---------------------------------------------------------------------------
# High-Level Document Loading & Splitting Dispatcher
# ---------------------------------------------------------------------------
def load_document(file_path: str, filename: str) -> List[Document]:
    """
    Load a single file into LangChain Document objects by routing to the appropriate processor:
    - Text documents (.pdf, .docx, .txt, .md) -> utils.text_processor
    - Tabular data (.csv, .xlsx, .xls) -> utils.tabular_processor
    - Presentations (.pptx, .ppt) and Images (.png, .jpg, etc.) -> utils.media_processor
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in TEXT_EXTENSIONS:
        docs = load_text_document(file_path, filename)
    elif ext in TABULAR_EXTENSIONS:
        docs = load_tabular_document(file_path, filename)
    elif ext in MEDIA_EXTENSIONS:
        docs = load_media_document(file_path, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loaded %d pages/sections from '%s'", len(docs), filename)
    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split loaded documents into fixed-size overlapping chunks.
    Pre-structured tabular documents (.csv, .xlsx, .xls) are preserved as-is.
    Slide documents and Image OCR documents are preserved or split if exceeding chunk_size.
    """
    tabular_docs = [d for d in documents if d.metadata.get("is_tabular")]
    non_tabular_docs = [d for d in documents if not d.metadata.get("is_tabular")]

    chunks: List[Document] = []

    if non_tabular_docs:
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=DEFAULT_CHUNK_SEPARATOR,
        )
        split_chunks = splitter.split_documents(non_tabular_docs)
        chunks.extend(split_chunks)
        logger.info(
            "Processed %d documents → %d chunks (size=%d, overlap=%d)",
            len(non_tabular_docs),
            len(split_chunks),
            chunk_size,
            chunk_overlap,
        )

    if tabular_docs:
        chunks.extend(tabular_docs)
        logger.info("Preserved %d pre-structured tabular chunks", len(tabular_docs))

    return chunks


# ---------------------------------------------------------------------------
# Vector Embedding & Neo4j Storage
# ---------------------------------------------------------------------------
def _compute_embeddings_batched(
    texts: List[str],
    embedder: Any,
    batch_size: int = EMBED_BATCH_SIZE,
    max_workers: int = MAX_EMBED_WORKERS,
) -> List[List[float]]:
    """
    Compute embeddings for a list of texts in micro-batches with optional concurrency.
    Preserves strict original order of texts.
    """
    if not texts:
        return []

    batches = [
        (idx, texts[i : i + batch_size])
        for idx, i in enumerate(range(0, len(texts), batch_size))
    ]
    num_batches = len(batches)
    effective_workers = max(1, min(max_workers, num_batches))

    # Fast path: single worker or single batch
    if effective_workers <= 1 or num_batches <= 1:
        all_embeddings: List[List[float]] = []
        for _, batch_texts in batches:
            all_embeddings.extend(embedder.embed_documents(batch_texts))
        return all_embeddings

    # Concurrent micro-batches
    batch_results: List[Tuple[int, List[List[float]]]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_idx = {
            executor.submit(embedder.embed_documents, batch_texts): idx
            for idx, batch_texts in batches
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch_results.append((idx, future.result()))

    # Sort back by batch index to guarantee original chunk ordering
    batch_results.sort(key=lambda x: x[0])
    return [emb for _, batch_embs in batch_results for emb in batch_embs]


def embed_and_store_chunks(
    chunks: List[Document],
    doc_id: str,
    filename: str,
    user_id: str,
    description: str,
    graph: Any,
    embedder: Any,
    file_hash: Optional[str] = None,
    max_workers: int = MAX_EMBED_WORKERS,
) -> Dict[str, Any]:
    """
    Compute embeddings for all chunks and write them in batches to Neo4j.

    Creates:
        (d:Document {id, filename, file_type, upload_date, user_id, chunk_count, description, file_hash})
        (c:DocumentChunk {id, content, chunk_index, source, embedding})
        (d)-[:HAS_CHUNK]->(c)
    """
    file_ext = os.path.splitext(filename)[1].lower().lstrip(".")
    upload_date = datetime.now(timezone.utc).isoformat()
    chunk_count = len(chunks)

    # 1. Upsert parent Document node
    graph.query(
        _UPSERT_DOCUMENT_QUERY,
        params={
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_ext,
            "upload_date": upload_date,
            "user_id": user_id,
            "chunk_count": chunk_count,
            "description": description,
            "file_hash": file_hash or "",
        },
    )
    logger.info("Document node upserted: id=%s, filename=%s", doc_id, filename)

    # 2. Compute embeddings concurrently
    texts = [c.page_content for c in chunks]
    logger.info("Computing %d embeddings for '%s'...", len(texts), filename)
    all_embeddings = _compute_embeddings_batched(
        texts=texts,
        embedder=embedder,
        batch_size=EMBED_BATCH_SIZE,
        max_workers=max_workers,
    )

    # 3. Build chunk records
    chunk_records = [
        {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "content": chunk.page_content,
            "chunk_index": idx,
            "source": chunk.metadata.get("source", filename),
            "embedding": embedding,
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    # 4. Batch-write chunks to Neo4j
    for i in range(0, len(chunk_records), WRITE_BATCH_SIZE):
        batch = chunk_records[i : i + WRITE_BATCH_SIZE]
        graph.query(_INSERT_CHUNKS_QUERY, params={"chunks": batch})

    logger.info("Stored %d chunks for document '%s' (id=%s)", chunk_count, filename, doc_id)
    return {"doc_id": doc_id, "filename": filename, "chunk_count": chunk_count}


# ---------------------------------------------------------------------------
# High-Level Upload Pipeline
# ---------------------------------------------------------------------------
def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    description: str,
    graph: Any,
    embedder: Any,
    temp_dir: str = DEFAULT_TEMP_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
    engine: str = "pandas",
) -> Dict[str, Any]:
    """
    End-to-end pipeline for a single uploaded file:
        bytes → duplicate check → temp file → load → chunk → embed → store → cleanup

    Supports engine="pandas" (default, structured RAG tables) and engine="apoc" (database-side batch loading).
    If the document already exists in Neo4j (by filename or file hash) and force=False,
    skips processing and returns status='skipped'.
    """
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for existing duplicate document
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

    doc_id = str(uuid.uuid4())

    # Branch 1: APOC engine for CSV files
    if engine.lower() == "apoc" and filename.lower().endswith(".csv"):
        result = ingest_csv_with_apoc(
            file_bytes=file_bytes,
            filename=filename,
            doc_id=doc_id,
            user_id=user_id,
            description=description,
            graph=graph,
            embedder=embedder,
            file_hash=file_hash,
        )
        result["status"] = "success"
        result["message"] = f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks using APOC engine."
        return result

    # Branch 2: Standard Pandas / LangChain engine
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{doc_id}_{filename}")

    try:
        with open(temp_path, "wb") as fh:
            fh.write(file_bytes)

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
        result["engine"] = "pandas"
        result["message"] = f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks."
        return result

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.warning("Could not remove temp file '%s': %s", temp_path, err)
