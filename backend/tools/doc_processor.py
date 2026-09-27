from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from utils.media_processor import MEDIA_EXTENSIONS, load_media_document
from utils.tabular_processor import TABULAR_EXTENSIONS, ingest_csv_with_apoc, load_tabular_document
from utils.text_processor import TEXT_EXTENSIONS, load_text_document

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_SEPARATOR = "\n"
EMBED_BATCH_SIZE = 32
MAX_EMBED_WORKERS = int(os.getenv("MAX_EMBED_WORKERS", "4"))
WRITE_BATCH_SIZE = 50
DEFAULT_TEMP_DIR = "/tmp/lolly_rag_uploads"
SUPPORTED_EXTENSIONS = set.union(TEXT_EXTENSIONS, TABULAR_EXTENSIONS, MEDIA_EXTENSIONS)

_UPSERT_DOC = """
MERGE (d:Document {id: $doc_id})
ON CREATE SET d.filename = $filename,
              d.file_type = $file_type,
              d.upload_date = $upload_date,
              d.user_id = $user_id,
              d.chunk_count = $chunk_count,
              d.description = $description,
              d.file_hash = $file_hash
ON MATCH SET d.upload_date = $upload_date,
             d.chunk_count = $chunk_count,
             d.description = $description,
             d.file_hash = $file_hash
RETURN d.id AS id
"""

_INSERT_CHUNKS = """
UNWIND $chunks AS c
MERGE (chunk:DocumentChunk {id: c.id})
ON CREATE SET chunk.content = c.content,
              chunk.chunk_index = c.chunk_index,
              chunk.source = c.source,
              chunk.embedding = c.embedding
WITH chunk, c
MATCH (d:Document {id: c.doc_id})
MERGE (d)-[:HAS_CHUNK]->(chunk)
"""

_LIST_DOCS = """
MATCH (d:Document)
RETURN d.id AS id,
       d.filename AS filename,
       d.file_type AS file_type,
       d.upload_date AS upload_date,
       d.user_id AS user_id,
       d.chunk_count AS chunk_count,
       d.description AS description,
       d.file_hash AS file_hash
ORDER BY d.upload_date DESC
"""

_DEL_DOC = """
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
DETACH DELETE d, c
"""

_GET_DOC = """
MATCH (d:Document {id: $doc_id})
RETURN d.id AS id, d.filename AS filename
"""

_FIND_DOC = """
MATCH (d:Document)
WHERE d.filename = $filename
   OR (d.file_hash IS NOT NULL AND d.file_hash <> '' AND d.file_hash = $file_hash)
RETURN d.id AS id,
       d.filename AS filename,
       d.chunk_count AS chunk_count,
       d.file_hash AS file_hash
LIMIT 1
"""

_GET_CHUNKS = """
MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:DocumentChunk)
RETURN c.id AS id,
       c.chunk_index AS chunk_index,
       c.content AS content,
       c.source AS source
ORDER BY c.chunk_index ASC
"""

_UPDATE_DOC = """
MATCH (d:Document {id: $doc_id})
SET d.description = $description
RETURN d.id AS id
"""

def find_existing_document(filename: str, file_hash: Optional[str], graph: Any) -> Optional[Dict[str, Any]]:
    """Args: filename: File name, file_hash: File hash, graph: Neo4j connection."""
    res = graph.query(
        _FIND_DOC,
        params={
            "filename": filename,
            "file_hash": file_hash or "",
        },
    )
    return dict(res[0]) if res else None

def get_document_chunks(doc_id: str, graph: Any) -> List[Dict[str, Any]]:
    """Args: doc_id: Document ID, graph: Neo4j connection."""
    return [dict(r) for r in (graph.query(_GET_CHUNKS, params={"doc_id": doc_id}) or [])]

def list_documents(graph: Any) -> List[Dict[str, Any]]:
    """Args: graph: Neo4j connection."""
    return [dict(r) for r in (graph.query(_LIST_DOCS) or [])]

def update_document(doc_id: str, description: str, graph: Any) -> bool:
    """Args: doc_id: Document ID, description: Description, graph: Neo4j connection."""
    graph.query(
        _UPDATE_DOC,
        params={
            "doc_id": doc_id,
            "description": description,
        },
    )
    return True

def delete_document(doc_id: str, graph: Any) -> bool:
    """Args: doc_id: Document ID, graph: Neo4j connection."""
    if not graph.query(_GET_DOC, params={"doc_id": doc_id}):
        return False
    graph.query(_DEL_DOC, params={"doc_id": doc_id})
    return True

def load_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None) -> List[Document]:
    """Args: file_path: File path, filename: File name, ocr_engine: Optional OCR engine."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return load_text_document(file_path, filename, ocr_engine=ocr_engine)
    if ext in TABULAR_EXTENSIONS:
        return load_tabular_document(file_path, filename)
    if ext in MEDIA_EXTENSIONS:
        return load_media_document(file_path, filename, ocr_engine=ocr_engine)
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separator: str = DEFAULT_CHUNK_SEPARATOR,
) -> List[Document]:
    """Args: documents: Document objects, chunk_size: Chunk size, chunk_overlap: Overlap, separator: Split separator."""
    pre_chunked = [d for d in documents if d.metadata.get("is_tabular") or d.metadata.get("is_presentation") or d.metadata.get("is_image")]
    to_split = [d for d in documents if not (d.metadata.get("is_tabular") or d.metadata.get("is_presentation") or d.metadata.get("is_image"))]
    split_chunks: List[Document] = []
    if to_split:
        splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_chunks = splitter.split_documents(to_split)
    all_chunks = split_chunks + pre_chunked
    logger.info("Chunked %d document(s) -> %d chunks", len(documents), len(all_chunks))
    return all_chunks

def _resolve_engine(filename: str, engine: Optional[str]) -> str:
    """Args: filename: File name, engine: Engine override or None."""
    if engine and engine.lower() in ("apoc", "pandas"):
        return engine.lower()
    return "apoc" if filename.lower().endswith(".csv") else "pandas"

def _compute_embeddings_batched(texts: List[str], embedder: Any, batch_size: int, max_workers: int, progress_cb: Optional[Callable[[int, int], None]] = None) -> List[Optional[List[float]]]:
    """Args: texts: Text list, embedder: Embedder, batch_size: Batch size, max_workers: Worker count, progress_cb: Callback."""
    total_texts = len(texts)
    results: List[Optional[List[float]]] = [None] * total_texts
    batches = [(idx, texts[idx: idx + batch_size]) for idx in range(0, total_texts, batch_size)]
    completed = 0

    def _embed_slice(start_idx: int, slice_texts: List[str]) -> Tuple[int, List[List[float]]]:
        return start_idx, embedder.embed_documents(slice_texts)

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(batches)))) as executor:
        futures = {executor.submit(_embed_slice, s_idx, b_texts): s_idx for s_idx, b_texts in batches}
        for future in as_completed(futures):
            start_idx, embeddings = future.result()
            for offset, emb in enumerate(embeddings):
                results[start_idx + offset] = emb
            completed += len(embeddings)
            if progress_cb:
                progress_cb(completed, total_texts)
    return results

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
    progress_callback: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Args: chunks: Document chunks, doc_id: Document ID, filename: File name, user_id: User ID, description: Description, graph: Neo4j connection, embedder: Embedder, file_hash: File hash, max_workers: Worker count, progress_callback: Progress callback."""
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    upload_date = datetime.now(timezone.utc).isoformat()
    chunk_count = len(chunks)

    graph.query(
        _UPSERT_DOC,
        params={
            "doc_id": doc_id,
            "filename": filename,
            "file_type": ext,
            "upload_date": upload_date,
            "user_id": user_id,
            "chunk_count": chunk_count,
            "description": description,
            "file_hash": file_hash or "",
        },
    )
    if progress_callback:
        progress_callback("upsert", 0.15, f"Registered '{filename}' in knowledge graph", {"doc_id": doc_id})

    texts = [c.page_content for c in chunks]

    def _embed_cb(done: int, total: int):
        if progress_callback:
            prog = 0.15 + 0.65 * (done / max(1, total))
            progress_callback(
                "embedding",
                prog,
                f"Embedding chunks: {done}/{total}",
                {"completed_chunks": done, "total_chunks": total},
            )

    all_embeddings = _compute_embeddings_batched(texts, embedder, EMBED_BATCH_SIZE, max_workers, _embed_cb)
    records = [
        {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "content": chunk.page_content,
            "chunk_index": idx,
            "source": chunk.metadata.get("source", filename),
            "embedding": emb,
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, all_embeddings))
    ]

    total_w_batches = max(1, (len(records) + WRITE_BATCH_SIZE - 1) // WRITE_BATCH_SIZE)
    written = 0
    for w_idx, i in enumerate(range(0, len(records), WRITE_BATCH_SIZE)):
        batch = records[i: i + WRITE_BATCH_SIZE]
        graph.query(_INSERT_CHUNKS, params={"chunks": batch})
        written += len(batch)
        if progress_callback:
            prog = 0.80 + 0.18 * ((w_idx + 1) / total_w_batches)
            progress_callback(
                "uploading",
                prog,
                f"Writing batch {w_idx + 1}/{total_w_batches} to Neo4j",
                {
                    "written_batches": w_idx + 1,
                    "total_write_batches": total_w_batches,
                    "written_chunks": written,
                },
            )

    logger.info("Stored %d chunks for '%s'", chunk_count, filename)
    if progress_callback:
        progress_callback(
            "complete",
            1.0,
            f"Successfully stored all {chunk_count} chunks in Neo4j",
            {
                "chunk_count": chunk_count,
                "doc_id": doc_id,
            },
        )
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": chunk_count,
    }

def _run_with_queue(fn: Callable[..., Any], success_msg: str, *args, **kwargs) -> Generator[Dict[str, Any], None, None]:
    """Args: fn: Target callable, success_msg: Success template."""
    filename = kwargs.get("filename", "")
    q: queue.Queue[Dict[str, Any]] = queue.Queue()

    def _cb(stage: str, prog: float, msg: str, meta: Dict[str, Any]):
        q.put({
            "type": "progress" if stage != "complete" else "complete",
            "stage": stage,
            "progress": prog,
            "filename": filename,
            "message": msg,
            **meta,
        })

    def _worker():
        try:
            res = fn(*args, progress_callback=_cb, **kwargs)
            q.put({"type": "_done", "result": res})
        except Exception as e:
            q.put({"type": "_error", "error": str(e)})

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    while True:
        try:
            item = q.get(timeout=0.05)
            if item["type"] == "_done":
                res = item["result"]
                yield {
                    "type": "complete",
                    "status": "success",
                    "progress": 1.0,
                    "doc_id": res["doc_id"],
                    "filename": filename,
                    "chunk_count": res.get("chunk_count", 0),
                    "message": success_msg.format(filename=filename, count=res.get("chunk_count", 0)),
                }
                break
            if item["type"] == "_error":
                yield {
                    "type": "error",
                    "status": "error",
                    "progress": 1.0,
                    "filename": filename,
                    "message": f"Ingestion failed: {item['error']}",
                }
                break
            yield item
        except queue.Empty:
            if not th.is_alive() and q.empty():
                break

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
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """Args: file_bytes: File bytes, filename: File name, user_id: User ID, description: Description, graph: Neo4j connection, embedder: Embedder, temp_dir: Temp directory, chunk_size: Chunk size, chunk_overlap: Overlap, force: Overwrite flag, engine: Engine override."""
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = find_existing_document(filename, file_hash, graph)
    if existing and not force:
        return {
            "status": "skipped",
            "doc_id": existing["id"],
            "filename": filename,
            "chunk_count": existing.get("chunk_count", 0),
            "message": f"Document '{filename}' was already ingested.",
        }
    if existing and force:
        delete_document(existing["id"], graph)

    doc_id = str(uuid.uuid4())
    target_engine = _resolve_engine(filename, engine)
    if target_engine == "apoc" and filename.lower().endswith(".csv"):
        result = ingest_csv_with_apoc(file_bytes, filename, doc_id, user_id, description, graph, embedder, file_hash, async_embed=True)
        result.update({
            "status": "success",
            "message": f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks using APOC engine.",
        })
        return result

    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
    try:
        with open(temp_path, "wb") as fh:
            fh.write(file_bytes)
        chunks = chunk_documents(load_document(temp_path, filename), chunk_size, chunk_overlap)
        result = embed_and_store_chunks(chunks, doc_id, filename, user_id, description, graph, embedder, file_hash)
        result.update({
            "status": "success",
            "engine": "pandas",
            "message": f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks.",
        })
        return result
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.warning("Could not remove temp file '%s': %s", temp_path, err)

def process_uploaded_file_stream(
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
    engine: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Args: file_bytes: File bytes, filename: File name, user_id: User ID, description: Description, graph: Neo4j connection, embedder: Embedder, temp_dir: Temp directory, chunk_size: Chunk size, chunk_overlap: Overlap, force: Overwrite flag, engine: Engine override."""
    yield {
        "type": "progress",
        "stage": "init",
        "progress": 0.02,
        "filename": filename,
        "message": f"Checking duplicate records for '{filename}'...",
    }
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = find_existing_document(filename, file_hash, graph)
    if existing and not force:
        yield {
            "type": "complete",
            "status": "skipped",
            "progress": 1.0,
            "doc_id": existing["id"],
            "filename": filename,
            "chunk_count": existing.get("chunk_count", 0),
            "message": f"Document '{filename}' was already ingested.",
        }
        return
    if existing and force:
        delete_document(existing["id"], graph)
        yield {
            "type": "progress",
            "stage": "cleaned",
            "progress": 0.05,
            "filename": filename,
            "message": f"Removed existing version of '{filename}' for re-ingestion.",
        }

    doc_id = str(uuid.uuid4())
    target_engine = _resolve_engine(filename, engine)
    if target_engine == "apoc" and filename.lower().endswith(".csv"):
        yield {
            "type": "progress",
            "stage": "apoc_start",
            "progress": 0.08,
            "filename": filename,
            "message": "Loading CSV rows via APOC into Neo4j...",
        }
        yield from _run_with_queue(
            ingest_csv_with_apoc,
            "Document '{filename}' ingested successfully with {count} chunks using APOC engine.",
            file_bytes=file_bytes,
            filename=filename,
            doc_id=doc_id,
            user_id=user_id,
            description=description,
            graph=graph,
            embedder=embedder,
            file_hash=file_hash,
            async_embed=False,
        )
        return

    yield {
        "type": "progress",
        "stage": "parsing",
        "progress": 0.05,
        "filename": filename,
        "message": f"Loading content and parsing structure for '{filename}'...",
    }
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
    try:
        with open(temp_path, "wb") as fh:
            fh.write(file_bytes)
        documents = load_document(temp_path, filename)
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        yield {
            "type": "progress",
            "stage": "chunking",
            "progress": 0.10,
            "filename": filename,
            "total_chunks": len(chunks),
            "message": f"Extracted {len(documents)} sections -> {len(chunks)} structured chunks",
        }
        yield from _run_with_queue(
            embed_and_store_chunks,
            "Document '{filename}' ingested successfully with {count} chunks.",
            chunks=chunks,
            doc_id=doc_id,
            filename=filename,
            user_id=user_id,
            description=description,
            graph=graph,
            embedder=embedder,
            file_hash=file_hash,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.warning("Could not remove temp file '%s': %s", temp_path, err)
