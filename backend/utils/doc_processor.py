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
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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
MAX_TABULAR_ROWS_PER_CHUNK = 30  # rows per spreadsheet/CSV chunk
MAX_TABULAR_CHARS_PER_CHUNK = 2000  # maximum chars per spreadsheet chunk

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls"}

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
# Tabular Data Helpers (CSV & Excel)
# ---------------------------------------------------------------------------

def _format_dataframe_as_markdown(sub_df: pd.DataFrame) -> str:
    """
    Format a DataFrame slice into a clean Markdown table with headers,
    escaping special pipe characters and handling linebreaks.
    """
    cols = [f"{c}".strip() for c in sub_df.columns]
    header_line = "| " + " | ".join(cols) + " |"
    separator_line = "| " + " | ".join(["---"] * len(cols)) + " |"
    
    row_lines = []
    for _, row in sub_df.iterrows():
        row_vals = [
            f"{val}".replace("\n", " ").replace("|", "\\|").strip()
            if pd.notnull(val) and val != ""
            else ""
            for val in row
        ]
        row_lines.append("| " + " | ".join(row_vals) + " |")
        
    return "\n".join([header_line, separator_line] + row_lines)


def _dataframe_to_chunks(
    df: pd.DataFrame,
    filename: str,
    sheet_name: Optional[str] = None,
    max_rows_per_chunk: int = MAX_TABULAR_ROWS_PER_CHUNK,
    max_chars_per_chunk: int = MAX_TABULAR_CHARS_PER_CHUNK,
) -> List[Document]:
    """
    Convert a Pandas DataFrame into a list of structured LangChain Document chunks.
    Each chunk preserves column headers in Markdown table format and tracks row ranges.
    """
    sheet_info = f"Sheet: '{sheet_name}'" if sheet_name else "Table"
    
    if df.empty or len(df.columns) == 0:
        source_label = f"{filename} ({sheet_info})" if sheet_name else filename
        content = f"# {sheet_info} in {filename}\n*(Empty spreadsheet/table or no rows)*"
        return [
            Document(
                page_content=content,
                metadata={
                    "source": source_label,
                    "filename": filename,
                    "sheet_name": sheet_name or "",
                    "is_tabular": True,
                    "total_rows": 0,
                },
            )
        ]

    # Clean column names
    df.columns = [f"{c}".strip() or f"Col_{i+1}" for i, c in enumerate(df.columns)]
    cols = [f"{c}" for c in df.columns]
    total_rows = len(df)
    
    docs: List[Document] = []
    start_idx = 0

    while start_idx < total_rows:
        # Determine slice row bound
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        sub_df = df.iloc[start_idx:end_idx]
        table_md = _format_dataframe_as_markdown(sub_df)
        
        # If markdown text is larger than max_chars_per_chunk and we have multiple rows, reduce slice size
        if len(table_md) > max_chars_per_chunk and (end_idx - start_idx) > 5:
            # Reduce row count adaptively
            reduced_rows = max(5, (end_idx - start_idx) // 2)
            end_idx = start_idx + reduced_rows
            sub_df = df.iloc[start_idx:end_idx]
            table_md = _format_dataframe_as_markdown(sub_df)

        row_start_1indexed = start_idx + 1
        row_end_1indexed = end_idx
        
        # Build informative chunk context header
        sheet_desc = f" (Sheet: {sheet_name})" if sheet_name else ""
        source_label = (
            f"{filename} (Sheet: {sheet_name}, Rows {row_start_1indexed}-{row_end_1indexed})"
            if sheet_name
            else f"{filename} (Rows {row_start_1indexed}-{row_end_1indexed})"
        )
        
        header_text = (
            f"# Tabular Data: {filename}{sheet_desc}\n"
            f"**Rows {row_start_1indexed} to {row_end_1indexed} of {total_rows}** | "
            f"**Columns ({len(cols)}):** {', '.join(cols)}\n\n"
        )
        page_content = f"{header_text}{table_md}"
        
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": source_label,
                    "filename": filename,
                    "sheet_name": sheet_name or "",
                    "row_start": row_start_1indexed,
                    "row_end": row_end_1indexed,
                    "total_rows": total_rows,
                    "columns": cols,
                    "is_tabular": True,
                },
            )
        )
        start_idx = end_idx

    logger.info(
        "Converted DataFrame [%s%s] (%d rows, %d cols) → %d chunks",
        filename,
        f", Sheet: {sheet_name}" if sheet_name else "",
        total_rows,
        len(cols),
        len(docs),
    )
    return docs


def _load_csv(file_path: str, filename: str) -> List[Document]:
    """Load a CSV file into structured Document chunks using Pandas."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    df = None
    last_err = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        raise ValueError(f"Could not parse CSV file '{filename}': {last_err}")

    return _dataframe_to_chunks(df, filename=filename)


def _load_excel(file_path: str, filename: str) -> List[Document]:
    """Load an Excel (.xlsx, .xls) file into structured Document chunks across all sheets."""
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open Excel file '{filename}': {exc}")

    all_docs: List[Document] = []
    sheet_names = excel_file.sheet_names

    for sheet_name in sheet_names:
        s_name_str = str(sheet_name)
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            sheet_docs = _dataframe_to_chunks(df, filename=filename, sheet_name=s_name_str)
            all_docs.extend(sheet_docs)
        except Exception as sheet_err:
            logger.warning("Error reading sheet '%s' in '%s': %s", s_name_str, filename, sheet_err)
            # Create a placeholder chunk for the failed sheet
            all_docs.append(
                Document(
                    page_content=f"# Sheet: {s_name_str} in {filename}\n*(Could not parse sheet: {sheet_err})*",
                    metadata={
                        "source": f"{filename} (Sheet: {s_name_str})",
                        "filename": filename,
                        "sheet_name": s_name_str,
                        "is_tabular": True,
                        "total_rows": 0,
                    },
                )
            )

    return all_docs


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_document(file_path: str, filename: str) -> List[Document]:
    """
    Load a single file into LangChain Document objects.

    Supports .pdf, .docx, .txt, .md, .csv, .xlsx, and .xls files.
    Raises ValueError for unsupported extensions.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)  # type: ignore[abstract]
        docs = loader.load()
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif ext == ".csv":
        docs = _load_csv(file_path, filename)
    elif ext in (".xlsx", ".xls"):
        docs = _load_excel(file_path, filename)
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
    Tabular documents (.csv, .xlsx, .xls) are already structured with table headers
    and are preserved as-is.
    """
    # Separate tabular documents from text documents
    tabular_docs = [d for d in documents if d.metadata.get("is_tabular")]
    text_docs = [d for d in documents if not d.metadata.get("is_tabular")]

    chunks: List[Document] = []

    if text_docs:
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=DEFAULT_CHUNK_SEPARATOR,
        )
        split_chunks = splitter.split_documents(text_docs)
        chunks.extend(split_chunks)
        logger.info("Split %d text docs → %d chunks (size=%d, overlap=%d)",
                    len(text_docs), len(split_chunks), chunk_size, chunk_overlap)

    if tabular_docs:
        chunks.extend(tabular_docs)
        logger.info("Preserved %d pre-structured tabular chunks", len(tabular_docs))

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


def is_apoc_available(graph: Any) -> bool:
    """Check if APOC procedures are available in the connected Neo4j instance."""
    try:
        results = graph.query("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(name) AS c")
        if results and results[0].get("c", 0) > 0:
            return True
    except Exception as e:
        logger.debug("Could not verify APOC availability: %s", e)
    return False


def ingest_csv_with_apoc(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
    user_id: str,
    description: str,
    graph: Any,
    embedder: Any,
    file_hash: Optional[str] = None,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Ingest a CSV file using APOC (Awesome Procedures on Cypher).
    Uses apoc.periodic.iterate / UNWIND batches and embeds the generated chunks.
    """
    import csv
    import io
    from setup.init_config import embed_missing_nodes

    file_ext = "csv"
    upload_date = datetime.now(timezone.utc).isoformat()

    # 1. Upsert parent Document node
    graph.query(
        _UPSERT_DOCUMENT_QUERY,
        params={
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_ext,
            "upload_date": upload_date,
            "user_id": user_id,
            "chunk_count": 0,
            "description": description,
            "file_hash": file_hash or "",
        },
    )

    # 2. Decode CSV bytes safely
    text = None
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            text = file_bytes.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for idx, row in enumerate(reader):
        clean_row = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}
        rows.append({"index": idx + 1, "data": clean_row})

    total_rows = len(rows)
    logger.info("Ingesting %d rows with APOC engine for '%s'", total_rows, filename)

    # 3. Run APOC periodic iterate or batch UNWIND query
    _APOC_BATCH_INSERT_QUERY = """
    CALL apoc.periodic.iterate(
        "UNWIND $rows AS r RETURN r",
        "MATCH (d:Document {id: $doc_id})
         CREATE (chunk:DocumentChunk {
             id: apoc.create.uuid(),
             doc_id: $doc_id,
             content: apoc.convert.toJson(r.data),
             chunk_index: r.index,
             source: $filename + ' (Row ' + toString(r.index) + ')'
         })
         MERGE (d)-[:HAS_CHUNK]->(chunk)",
        {batchSize: $batch_size, parallel: false, params: {rows: $rows, doc_id: $doc_id, filename: $filename}}
    )
    YIELD batches, total, errorMessages
    RETURN batches, total, errorMessages
    """

    try:
        res = graph.query(
            _APOC_BATCH_INSERT_QUERY,
            params={
                "rows": rows,
                "doc_id": doc_id,
                "filename": filename,
                "batch_size": batch_size,
            },
        )
        logger.info("APOC periodic iterate completed: %s", res)
    except Exception as apoc_err:
        logger.warning("APOC procedure failed (%s); falling back to standard batch UNWIND query.", apoc_err)
        for i in range(0, total_rows, WRITE_BATCH_SIZE):
            sub_batch = rows[i : i + WRITE_BATCH_SIZE]
            chunk_batch = [
                {
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "content": json.dumps(r["data"]),
                    "chunk_index": r["index"],
                    "source": f"{filename} (Row {r['index']})",
                    "embedding": None,
                }
                for r in sub_batch
            ]
            graph.query(_INSERT_CHUNKS_QUERY, params={"chunks": chunk_batch})

    # 4. Update Document node chunk count
    graph.query(
        """
        MATCH (d:Document {id: $doc_id})
        MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
        WITH d, count(c) AS total_chunks
        SET d.chunk_count = total_chunks
        RETURN total_chunks
        """,
        params={"doc_id": doc_id},
    )

    # 5. Populate vector embeddings for all newly created DocumentChunk nodes
    logger.info("Generating embeddings for APOC-ingested chunks...")
    try:
        embedded_count = embed_missing_nodes(driver=graph, batch_size=EMBED_BATCH_SIZE)
        logger.info("Embedded %d missing nodes for APOC document '%s'", embedded_count, filename)
    except Exception as emb_err:
        logger.warning("Embedding generation for APOC chunks warning: %s", emb_err)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": total_rows,
        "engine": "apoc",
    }


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
        result["engine"] = "pandas"
        result["message"] = f"Document '{filename}' ingested successfully with {result['chunk_count']} chunks."
        return result

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
