"""
tabular_processor.py
--------------------
Document loader and structured chunk processor for tabular and spreadsheet data:
- CSV: .csv (via Pandas chunker or APOC database batch loading)
- Excel: .xlsx, .xls (via Pandas across all sheets)

Converts tabular data into structured Markdown table chunks with preserved headers
and tracks row ranges and column names in metadata for precise RAG retrieval.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_TABULAR_ROWS_PER_CHUNK = 20
DEFAULT_TABULAR_ROW_OVERLAP = 2
MAX_TABULAR_CHARS_PER_CHUNK = 2500
EMBED_BATCH_SIZE = 32
WRITE_BATCH_SIZE = 50
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

# Cypher Queries for Tabular & APOC Ingestion
_UPSERT_DOCUMENT_QUERY = """MERGE (d:Document {id: $doc_id})
ON CREATE SET d.filename = $filename, d.file_type = $file_type, d.upload_date = $upload_date,
    d.user_id = $user_id, d.chunk_count = $chunk_count, d.description = $description, d.file_hash = $file_hash
ON MATCH SET d.upload_date = $upload_date, d.chunk_count = $chunk_count, d.description = $description, d.file_hash = $file_hash
RETURN d.id AS id"""

_INSERT_CHUNKS_QUERY = """UNWIND $chunks AS c
MERGE (chunk:DocumentChunk {id: c.id})
ON CREATE SET chunk.content = c.content, chunk.chunk_index = c.chunk_index, chunk.source = c.source, chunk.embedding = c.embedding
WITH chunk, c
MATCH (d:Document {id: c.doc_id})
MERGE (d)-[:HAS_CHUNK]->(chunk)"""

_UPDATE_CHUNK_COUNT_QUERY = """MATCH (d:Document {id: $doc_id})
MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
WITH d, count(c) AS total_chunks
SET d.chunk_count = total_chunks
RETURN total_chunks"""

_APOC_BATCH_INSERT_QUERY = """CALL apoc.periodic.iterate(
    "UNWIND $rows AS r RETURN r",
    "MATCH (d:Document {id: $doc_id})
     CREATE (chunk:DocumentChunk {
         id: apoc.create.uuid(), doc_id: $doc_id, content: apoc.convert.toJson(r.data),
         chunk_index: r.index, source: $filename + ' (Row ' + toString(r.index) + ')'
     })
     MERGE (d)-[:HAS_CHUNK]->(chunk)",
    {batchSize: $batch_size, parallel: false, params: {rows: $rows, doc_id: $doc_id, filename: $filename}}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages"""

def _decode_bytes(data: bytes) -> str:
    """Args: data: Raw byte buffer."""
    for enc in _CSV_ENCODINGS:
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")

def _format_cell_value(val: Any) -> str:
    """Args: val: Cell value."""
    if pd.isnull(val) or val is None:
        return ""
    return str(val).strip().replace("\n", " ").replace("\r", "")

def _generate_table_schema_summary(df: pd.DataFrame, filename: str, sheet_name: Optional[str] = None) -> Document:
    """Args: df: Dataset DataFrame, filename: File name, sheet_name: Optional sheet name."""
    sheet_desc = f" [Sheet: '{sheet_name}']" if sheet_name else ""
    total_rows, cols = len(df), list(df.columns)
    if total_rows == 0 or len(cols) == 0:
        return Document(
            page_content=f"# Tabular Dataset Overview: {filename}{sheet_desc}\n*(Empty dataset with 0 rows or 0 columns)*\n",
            metadata={
                "source": f"{filename} (Dataset Overview{f' - {sheet_name}' if sheet_name else ''})",
                "filename": filename,
                "sheet_name": sheet_name or "",
                "is_tabular": True,
                "is_table_summary": True,
                "total_rows": 0,
                "columns": cols,
            },
        )

    col_summaries: List[str] = []
    for col in cols:
        series = df[col]
        null_cnt = total_rows - series.count()
        null_note = f", {null_cnt} missing" if null_cnt > 0 else ""
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                min_v, max_v, mean_v = clean.min(), clean.max(), clean.mean()
                stat = f"Range: {int(min_v)} to {int(max_v)}" if pd.api.types.is_integer_dtype(clean) else f"Range: {min_v:.2f} to {max_v:.2f}"
                col_summaries.append(f"- **{col}** (Numeric): {stat}, Mean: {mean_v:.2f}{null_note}")
            else:
                col_summaries.append(f"- **{col}** (Numeric): All values missing")
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                col_summaries.append(f"- **{col}** (Date/Time): Range from {clean.min()} to {clean.max()}{null_note}")
            else:
                col_summaries.append(f"- **{col}** (Date/Time): All values missing")
        else:
            uv = series.dropna().unique()
            if len(uv) <= 15:
                samples = [f"'{_format_cell_value(v)}'" for v in uv[:15] if _format_cell_value(v)]
                col_summaries.append(f"- **{col}** (Categorical, {len(uv)} values): {', '.join(samples)}{null_note}")
            else:
                samples = [f"'{_format_cell_value(v)}'" for v in uv[:6] if _format_cell_value(v)]
                col_summaries.append(f"- **{col}** (Text, {len(uv)} distinct): Samples [{', '.join(samples)}, ...]{null_note}")

    sample_records = [
        f"[Sample Record {idx + 1}] " + " | ".join(f"**{col}**: {_format_cell_value(val)}" for col, val in zip(cols, row) if _format_cell_value(val))
        for idx, row in enumerate(df.head(3).itertuples(index=False, name=None))
    ]
    overview = f"# Tabular Dataset Overview: {filename}{sheet_desc}\n**Dataset Summary:** {total_rows:,} total records across {len(cols)} columns.\n**Columns:** {', '.join(cols)}\n\n"
    content = overview + "### Column Schema & Attributes:\n" + "\n".join(col_summaries) + "\n\n### Sample Records:\n" + "\n".join(sample_records)
    return Document(
        page_content=content,
        metadata={
            "source": f"{filename} (Dataset Overview{f' - {sheet_name}' if sheet_name else ''})",
            "filename": filename,
            "sheet_name": sheet_name or "",
            "is_tabular": True,
            "is_table_summary": True,
            "total_rows": total_rows,
            "columns": cols,
        },
    )

def _format_records_semantic(sub_df: pd.DataFrame, start_row_1indexed: int, total_rows: int, filename: str, sheet_name: Optional[str] = None) -> str:
    """Args: sub_df: Slice DataFrame, start_row_1indexed: Start row index, total_rows: Total rows, filename: File name, sheet_name: Optional sheet name."""
    sheet_desc = f" [Sheet: '{sheet_name}']" if sheet_name else ""
    end_row = start_row_1indexed + len(sub_df) - 1
    cols = list(sub_df.columns)
    header = f"# Tabular Records: {filename}{sheet_desc}\n**Rows {start_row_1indexed} to {end_row} of {total_rows}** | **Columns ({len(cols)}):** {', '.join(cols)}\n\n"
    lines = [
        f"[Row {start_row_1indexed + offset}] " + " | ".join(f"**{col}**: {(_format_cell_value(val) or '*(null)*')}" for col, val in zip(cols, row))
        for offset, row in enumerate(sub_df.itertuples(index=False, name=None))
    ]
    return header + "\n".join(lines)

def _dataframe_to_chunks(
    df: pd.DataFrame,
    filename: str,
    sheet_name: Optional[str] = None,
    max_rows_per_chunk: int = DEFAULT_TABULAR_ROWS_PER_CHUNK,
    row_overlap: int = DEFAULT_TABULAR_ROW_OVERLAP,
    max_chars_per_chunk: int = MAX_TABULAR_CHARS_PER_CHUNK,
) -> List[Document]:
    """Args: df: DataFrame, filename: File name, sheet_name: Optional sheet name, max_rows_per_chunk: Row chunk size, row_overlap: Row overlap, max_chars_per_chunk: Character limit."""
    df.columns = [c.strip() if isinstance(c, str) else str(c).strip() or f"Col_{i+1}" for i, c in enumerate(df.columns)]
    cols, total_rows = list(df.columns), len(df)
    docs = [_generate_table_schema_summary(df, filename=filename, sheet_name=sheet_name)]
    if total_rows == 0:
        return docs

    start_idx, step_size = 0, max(1, max_rows_per_chunk - row_overlap)
    while start_idx < total_rows:
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        sub_df = df.iloc[start_idx:end_idx]
        text = _format_records_semantic(sub_df, start_idx + 1, total_rows, filename, sheet_name)
        if len(text) > max_chars_per_chunk and (end_idx - start_idx) > 4:
            end_idx = start_idx + max(4, (end_idx - start_idx) // 2)
            sub_df = df.iloc[start_idx:end_idx]
            text = _format_records_semantic(sub_df, start_idx + 1, total_rows, filename, sheet_name)

        label = f"{filename} (Sheet: {sheet_name}, Rows {start_idx + 1}-{end_idx})" if sheet_name else f"{filename} (Rows {start_idx + 1}-{end_idx})"
        docs.append(Document(
            page_content=text,
            metadata={
                "source": label,
                "filename": filename,
                "sheet_name": sheet_name or "",
                "row_start": start_idx + 1,
                "row_end": end_idx,
                "total_rows": total_rows,
                "columns": cols,
                "is_tabular": True,
                "is_table_summary": False,
            },
        ))
        if end_idx >= total_rows:
            break
        start_idx += step_size

    logger.info("Converted [%s] (%d rows, %d cols) -> %d chunks", filename, total_rows, len(cols), len(docs))
    return docs

def _load_csv(file_path: str, filename: str) -> List[Document]:
    """Args: file_path: CSV file path, filename: File name."""
    last_err = None
    for enc in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            return _dataframe_to_chunks(df, filename=filename)
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not parse CSV file '{filename}': {last_err}")

def _load_excel(file_path: str, filename: str) -> List[Document]:
    """Args: file_path: Excel file path, filename: File name."""
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open Excel file '{filename}': {exc}")
    all_docs: List[Document] = []
    for sheet_name in excel_file.sheet_names:
        s_name = str(sheet_name)
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            all_docs.extend(_dataframe_to_chunks(df, filename=filename, sheet_name=s_name))
        except Exception as err:
            logger.warning("Error reading sheet '%s' in '%s': %s", s_name, filename, err)
            all_docs.append(Document(
                page_content=f"# Sheet: {s_name} in {filename}\n*(Could not parse: {err})*",
                metadata={"source": f"{filename} (Sheet: {s_name})", "filename": filename, "sheet_name": s_name, "is_tabular": True, "total_rows": 0},
            ))
    return all_docs

def load_tabular_document(file_path: str, filename: str) -> List[Document]:
    """Args: file_path: Tabular file path, filename: File name."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return _load_csv(file_path, filename)
    if ext in (".xlsx", ".xls"):
        return _load_excel(file_path, filename)
    raise ValueError(f"Unsupported tabular file type '{ext}'. Supported: {', '.join(sorted(TABULAR_EXTENSIONS))}")

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
    async_embed: bool = False,
    progress_callback: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Args: file_bytes: File bytes
    filename: File name
    doc_id: Document ID
    user_id: User ID
    description: Description
    graph: Neo4j connection
    embedder: Embedder
    file_hash: File hash
    batch_size: APOC batch size
    async_embed: Background embedding flag
    progress_callback: Progress callback
    """
    from setup.init_config import embed_missing_nodes
    upload_date = datetime.now(timezone.utc).isoformat()

    graph.query(
        _UPSERT_DOCUMENT_QUERY,
        params={
            "doc_id": doc_id, 
            "filename": filename, 
            "file_type": "csv", 
            "upload_date": upload_date, 
            "user_id": user_id, 
            "chunk_count": 0, 
            "description": description, 
            "file_hash": file_hash or ""
            },
    )
    if progress_callback:
        progress_callback("upsert", 0.05, f"Registered CSV document '{filename}' in knowledge graph", {"doc_id": doc_id})

    text = _decode_bytes(file_bytes)
    reader = csv.DictReader(io.StringIO(text))
    rows = [{"index": idx + 1, "data": {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}} for idx, row in enumerate(reader)]
    total_rows = len(rows)
    logger.info("Ingesting %d rows with APOC engine for '%s'", total_rows, filename)

    try:
        res = graph.query(_APOC_BATCH_INSERT_QUERY, params={"rows": rows, "doc_id": doc_id, "filename": filename, "batch_size": batch_size})
        logger.info("APOC periodic iterate completed: %s", res)
    except Exception as apoc_err:
        logger.warning("APOC procedure failed (%s); falling back to standard batch UNWIND query.", apoc_err)
        for i in range(0, total_rows, WRITE_BATCH_SIZE):
            sub_batch = rows[i : i + WRITE_BATCH_SIZE]
            chunk_batch = [{"id": str(uuid.uuid4()), "doc_id": doc_id, "content": json.dumps(r["data"]), "chunk_index": r["index"], "source": f"{filename} (Row {r['index']})", "embedding": None} for r in sub_batch]
            graph.query(_INSERT_CHUNKS_QUERY, params={"chunks": chunk_batch})

    graph.query(_UPDATE_CHUNK_COUNT_QUERY, params={"doc_id": doc_id})
    if progress_callback:
        progress_callback("uploading", 0.20, f"Loaded {total_rows} rows via APOC into Neo4j graph database", {"rows": total_rows})

    def _apoc_embed_cb(completed_b: int, total_b: int, completed_c: int, total_c: int):
        if progress_callback:
            prog = 0.20 + 0.78 * (completed_b / max(1, total_b))
            progress_callback("embedding", prog, f"Embedded batch {completed_b}/{total_b} ({completed_c}/{total_c} chunks) across worker threads", {"completed_batches": completed_b, "total_batches": total_b, "completed_chunks": completed_c, "total_chunks": total_c})

    def _run_background_embed():
        logger.info("Starting embedding for APOC document '%s' (id=%s)...", filename, doc_id)
        try:
            embedded_count = embed_missing_nodes(driver=graph, batch_size=EMBED_BATCH_SIZE, doc_id=doc_id, progress_callback=_apoc_embed_cb)
            logger.info("Completed embedding for APOC document '%s': %d nodes embedded.", filename, embedded_count)
            if progress_callback:
                progress_callback("complete", 1.0, f"Successfully embedded and stored all {total_rows} chunks via APOC", {"chunk_count": total_rows, "doc_id": doc_id})
        except Exception as emb_err:
            logger.warning("Embedding generation for APOC chunks warning: %s", emb_err)

    if async_embed:
        threading.Thread(target=_run_background_embed, name=f"apoc-embed-{doc_id[:8]}", daemon=True).start()
        logger.info("Dispatched concurrent background embedding thread for APOC document '%s' (id=%s).", filename, doc_id)
    else:
        _run_background_embed()

    return {"doc_id": doc_id, 
    "filename": filename, 
    "chunk_count": total_rows, 
    "engine": "apoc", 
    "embedding_mode": "concurrent" if async_embed else "synchronous"
    }
