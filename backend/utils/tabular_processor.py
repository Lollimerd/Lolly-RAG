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
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TABULAR_ROWS_PER_CHUNK = 20  # Manageable record count for semantic dense embeddings
DEFAULT_TABULAR_ROW_OVERLAP = 2     # Sliding overlap between consecutive chunks
MAX_TABULAR_CHARS_PER_CHUNK = 2500  # Soft character limit for tabular chunks
EMBED_BATCH_SIZE = 32  # Micro-batch size per embedding API call
WRITE_BATCH_SIZE = 50  # Chunks per Neo4j transaction

TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# ---------------------------------------------------------------------------
# Cypher Queries for Tabular & APOC Ingestion
# ---------------------------------------------------------------------------
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

_UPDATE_CHUNK_COUNT_QUERY = """
MATCH (d:Document {id: $doc_id})
MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
WITH d, count(c) AS total_chunks
SET d.chunk_count = total_chunks
RETURN total_chunks
"""

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


# ---------------------------------------------------------------------------
# Hierarchical Schema Analysis & Semantic Record Serialization
# ---------------------------------------------------------------------------

def _format_cell_value(val: Any) -> str:
    """Safely format a DataFrame cell value for semantic text serialization."""
    if pd.isnull(val) or val is None:
        return ""
    val_str = str(val).strip().replace("\n", " ").replace("\r", "")
    return val_str


def _generate_table_schema_summary(
    df: pd.DataFrame,
    filename: str,
    sheet_name: Optional[str] = None,
) -> Document:
    """
    Generate a Level 1 Table Schema & Summary Chunk (Chunk 0).
    Captures:
    - Overall dataset dimensions (rows, columns)
    - Column data types and non-null statistics
    - Categorical/enum distinct sample values
    - Numerical summary statistics (min, max, mean/ranges)
    - Date ranges
    - Top sample records serialized in Key-Value format
    """
    sheet_desc = f" [Sheet: '{sheet_name}']" if sheet_name else ""
    total_rows = len(df)
    cols = list(df.columns)

    if total_rows == 0 or len(cols) == 0:
        content = (
            f"# Tabular Dataset Overview: {filename}{sheet_desc}\n"
            f"*(Empty dataset with 0 rows or 0 columns)*\n"
        )
        return Document(
            page_content=content,
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

    # 1. Column Schemas & Statistics
    col_summaries: List[str] = []
    for col in cols:
        series = df[col]
        non_null_cnt = series.count()
        null_cnt = total_rows - non_null_cnt
        null_note = f", {null_cnt} missing" if null_cnt > 0 else ""

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if not clean_series.empty:
                min_v = clean_series.min()
                max_v = clean_series.max()
                mean_v = clean_series.mean()
                if pd.api.types.is_integer_dtype(clean_series):
                    stat_info = f"Range: {int(min_v)} to {int(max_v)}, Mean: {mean_v:.2f}{null_note}"
                else:
                    stat_info = f"Range: {min_v:.2f} to {max_v:.2f}, Mean: {mean_v:.2f}{null_note}"
                col_summaries.append(f"- **{col}** (Numeric): {stat_info}")
            else:
                col_summaries.append(f"- **{col}** (Numeric): All values missing")

        # Check datetime
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean_series = series.dropna()
            if not clean_series.empty:
                min_d = clean_series.min()
                max_d = clean_series.max()
                col_summaries.append(f"- **{col}** (Date/Time): Range from {min_d} to {max_d}{null_note}")
            else:
                col_summaries.append(f"- **{col}** (Date/Time): All values missing")

        # Categorical / String / Boolean
        else:
            unique_vals = series.dropna().unique()
            n_unique = len(unique_vals)
            if n_unique <= 15:
                samples = [f"'{_format_cell_value(v)}'" for v in unique_vals[:15] if _format_cell_value(v)]
                col_summaries.append(f"- **{col}** (Categorical, {n_unique} values): {', '.join(samples)}{null_note}")
            else:
                samples = [f"'{_format_cell_value(v)}'" for v in unique_vals[:6] if _format_cell_value(v)]
                col_summaries.append(f"- **{col}** (Text, {n_unique} distinct): Samples [{', '.join(samples)}, ...]{null_note}")

    # 2. Sample Records (Top 3)
    sample_records: List[str] = []
    for idx, row in enumerate(df.head(3).itertuples(index=False, name=None)):
        row_kv = [
            f"**{col}**: {_format_cell_value(val)}"
            for col, val in zip(cols, row)
            if _format_cell_value(val) != ""
        ]
        sample_records.append(f"[Sample Record {idx + 1}] " + " | ".join(row_kv))

    summary_parts = [
        f"# Tabular Dataset Overview: {filename}{sheet_desc}\n",
        f"**Dataset Summary:** {total_rows:,} total records across {len(cols)} columns.\n",
        f"**Columns:** {', '.join(cols)}\n\n",
        "### Column Schema & Attributes:\n",
        "\n".join(col_summaries) + "\n\n",
        "### Sample Records:\n",
        "\n".join(sample_records),
    ]

    content = "".join(summary_parts)
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


def _format_records_semantic(
    sub_df: pd.DataFrame,
    start_row_1indexed: int,
    total_rows: int,
    filename: str,
    sheet_name: Optional[str] = None,
) -> str:
    """
    Format a slice of DataFrame rows into structured, semantically rich Key-Value records.
    Key-Value pairs significantly enhance dense embedding retrieval and cross-encoder reranking
    compared to ASCII pipe markdown grids.
    """
    sheet_desc = f" [Sheet: '{sheet_name}']" if sheet_name else ""
    end_row_1indexed = start_row_1indexed + len(sub_df) - 1
    cols = list(sub_df.columns)

    header = (
        f"# Tabular Records: {filename}{sheet_desc}\n"
        f"**Rows {start_row_1indexed} to {end_row_1indexed} of {total_rows}** | "
        f"**Columns ({len(cols)}):** {', '.join(cols)}\n\n"
    )

    record_lines: List[str] = []
    for offset, row in enumerate(sub_df.itertuples(index=False, name=None)):
        current_row_idx = start_row_1indexed + offset
        row_kvs = []
        for col, val in zip(cols, row):
            val_str = _format_cell_value(val)
            if val_str != "":
                row_kvs.append(f"**{col}**: {val_str}")
            else:
                row_kvs.append(f"**{col}**: *(null)*")
        record_lines.append(f"[Row {current_row_idx}] " + " | ".join(row_kvs))

    return header + "\n".join(record_lines)


def _dataframe_to_chunks(
    df: pd.DataFrame,
    filename: str,
    sheet_name: Optional[str] = None,
    max_rows_per_chunk: int = DEFAULT_TABULAR_ROWS_PER_CHUNK,
    row_overlap: int = DEFAULT_TABULAR_ROW_OVERLAP,
    max_chars_per_chunk: int = MAX_TABULAR_CHARS_PER_CHUNK,
) -> List[Document]:
    """
    Convert a Pandas DataFrame into a hierarchical list of structured LangChain Document chunks:
    1. Level 1: Table Schema & Summary Chunk (Chunk 0)
    2. Level 2: Semantic Key-Value Record Chunks with sliding overlap
    """
    # Clean and standardize column names
    df.columns = [c.strip() if isinstance(c, str) else str(c).strip() or f"Col_{i+1}" for i, c in enumerate(df.columns)]
    cols = list(df.columns)
    total_rows = len(df)

    docs: List[Document] = []

    # 1. Level 1: Table Schema & Summary Chunk (Always Chunk 0)
    schema_doc = _generate_table_schema_summary(df, filename=filename, sheet_name=sheet_name)
    docs.append(schema_doc)

    if total_rows == 0:
        return docs

    # 2. Level 2: Semantic Key-Value Record Chunks with sliding overlap
    start_idx = 0
    step_size = max(1, max_rows_per_chunk - row_overlap)

    while start_idx < total_rows:
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        sub_df = df.iloc[start_idx:end_idx]
        record_text = _format_records_semantic(
            sub_df=sub_df,
            start_row_1indexed=start_idx + 1,
            total_rows=total_rows,
            filename=filename,
            sheet_name=sheet_name,
        )

        # Adaptively reduce row slice if generated text exceeds character limit
        if len(record_text) > max_chars_per_chunk and (end_idx - start_idx) > 4:
            reduced_rows = max(4, (end_idx - start_idx) // 2)
            end_idx = start_idx + reduced_rows
            sub_df = df.iloc[start_idx:end_idx]
            record_text = _format_records_semantic(
                sub_df=sub_df,
                start_row_1indexed=start_idx + 1,
                total_rows=total_rows,
                filename=filename,
                sheet_name=sheet_name,
            )

        row_start_1indexed = start_idx + 1
        row_end_1indexed = end_idx

        sheet_desc = f" (Sheet: {sheet_name})" if sheet_name else ""
        source_label = (
            f"{filename} (Sheet: {sheet_name}, Rows {row_start_1indexed}-{row_end_1indexed})"
            if sheet_name
            else f"{filename} (Rows {row_start_1indexed}-{row_end_1indexed})"
        )

        docs.append(
            Document(
                page_content=record_text,
                metadata={
                    "source": source_label,
                    "filename": filename,
                    "sheet_name": sheet_name or "",
                    "row_start": row_start_1indexed,
                    "row_end": row_end_1indexed,
                    "total_rows": total_rows,
                    "columns": cols,
                    "is_tabular": True,
                    "is_table_summary": False,
                },
            )
        )

        # Advance with sliding window overlap
        if end_idx >= total_rows:
            break
        start_idx += step_size

    logger.info(
        "Converted DataFrame [%s%s] (%d rows, %d cols) → %d hierarchical chunks (1 summary + %d record chunks)",
        filename,
        f", Sheet: {sheet_name}" if sheet_name else "",
        total_rows,
        len(cols),
        len(docs),
        len(docs) - 1,
    )
    return docs


def _load_csv(file_path: str, filename: str) -> List[Document]:
    """Load a CSV file into structured Document chunks using Pandas with encoding fallbacks."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    last_err: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            return _dataframe_to_chunks(df, filename=filename)
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not parse CSV file '{filename}': {last_err}")


def _load_excel(file_path: str, filename: str) -> List[Document]:
    """Load an Excel (.xlsx, .xls) file into structured Document chunks across all sheets."""
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open Excel file '{filename}': {exc}")

    all_docs: List[Document] = []
    for sheet_name in excel_file.sheet_names:
        s_name_str = str(sheet_name)
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            sheet_docs = _dataframe_to_chunks(df, filename=filename, sheet_name=s_name_str)
            all_docs.extend(sheet_docs)
        except Exception as sheet_err:
            logger.warning("Error reading sheet '%s' in '%s': %s", s_name_str, filename, sheet_err)
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


def load_tabular_document(file_path: str, filename: str) -> List[Document]:
    """
    Load a tabular file (.csv, .xlsx, .xls) into structured LangChain Document objects.

    Args:
        file_path: Filesystem path to the tabular file.
        filename: Original filename.

    Returns:
        List of structured Document chunks.

    Raises:
        ValueError: If file format is unsupported or parsing fails.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return _load_csv(file_path, filename)
    elif ext in (".xlsx", ".xls"):
        return _load_excel(file_path, filename)
    else:
        raise ValueError(
            f"Unsupported tabular file type '{ext}'. Supported: {', '.join(sorted(TABULAR_EXTENSIONS))}"
        )


# ---------------------------------------------------------------------------
# APOC Database Batch Ingestion
# ---------------------------------------------------------------------------

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
    Ingest a CSV file using APOC periodic iterate / UNWIND batches and embeds generated chunks.
    """
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
    rows = [
        {"index": idx + 1, "data": {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}}
        for idx, row in enumerate(reader)
    ]
    total_rows = len(rows)
    logger.info("Ingesting %d rows with APOC engine for '%s'", total_rows, filename)

    # 3. Execute APOC periodic iterate with fallback to UNWIND
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
    graph.query(_UPDATE_CHUNK_COUNT_QUERY, params={"doc_id": doc_id})

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
