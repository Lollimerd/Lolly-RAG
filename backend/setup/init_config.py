"""Setting up ollama models, vectorstores and Neo4j Configs"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from typing import Any, Dict, List, Optional, Tuple
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

# ===========================================================================================================================================================
# Step 1: Load Configuration: Docker, Neo4j, Ollama, Langchain
# ===========================================================================================================================================================

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


def answer_LLM():
    """main model for RAG agent"""
    return ChatOllama(
        model="qwen3.5:4b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=40968,
        num_predict=8192,  # max tokens in answer
        temperature=0.7,  # balanced creativity
        repeat_penalty=1.1,  # standard mild penalty
        repeat_last_n=64,  # look back 64 tokens (-1 penalized entire 40k context)
        top_p=0.95,  # nucleus sampling
        top_k=40,  # standard candidate pool
        reasoning=True,
        tags=["answer_llm"],
    )

@lru_cache(maxsize=1)
def embedding_model():
    """embedding model"""
    return OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,  # 16k context
    )

@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Dynamically determine the embedding dimension of the configured embedding model."""
    try:
        embedder = embedding_model()
        test_vector = embedder.embed_query("dimension_probe")
        dim = len(test_vector)
        logger.info(f"Detected embedding model dimension: {dim}")
        return dim
    except Exception as e:
        logger.warning(f"Could not probe embedding model dimension, defaulting to 1024: {e}")
        return 1024

@lru_cache(maxsize=1)
def reranker_model():
    """reranker model"""
    import torch
    return HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={
            "device": "cuda",  # Use 'cuda' for GPU acceleration
        },
    )

def summarizer():
    """summarizes historical context"""
    return ChatOllama(
        model="qwen3.5:0.8b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=8192,  # 40k context
        tags=["summarizer_llm"],
    )

@lru_cache(maxsize=1)
def ocr_model():
    """
    OCR Engine / Model configuration for text extraction from images and scanned documents.
    Provides a unified interface `.extract_text(image_input)` supporting Tesseract OCR.
    """
    class OCREngine:
        def __init__(self, lang: str = "eng"):
            self.lang = lang

        def extract_text(self, img_input: Any) -> str:
            import pytesseract
            from PIL import Image

            if isinstance(img_input, (str, os.PathLike)):
                with Image.open(img_input) as img:
                    return self._extract(img)
            elif isinstance(img_input, bytes):
                import io
                with Image.open(io.BytesIO(img_input)) as img:
                    return self._extract(img)
            else:
                return self._extract(img_input)

        def _extract(self, img: Any) -> str:
            import pytesseract
            ocr_img = img
            if getattr(img, "mode", None) in ("RGBA", "LA", "P"):
                ocr_img = img.convert("RGB")
            raw_result = pytesseract.image_to_string(ocr_img, lang=self.lang)
            return str(raw_result).strip() if raw_result else ""

    ocr_lang = os.getenv("OCR_LANG", "eng")
    return OCREngine(lang=ocr_lang)


_graph_instance = None

# Initialize the Graph connection
def get_graph_instance() -> Neo4jGraph:
    """Get or create a reusable Neo4j graph instance (connection pooling)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(
            url=NEO4J_URL, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD,
            enhanced_schema=True,
            refresh_schema=True
        )
    return _graph_instance

# print(get_graph_instance().schema)

def create_vector_indexes(driver, dimensions: Optional[int] = None, recreate_if_dimension_mismatch: bool = True) -> None:
    """Creates vector schema indexes for DocumentChunk nodes if they do not exist.
    If an existing index has mismatched dimensions, optionally drops and recreates it.
    """
    if dimensions is None:
        dimensions = get_embedding_dimension()

    indexes = [
        ("DocumentChunk_index", "DocumentChunk", "dc"),
    ]
    for index_name, label, var in indexes:
        if recreate_if_dimension_mismatch:
            try:
                check_query = f"SHOW VECTOR INDEXES YIELD name, options WHERE name = '{index_name}'"
                existing = driver.query(check_query)
                if existing:
                    opts = existing[0].get("options", {})
                    idx_config = opts.get("indexConfig", {}) if isinstance(opts, dict) else {}
                    existing_dim = idx_config.get("vector.dimensions")
                    if existing_dim is not None and int(existing_dim) != dimensions:
                        logger.warning(
                            f"Vector index {index_name} has dimension {existing_dim}, "
                            f"recreating with new dimension {dimensions}"
                        )
                        driver.query(f"DROP INDEX {index_name} IF EXISTS")
            except Exception as e:
                logger.warning(f"Could not verify existing dimension for index {index_name}: {e}")

        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR ({var}:{label})
        ON ({var}.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            driver.query(cypher)
        except Exception as e:
            logger.warning(f"Could not create vector index {index_name}: {e}")

def create_fulltext_indexes(driver) -> None:
    """Creates fulltext schema indexes for Document and DocumentChunk nodes if they do not exist."""
    indexes = [
        ("DocumentChunk_keyword_index", "DocumentChunk", ["content", "source"]),
        ("Document_keyword_index", "Document", ["filename", "description"]),
    ]
    for index_name, label, props in indexes:
        props_str = ", ".join(f"n.{prop}" for prop in props)
        cypher = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON EACH [{props_str}]
        """
        try:
            driver.query(cypher)
        except Exception as e:
            logger.warning(f"Could not create fulltext index {index_name}: {e}")

# Alias for singular call convention
create_fulltext_index = create_fulltext_indexes


def create_text_indexes(driver) -> None:
    """Creates text schema indexes for Document, DocumentChunk, and AppUser nodes if they do not exist."""
    indexes = [
        ("AppUser_id_text_index", "AppUser", "id"),
        ("DocumentChunk_source_text_index", "DocumentChunk", "source"),
        ("Document_source_text_index", "Document", "source"),
        ("Document_filename_text_index", "Document", "filename"),
        ("Document_file_type_text_index", "Document", "file_type"),
        ("Document_description_text_index", "Document", "description"),
    ]
    for index_name, label, prop in indexes:
        cypher = f"""
        CREATE TEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{prop})
        """
        try:
            driver.query(cypher)
        except Exception as e:
            logger.warning(f"Could not create text index {index_name}: {e}")

# Alias for singular call convention
create_text_index = create_text_indexes

def _embed_missing_batch_worker(
    batch_idx: int, 
    batch: List[Dict[str, Any]], 
    embedder: Any
) -> Tuple[int, List[Dict[str, Any]]]:
    """Worker task to embed a batch of unembedded nodes."""
    texts = [r["text"] for r in batch]
    embeddings = embedder.embed_documents(texts)
    updates = [
        {"elem_id": rec["elem_id"], "embedding": emb}
        for rec, emb in zip(batch, embeddings)
    ]
    return batch_idx, updates


def embed_missing_nodes(
    driver=None,
    batch_size: int = 32,
    label: str = "DocumentChunk",
    text_property: str = "content",
    embedding_property: str = "embedding",
    expected_dimension: Optional[int] = None,
    max_workers: int = 4,
) -> int:
    """Embeds nodes that do not have embeddings or have mismatched embedding dimensions using multithreading.

    Args:
        driver: Neo4jGraph connection instance (defaults to get_graph_instance())
        batch_size: Number of nodes to embed per batch
        label: Node label (default 'DocumentChunk')
        text_property: Property containing text to embed (default 'content')
        embedding_property: Property to store embedding vector (default 'embedding')
        expected_dimension: Desired embedding dimension (auto-detected if None)
        max_workers: Maximum worker threads for concurrent embedding calls

    Returns:
        int: Total number of nodes updated with new embeddings.
    """
    if driver is None:
        driver = get_graph_instance()

    if expected_dimension is None:
        expected_dimension = get_embedding_dimension()

    embedder = embedding_model()

    query_unembedded = f"""
    MATCH (n:`{label}`)
    WHERE (n.`{embedding_property}` IS NULL OR size(n.`{embedding_property}`) <> $expected_dim)
      AND n.`{text_property}` IS NOT NULL AND n.`{text_property}` <> ''
    RETURN elementId(n) AS elem_id, n.id AS id, n.`{text_property}` AS text
    """
    try:
        records = driver.query(query_unembedded, params={"expected_dim": expected_dimension})
    except Exception as e:
        logger.error(f"Error querying nodes missing embeddings for {label}: {e}")
        return 0

    if not records:
        logger.info(f"All '{label}' nodes already have valid {expected_dimension}-dim embeddings.")
        return 0

    total_count = len(records)
    logger.info(f"Found {total_count} '{label}' nodes requiring embeddings (expected dim: {expected_dimension}).")

    batches = [
        (idx, records[i : i + batch_size])
        for idx, i in enumerate(range(0, total_count, batch_size))
    ]
    num_batches = len(batches)
    effective_workers = max(1, min(max_workers, num_batches))

    logger.info(
        "Embedding %d nodes across %d batches using %d worker threads...",
        total_count,
        num_batches,
        effective_workers,
    )

    batch_updates: List[Tuple[int, List[Dict[str, Any]]]] = []

    if effective_workers <= 1 or num_batches <= 1:
        for b_idx, b_records in batches:
            try:
                _, updates = _embed_missing_batch_worker(b_idx, b_records, embedder)
                batch_updates.append((b_idx, updates))
            except Exception as e:
                logger.error(f"Failed to compute embeddings for batch {b_idx + 1}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {
                executor.submit(_embed_missing_batch_worker, b_idx, b_records, embedder): b_idx
                for b_idx, b_records in batches
            }
            for future in as_completed(future_to_idx):
                try:
                    b_idx, updates = future.result()
                    batch_updates.append((b_idx, updates))
                except Exception as e:
                    logger.error(f"Worker failed for batch: {e}")

    # Sort batches by index
    batch_updates.sort(key=lambda x: x[0])

    updated_count = 0
    update_cypher = f"""
    UNWIND $updates AS item
    MATCH (n:`{label}`)
    WHERE elementId(n) = item.elem_id
    SET n.`{embedding_property}` = item.embedding
    """

    for _, updates in batch_updates:
        if not updates:
            continue
        try:
            driver.query(update_cypher, params={"updates": updates})
            updated_count += len(updates)
            logger.info(f"Embedded {updated_count}/{total_count} '{label}' nodes...")
        except Exception as e:
            logger.error(f"Failed to write embeddings to Neo4j for batch: {e}")

    logger.info(f"Completed embedding update for {updated_count}/{total_count} '{label}' nodes.")
    return updated_count


def create_constraints(driver) -> None:
    """Creates minimum necessary constraints for data integrity and traversal optimization."""
    driver.query(
        "CREATE CONSTRAINT appuser_id IF NOT EXISTS FOR (u:AppUser) REQUIRE (u.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE (s.id) IS UNIQUE"
    )
    # Unstructured document ingestion constraints
    driver.query(
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE (d.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT document_chunk_id IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE (c.id) IS UNIQUE"
    )
    create_vector_indexes(driver)
    create_fulltext_indexes(driver)
    create_text_indexes(driver)
    embed_missing_nodes(driver)
