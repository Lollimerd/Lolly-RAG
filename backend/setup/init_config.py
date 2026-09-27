"""Ollama models, vectorstore, and Neo4j configuration setup."""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
from PIL import Image

logger = logging.getLogger(__name__)

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

def answer_LLM():
    """Args: None."""
    return ChatOllama(
        model="qwen3.5:4b", base_url=OLLAMA_BASE_URL,
        num_ctx=40968, num_predict=4096, temperature=0.7, repeat_penalty=1.1,
        repeat_last_n=64, top_p=0.95, top_k=40, reasoning=True, tags=["answer_llm"],
    )

@lru_cache(maxsize=1)
def embedding_model():
    """Args: None."""
    return OllamaEmbeddings(model="qwen3-embedding:0.6b", base_url=OLLAMA_BASE_URL, num_ctx=16384)

@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Args: None."""
    try:
        dim = len(embedding_model().embed_query("dimension_probe"))
        logger.info("Detected embedding dimension: %d", dim)
        return dim
    except Exception as e:
        logger.warning("Could not probe embedding dimension, defaulting to 1024: %s", e)
        return 1024

@lru_cache(maxsize=1)
def reranker_model() -> HuggingFaceCrossEncoder:
    """Args: None."""
    import torch
    return HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

def summarizer():
    """Args: None."""
    return ChatOllama(model="qwen3.5:0.8b", base_url=OLLAMA_BASE_URL, num_ctx=8192, tags=["summarizer_llm"])

class NemotronOCRWrapper:
    """Lazy wrapper for NVIDIA Nemotron OCR v2."""
    name: str = "nemotron-ocr-v2"

    def __init__(self, lang="en", merge_level="paragraph", model_dir=None, detector_only=False, skip_relational=False):
        """Args: lang: Language, merge_level: Merge level, model_dir: Model path, detector_only: Detector flag, skip_relational: Relational flag."""
        self.lang, self.merge_level = lang, merge_level
        self.model_dir, self.detector_only, self.skip_relational = model_dir, detector_only, skip_relational
        self._pipeline = None

    @property
    def pipeline(self) -> Any:
        """Args: None."""
        if self._pipeline is None:
            logger.info("Initializing NemotronOCRV2 (lang=%s, merge_level=%s)...", self.lang, self.merge_level)
            kwargs: Dict[str, Any] = {"model_dir": self.model_dir} if self.model_dir else {"lang": self.lang}
            if self.detector_only: kwargs["detector_only"] = True
            if self.skip_relational: kwargs["skip_relational"] = True
            self._pipeline = NemotronOCRV2(**kwargs)
            logger.info("NemotronOCRV2 initialized.")
        return self._pipeline

    def _normalize_image_input(self, img: Any) -> Any:
        """Args: img: Image input."""
        if isinstance(img, (str, os.PathLike, io.BytesIO)): return img
        if isinstance(img, (bytes, bytearray)): return io.BytesIO(img)
        if isinstance(img, Image.Image):
            import numpy as np, torch
            return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)
        return img

    def extract_text(self, img: Any, merge_level: Optional[str] = None) -> str:
        """Args: img: Image input, merge_level: Merge level."""
        preds = self.pipeline(self._normalize_image_input(img), merge_level=merge_level or self.merge_level)
        if not preds: return ""
        items = [preds] if isinstance(preds, dict) else preds
        return "\n\n".join(p["text"] for p in items if isinstance(p, dict) and p.get("text", "").strip())

    def invoke(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """Args: input_data: Input data, config: Optional configuration."""
        return self.extract_text(input_data)

@lru_cache(maxsize=1)
def ocr_model() -> NemotronOCRWrapper:
    """Args: None."""
    return NemotronOCRWrapper(
        lang=os.getenv("NEMOTRON_OCR_LANG", "en"),
        merge_level=os.getenv("NEMOTRON_OCR_MERGE_LEVEL", "paragraph"),
        model_dir=os.getenv("NEMOTRON_OCR_MODEL_DIR"),
    )

_graph_instance: Optional[Neo4jGraph] = None

def get_graph_instance() -> Neo4jGraph:
    """Args: None."""
    global _graph_instance
    if _graph_instance is None:
        try:
            _graph_instance = Neo4jGraph(
                url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                enhanced_schema=False, refresh_schema=False,
            )
        except Exception as e:
            logger.error("Failed to connect to Neo4j at %s: %s", NEO4J_URL, e)
            raise
    return _graph_instance

def create_vector_indexes(driver, dimensions: Optional[int] = None, recreate_if_dimension_mismatch: bool = True) -> None:
    """Args: driver: Neo4j driver, dimensions: Embedding size, recreate_if_dimension_mismatch: Flag."""
    dimensions = dimensions or get_embedding_dimension()
    for index_name, label, var in [("DocumentChunk_index", "DocumentChunk", "dc")]:
        if recreate_if_dimension_mismatch:
            try:
                existing = driver.query(f"SHOW VECTOR INDEXES YIELD name, options WHERE name = '{index_name}'")
                if existing:
                    cfg = (existing[0].get("options") or {}).get("indexConfig", {})
                    if cfg.get("vector.dimensions") and int(cfg["vector.dimensions"]) != dimensions:
                        logger.warning("Recreating index %s (dim mismatch: %s -> %d)", index_name, cfg["vector.dimensions"], dimensions)
                        driver.query(f"DROP INDEX {index_name} IF EXISTS")
            except Exception as e:
                logger.warning("Could not verify index dimension for %s: %s", index_name, e)
        try:
            driver.query(f"""CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR ({var}:{label}) ON ({var}.embedding)
            OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine' }} }}""")
        except Exception as e:
            logger.warning("Could not create vector index %s: %s", index_name, e)

def create_fulltext_indexes(driver) -> None:
    """Args: driver: Neo4j driver."""
    for name, label, props in [
        ("DocumentChunk_keyword_index", "DocumentChunk", ["content", "source"]),
        ("Document_keyword_index", "Document", ["filename", "description"]),
    ]:
        props_str = ", ".join(f"n.{p}" for p in props)
        try: driver.query(f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON EACH [{props_str}]")
        except Exception as e: logger.warning("Could not create fulltext index %s: %s", name, e)

create_fulltext_index = create_fulltext_indexes

def create_text_indexes(driver) -> None:
    """Args: driver: Neo4j driver."""
    for name, label, prop in [
        ("AppUser_id_text_index", "AppUser", "id"),
        ("DocumentChunk_source_text_index", "DocumentChunk", "source"),
        ("Document_source_text_index", "Document", "source"),
        ("Document_filename_text_index", "Document", "filename"),
        ("Document_file_type_text_index", "Document", "file_type"),
        ("Document_description_text_index", "Document", "description"),
    ]:
        try: driver.query(f"CREATE TEXT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
        except Exception as e: logger.warning("Could not create text index %s: %s", name, e)

create_text_index = create_text_indexes

def _embed_missing_batch_worker(batch_idx: int, batch: List[Dict[str, Any]], embedder: Any) -> Tuple[int, List[Dict[str, Any]]]:
    """Args: batch_idx: Batch index, batch: Record batch, embedder: Embedder."""
    embeddings = embedder.embed_documents([r["text"] for r in batch])
    return batch_idx, [{"elem_id": r["elem_id"], "embedding": emb} for r, emb in zip(batch, embeddings)]

def embed_missing_nodes(
    driver=None,
    batch_size: int = 32,
    label: str = "DocumentChunk",
    text_property: str = "content",
    embedding_property: str = "embedding",
    expected_dimension: Optional[int] = None,
    max_workers: int = 4,
    doc_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
) -> int:
    """Args:
        driver: Neo4jGraph instance.
        batch_size: Nodes per embedding call.
        label: Target node label.
        text_property: Property containing text.
        embedding_property: Property to store vector.
        expected_dimension: Target embedding dimension.
        max_workers: Concurrent thread pool size.
        doc_id: Optional document ID filter.
        progress_callback: Progress callback.
    """
    if driver is None: driver = get_graph_instance()
    if expected_dimension is None: expected_dimension = get_embedding_dimension()
    embedder = embedding_model()

    doc_filter = f"MATCH (d:Document {{id: $doc_id}})-[:HAS_CHUNK]->(n:`{label}`)" if doc_id else f"MATCH (n:`{label}`)"
    query = f"""{doc_filter}
    WHERE (n.`{embedding_property}` IS NULL OR size(n.`{embedding_property}`) <> $expected_dim)
      AND n.`{text_property}` IS NOT NULL AND n.`{text_property}` <> ''
    RETURN elementId(n) AS elem_id, n.id AS id, n.`{text_property}` AS text"""

    try: records = driver.query(query, params={"expected_dim": expected_dimension, **({"doc_id": doc_id} if doc_id else {})})
    except Exception as e: logger.error("Error querying unembedded nodes: %s", e); return 0

    if not records:
        if progress_callback: progress_callback(1, 1, 0, 0)
        return 0

    total_count = len(records)
    batches = [(idx, records[i: i + batch_size]) for idx, i in enumerate(range(0, total_count, batch_size))]
    num_batches = len(batches)
    effective_workers = max(1, min(max_workers, num_batches))
    update_cypher = f"UNWIND $updates AS item MATCH (n:`{label}`) WHERE elementId(n) = item.elem_id SET n.`{embedding_property}` = item.embedding"
    updated_count = completed_batches = 0

    def _process_batch(b_idx, updates):
        nonlocal updated_count, completed_batches
        if updates:
            driver.query(update_cypher, params={"updates": updates})
            updated_count += len(updates)
            completed_batches += 1
            if progress_callback: progress_callback(completed_batches, num_batches, updated_count, total_count)

    if effective_workers <= 1 or num_batches <= 1:
        for b_idx, b_records in batches:
            try:
                _, updates = _embed_missing_batch_worker(b_idx, b_records, embedder)
                _process_batch(b_idx, updates)
            except Exception as e: logger.error("Batch %d failed: %s", b_idx + 1, e)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            fmap = {executor.submit(_embed_missing_batch_worker, b_idx, b_records, embedder): b_idx for b_idx, b_records in batches}
            for future in as_completed(fmap):
                try:
                    b_idx, updates = future.result()
                    _process_batch(b_idx, updates)
                except Exception as e: logger.error("Worker failed: %s", e)

    return updated_count

def create_constraints(driver) -> None:
    """Args: driver: Neo4j driver."""
    for q in [
        "CREATE CONSTRAINT appuser_id IF NOT EXISTS FOR (u:AppUser) REQUIRE (u.id) IS UNIQUE",
        "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE (s.id) IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE (d.id) IS UNIQUE",
        "CREATE CONSTRAINT document_chunk_id IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE (c.id) IS UNIQUE",
    ]:
        driver.query(q)
    create_vector_indexes(driver)
    create_fulltext_indexes(driver)
    create_text_indexes(driver)
    embed_missing_nodes(driver)
