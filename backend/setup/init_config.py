"""Setting up ollama models, vectorstores and Neo4j Configs"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from typing import Dict, List
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ===========================================================================================================================================================
# Step 1: Load Configuration: Docker, Neo4j, Ollama, Langchain
# ===========================================================================================================================================================

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


# qwen3:8b works for now with limited context of 40k, qwen3:30b works with 256k max
def answer_LLM():
    """main model for RAG agent"""
    return ChatOllama(
        model="qwen3.5:4b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=40968,
        num_predict=8192,  # max tokens in answer
        temperature=0.7,  # balanced creativity
        repeat_penalty=1.1,  # standard mild penalty (1.5 caused severe gibberish)
        repeat_last_n=64,  # look back 64 tokens (-1 penalized entire 40k context)
        top_p=0.9,  # nucleus sampling
        top_k=40,  # standard candidate pool
        reasoning=True,
        tags=["answer_llm"],
    )


# embedding model — singleton to avoid reloading on every call
# snowflake artic embed2
@lru_cache(maxsize=1)
def embedding_model():
    """embedding model"""
    return OllamaEmbeddings(
        model="jina/jina-embeddings-v2-base-en:latest",
        base_url=OLLAMA_BASE_URL,
        num_ctx=8192,  # 8k context
    )


# reranker model — singleton: ONNX + TensorRT compilation happens once
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


# save llama3.1:8b for now
def summarizer():
    """summarizes historical context"""
    return ChatOllama(
        model="qwen3.5:0.8b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=8192,  # 40k context
        tags=["summarizer_llm"],
    )


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

import logging

logger = logging.getLogger(__name__)

def create_vector_indexes(driver, dimensions: int = 768) -> None:
    """Creates vector schema indexes for DocumentChunk nodes if they do not exist."""
    indexes = [
        ("DocumentChunk_index", "DocumentChunk", "dc"),
    ]
    for index_name, label, var in indexes:
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
