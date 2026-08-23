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


def cypher_LLM():
    """
    Dedicated LLM for Cypher query generation inside GraphCypherQAChain.
    Reasoning is explicitly disabled so that <think> tokens are never emitted
    during the tool's internal LLM calls — preventing them from leaking into
    the agent's streaming event bus and appearing as spurious thought output.
    Lower temperature + greedy top_k/top_p gives deterministic, schema-faithful queries.
    """
    return ChatOllama(
        model="qwen3.5:4b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=40960, # system prompt + user query + schema + retrieved docs
        num_predict=1024,  # Cypher queries are concise
        temperature=0.0,  # fully deterministic — critical for valid Cypher
        top_p=1.0,
        top_k=1,
        reasoning=False,  # MUST be False — no <think> tokens in tool steps
        tags=["cypher_llm"],
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
    """Creates vector schema indexes for Question, Answer, Tag, and User nodes if they do not exist."""
    indexes = [
        ("Question_index", "Question", "q"),
        ("Answer_index", "Answer", "a"),
        ("Tag_index", "Tag", "t"),
        ("User_index", "User", "u"),
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
    """Creates fulltext schema indexes for Question, Answer, Tag, and User nodes if they do not exist."""
    indexes = [
        ("Question_keyword_index", "Question", ["title", "body"]),
        ("Answer_keyword_index", "Answer", ["body"]),
        ("Tag_keyword_index", "Tag", ["name"]),
        ("User_keyword_index", "User", ["display_name"]),
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


def create_constraints(driver) -> None:
    """Creates minimum necessary constraints for data integrity and traversal optimization."""
    driver.query(
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT importlog_id IF NOT EXISTS FOR (i:ImportLog) REQUIRE (i.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE (s.id) IS UNIQUE"
    )
    create_vector_indexes(driver)
    create_fulltext_indexes(driver)



# print(f"\nschema: {graph.schema}\n")


# test embedding generation
# sample_embedding = EMBEDDINGS.embed_query("Hello world")
# print(f"\nSample embedding dimension: {len(sample_embedding)}")

# ===========================================================================================================================================================
# Creation of vector index, vectorstores and fulltext for hybrid vector search
# ===========================================================================================================================================================


def create_vector_stores(graph, EMBEDDINGS, retrieval_query) -> Dict[str, Neo4jVector]:
    """
    Creates Neo4jVector stores from an existing graph using a data-driven approach.

    Args:
        graph: The Neo4j graph instance.
        EMBEDDINGS: The embedding model.
        retrieval_query: The Cypher query for retrieval.

    Returns:
        A dictionary of Neo4jVector store instances, keyed by their node label.
    """

    # Define a list of configurations for each vector store
    store_configs = [
        {
            "node_label": "Tag",
            "text_node_properties": ["name"],
        },
        {
            "node_label": "User",
            "text_node_properties": ["reputation", "display_name"],
        },
        {
            "node_label": "Question",
            "text_node_properties": [
                "score",
                "link",
                "favourite_count",
                "id",
                "creation_date",
                "body",
                "title",
            ],
        },
        {
            "node_label": "Answer",
            "text_node_properties": [
                "score",
                "is_accepted",
                "id",
                "body",
                "creation_date",
            ],
        },
    ]

    vectorstores = {}

    # Loop through the configurations and create the vectorstores & vector indexes
    for config in store_configs:
        label: str = str(config["node_label"])
        text_props: List[str] = config["text_node_properties"]  # type: ignore[assignment]
        index_name = f"{label}_index"
        keyword_index_name = f"{label}_keyword_index"

        vectorstores[label.lower() + "store"] = Neo4jVector.from_existing_graph(
            graph=graph,
            node_label=label,
            embedding=EMBEDDINGS,
            embedding_node_property="embedding",
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=SearchType.HYBRID,
            text_node_properties=text_props,
            retrieval_query=retrieval_query,
        )
        print(f"Created vectorstore for {index_name} index")

    return vectorstores
