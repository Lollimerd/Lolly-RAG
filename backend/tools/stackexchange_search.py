import logging
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_neo4j import GraphCypherQAChain

from setup.init_config import (
    cypher_LLM, 
    embedding_model, 
    get_graph_instance, 
    reranker_model
)
from .template import CYPHER_GENERATION_TEMPLATE
from utils.utils import (
    format_docs_with_metadata,
    check_retrieval_hard_stop,
    increment_tool_call_count
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised singletons
# ---------------------------------------------------------------------------
_chain: GraphCypherQAChain | None = None
_compressor: CrossEncoderReranker | None = None


def _get_compressor() -> CrossEncoderReranker:
    """Build (or return cached) CrossEncoderReranker."""
    global _compressor
    if _compressor is None:
        _compressor = CrossEncoderReranker(
            model=reranker_model(),
            top_n=10,
        )
        logger.info("CrossEncoderReranker initialised (top_n=10).")
    return _compressor

def _get_chain() -> GraphCypherQAChain:
    """Build (or return cached) GraphCypherQAChain configured for retrieval only."""
    global _chain
    if _chain is None:
        graph = get_graph_instance()
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        _chain = GraphCypherQAChain.from_llm(
            llm=cypher_LLM(),                    # LLM for Cypher generation only
            graph=graph,
            cypher_prompt=cypher_prompt,         # inject domain-specific prompt
            validate_cypher=True,                # reject syntactically invalid queries
            return_intermediate_steps=True,      # exposes generated_cypher + context
            allow_dangerous_requests=True,       # required by langchain-neo4j ≥ 0.3
            verbose=False,                       # enable when debugging
            top_k=50,
        )
        logger.info("GraphCypherQAChain initialised (retrieval-only mode).")
    return _chain


# ---------------------------------------------------------------------------
# Internal helpers wrapped as named Runnables for SSE status visibility
# ---------------------------------------------------------------------------

def _run_graph_traversal(question: str) -> list[dict[str, Any]]:
    """
    Generate a Cypher query from *question* and execute it against Neo4j.
    Uses vector index search via $question_embedding for fast, indexed lookups.
    Returns the raw list of records returned by Neo4j.
    """
    chain = _get_chain()

    # Step 1: Embed the question — needed by the vector index in generated Cypher.
    question_embedding = embedding_model().embed_query(question)

    # Step 2: Generate Cypher using the cypher generation sub-chain only.
    # Passing question_embedding in the args allows the prompt template to
    # reference it; more importantly we pass it as a Bolt param in step 3.
    args = {
        "question": question,
        "schema": chain.graph_schema,
        "query": question,
    }
    from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
    raw_cypher = chain.cypher_generation_chain.invoke(args)
    generated_cypher = extract_cypher(raw_cypher)

    if chain.cypher_query_corrector:
        generated_cypher = chain.cypher_query_corrector(generated_cypher)

    logger.info("Generated Cypher: %s", generated_cypher)

    # Step 3: Execute Cypher against Neo4j with $question_embedding as a
    # Bolt parameter so CALL db.index.vector.queryNodes(...) can resolve it.
    raw_context: list[dict[str, Any]] = []
    if generated_cypher:
        try:
            raw_context = chain.graph.query(
                generated_cypher,
                params={"question_embedding": question_embedding},
            )
        except Exception as exc:
            logger.error("Failed to execute generated Cypher: %s", exc)
            raise exc

    logger.info("Raw context records: %d rows", len(raw_context))
    return raw_context


# Wrap as a named RunnableLambda so backend.py can detect "GraphTraversal" events.
GraphTraversal = RunnableLambda(_run_graph_traversal).with_config(
    {"run_name": "GraphTraversal"}
)


def _rerank_docs(inputs: Dict[str, Any]) -> List[Document]:
    """
    Cross-encoder reranking step.

    Expects ``inputs`` to be a dict with:
      • ``"docs"``     — list[dict] raw records from Neo4j
      • ``"question"`` — the original user question string

    Each raw dict is converted to a Document whose page_content is a
    human-readable title+body string and whose metadata holds the full
    record.  CrossEncoderReranker then scores every Document against the
    question and returns the top-N most relevant ones.
    """
    raw_records: List[Dict[str, Any]] = inputs.get("docs", [])
    question: str = inputs.get("question", "")

    if not raw_records:
        logger.warning("Reranking received 0 documents — skipping.")
        return []

    # --- Convert raw dicts → Documents ---
    # Truncate heavy fields *before* string concatenation to avoid
    # allocating multi-MB page_content strings for documents the reranker will never fully read (cross-encoder max is ~512 tokens ≈ 2 000 chars)
    _MAX_BODY = 1500   # chars per field fed to the cross-encoder
    _MAX_META = 3500   # chars for metadata strings passed to the answer LLM
    docs: List[Document] = []
    for record in raw_records:
        title   = (record.get("question_title") or "")[:200]
        q_body  = (record.get("question_body") or "")[:_MAX_BODY]
        a_body  = (record.get("answer_body") or record.get("best_answer_body") or "")[:_MAX_BODY]
        page_content = f"Title: {title}\nQuestion: {q_body}\nAnswer: {a_body}"

        # Truncate remaining metadata string fields for the answer LLM context
        meta: Dict[str, Any] = {
            k: (v[:_MAX_META] if isinstance(v, str) and len(v) > _MAX_META else v)
            for k, v in record.items()
        }
        docs.append(Document(page_content=page_content, metadata=meta))

    logger.info("Reranking %d documents...", len(docs))
    compressor = _get_compressor()

    try:
        reranked = compressor.compress_documents(documents=docs, query=question)
    except Exception as exc:
        logger.error("CrossEncoderReranker failed: %s — returning raw docs", exc)
        reranked = docs  # graceful fallback: return unranked docs

    final_docs = list(reranked)
    logger.info("✅ %d docs passed reranking.", len(final_docs))
    return final_docs


# Named Runnable so backend.py can detect "Reranking" events.
Reranking = RunnableLambda(_rerank_docs).with_config({"run_name": "Reranking"})


# ---------------------------------------------------------------------------
# Public LangChain tool
# ---------------------------------------------------------------------------

@tool
def graph_rag_tool(question: str) -> str:
    """
    Search the StackExchange / StackOverflow developer knowledge graph containing programming questions, answers, tags, and accepted code solutions.

    Use this tool whenever:
    - The user asks general programming language, library, syntax, API, framework, or software debugging questions.
    - The question is about developer community discussions and code examples.

    Args:
        question: The user's programming question or technical topic to look up in the graph.

    Returns:
        A formatted string containing the raw records retrieved from Neo4j.
    """
    logger.info("graph_rag_tool invoked: %r", question[:120])

    should_stop, stop_reason = check_retrieval_hard_stop()
    if should_stop:
        logger.warning("graph_rag_tool hard stopped: %s", stop_reason)
        return stop_reason

    increment_tool_call_count()

    try:
        # Step 1 — Cypher generation + Neo4j execution
        raw_records: List[Dict[str, Any]] = GraphTraversal.invoke(question)

        # Step 2 — Cross-encoder reranking
        reranked_docs: List[Document] = Reranking.invoke(
            {"docs": raw_records, "question": question}
        )

        if not reranked_docs:
            return (
                "No relevant data found in the knowledge graph for this question."
            )

        # Step 3 — Format reranked Documents into a context string
        return format_docs_with_metadata(reranked_docs)

    except Exception as exc:
        logger.error("graph_rag_tool error: %s", exc, exc_info=True)
        return f"Graph retrieval failed: {exc}."
