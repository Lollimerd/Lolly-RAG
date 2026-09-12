import asyncio
import gc
import json
import logging
import uuid
import uvicorn

from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware import Middleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from setup.init_config import (
    answer_LLM,
    embedding_model,
    get_graph_instance,
    NEO4J_URL,
    NEO4J_USERNAME,
    create_constraints,
)

from utils.doc_processor import (
    process_uploaded_file,
    list_documents,
    delete_document,
    get_document_chunks,
    update_document,
    SUPPORTED_EXTENSIONS,
)

from utils.dashboard import (
    get_database_summary,
    get_import_history,
    get_entity_counts,
    search_nodes,
    get_graph_sample,
)

from agents.agent import rag_agent

from utils.utils import (
    find_container_by_port,
    reset_tool_call_count,
)
from utils.memory import (
    add_ai_message_to_session,
    add_user_message_to_session,
    delete_session,
    delete_user,
    get_all_users,
    get_user_sessions,
    link_session_to_user,
    repair_missing_has_message_relationships,
    update_last_ai_message,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize constraints on startup manually handling graph driver setup
    try:
        graph = get_graph_instance()
        create_constraints(graph)
        logger.info("Database constraints verified/created successfully.")
        
        # Repair any sessions with missing HAS_MESSAGE relationships
        repaired = repair_missing_has_message_relationships()
        if repaired > 0:
            logger.info(f"✅ Checked {repaired} sessions for missing HAS_MESSAGE relationships on startup")
        else:
            logger.info("✅ All sessions have correct HAS_MESSAGE relationships")
    except Exception as e:
        logger.error(f"Failed to create database constraints: {e}")
    yield


# initialise fastapi
app = FastAPI(
    title="GraphRAG API", version="1.2.0", middleware=middleware, lifespan=lifespan
)

# Define routers
system_router = APIRouter(tags=["System"])
users_router = APIRouter(tags=["Users"])
chat_router = APIRouter(tags=["Chat"])
repair_router = APIRouter(tags=["Repair"])
ingest_router = APIRouter(prefix="/ingest", tags=["Ingestion"])
stats_router = APIRouter(prefix="/stats", tags=["Analytics"])
graph_router = APIRouter(prefix="/graph", tags=["Graph"])


class QueryRequest(BaseModel):
    """configure question template for answer LLM"""

    question: str
    session_id: str
    user_id: str = "test_user"  # fallback
    mode: str = "auto"  # 'auto' or 'custom'
    attached_files: Optional[List[str]] = None
    files: Optional[List[str]] = None
    images: Optional[List[Union[str, Dict[str, Any]]]] = None


@system_router.get("/")
def index():
    return {"status": "online", "message": "Welcome to the GraphRAG API"}


@system_router.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- Add config endpoint ---
@system_router.get("/config")
def get_configuration():
    """Provides frontend with configuration details for display."""
    try:
        parsed_url = urlparse(NEO4J_URL)
        neo4j_port = parsed_url.port or 7687
        neo4j_host = parsed_url.hostname
        discovered_name = find_container_by_port(neo4j_port)

        if (
            "not mounted" in discovered_name
            or "Error" in discovered_name
            or "Invalid" in discovered_name
        ):
            container_name = f"{neo4j_host} (Configured Host)"
        else:
            container_name = discovered_name

        return {
            "ollama_model": answer_LLM().model,
            "neo4j_url": NEO4J_URL,
            "container_name": container_name,
            "neo4j_user": NEO4J_USERNAME,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in get_configuration: {e}")
        return {
            "status": "error",
            "message": str(e),
            "ollama_model": "unknown",
            "neo4j_url": NEO4J_URL,
            "container_name": "unknown",
        }


@users_router.get("/users")
def get_users():
    """Returns a list of all application users."""
    try:
        users = get_all_users()
        return {"users": users, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return {"users": [], "status": "error", "message": str(e)}


@users_router.get("/user/{user_id}/chats")
def get_user_chats(user_id: str):
    """Returns a list of chat sessions for a specific user."""
    try:
        sessions = get_user_sessions(user_id)
        return {"chats": sessions, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching chats for user {user_id}: {e}")
        return {"chats": [], "status": "error", "message": str(e)}


@chat_router.get("/chat/{session_id}")
def get_chat_messages(session_id: str):
    """Returns the message history for a specific session, including thoughts for AI messages."""
    try:
        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_MESSAGE]->(m:Message)
        RETURN m.type AS role, m.content AS content, m.thought AS thought
        ORDER BY coalesce(m.created_at, elementId(m)) ASC
        LIMIT 1000
        """
        results = get_graph_instance().query(query, params={"session_id": session_id})

        if not results:
            results = []

        formatted_results: List[Dict[str, Any]] = []
        for res in results:
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except Exception:
                    res = {"role": "user", "content": res}
            if isinstance(res, dict):
                item: Dict[str, Any] = dict(res)
                if item.get("role") == "ai":
                    item["role"] = "assistant"
                if "thought" not in item:
                    item["thought"] = None
                formatted_results.append(item)

        return {"messages": formatted_results, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching chat history for {session_id}: {e}")
        return {"messages": [], "status": "error", "message": str(e)}


@chat_router.delete("/chat/{session_id}")
def delete_user_session(session_id: str):
    """Deletes a specific chat session."""
    try:
        delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return {"status": "error", "message": str(e)}


@repair_router.post("/repair-sessions")
def repair_sessions():
    """
    Repairs sessions with missing HAS_MESSAGE relationships.
    This fixes the issue where messages exist but are invisible to queries.
    """
    try:
        repaired = repair_missing_has_message_relationships()
        return {
            "status": "success",
            "message": f"Checked/repaired {repaired} sessions with missing HAS_MESSAGE relationships",
            "repaired_count": repaired,
        }
    except Exception as e:
        logger.error(f"Error repairing sessions: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to repair sessions: {str(e)}"
        )

@users_router.delete("/user/{user_id}")
def delete_app_user(user_id: str):
    """Deletes a user and all their data."""
    try:
        delete_user(user_id)
        return {"status": "success", "message": f"User {user_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return {"status": "error", "message": str(e)}

# ===========================================================================================================================================================
# Unstructured Document Ingestion Endpoints
# ===========================================================================================================================================================

class DocumentUploadResponse(BaseModel):
    status: str
    doc_id: str = ""
    filename: str = ""
    chunk_count: int = 0
    message: str = ""


@ingest_router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    description: str = Form(default=""),
    force: bool = Form(default=False),
    engine: str = Form(default="pandas"),
):
    """
    Upload a single document (PDF, DOCX, TXT, MD, CSV, XLSX, XLS) for ingestion into Neo4j.

    The file is chunked, embedded with the configured Ollama embedding model,
    and stored as (Document)-[:HAS_CHUNK]->(DocumentChunk) nodes in Neo4j.
    The resulting chunks are immediately queryable via document_search_tool.
    Supports engine='pandas' (structured RAG chunks) and engine='apoc' (database-side batch load).
    Duplicate files are skipped automatically unless force=True.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        file_bytes = await file.read()

        result = await asyncio.to_thread(
            process_uploaded_file,
            file_bytes,
            file.filename,
            user_id,
            description,
            get_graph_instance(),
            embedding_model(),
            force=force,
            engine=engine,
        )

        return DocumentUploadResponse(
            status=result.get("status", "success"),
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=result.get("message")
            or f"Document '{file.filename}' ingested successfully with {result['chunk_count']} chunks.",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting document '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@ingest_router.post("/apoc/csv", response_model=DocumentUploadResponse)
async def upload_csv_apoc(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    description: str = Form(default=""),
    force: bool = Form(default=False),
):
    """
    Dedicated endpoint for CSV ingestion directly using APOC batch procedures into Neo4j.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported for the APOC ingestion endpoint.")

    try:
        file_bytes = await file.read()

        result = await asyncio.to_thread(
            process_uploaded_file,
            file_bytes,
            file.filename,
            user_id,
            description,
            get_graph_instance(),
            embedding_model(),
            force=force,
            engine="apoc",
        )

        return DocumentUploadResponse(
            status=result.get("status", "success"),
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=result.get("message")
            or f"CSV '{file.filename}' ingested successfully via APOC with {result['chunk_count']} records.",
        )
    except Exception as e:
        logger.error(f"Error ingesting CSV via APOC '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"APOC Ingestion failed: {str(e)}")


@ingest_router.get("/documents")
async def get_documents():
    """
    List all uploaded documents stored in Neo4j.
    Returns metadata only (no embeddings or chunk content).
    """
    try:
        docs = await asyncio.to_thread(list_documents, get_graph_instance())
        return {"status": "success", "documents": docs, "count": len(docs)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"status": "error", "message": str(e), "documents": []}


@ingest_router.get("/documents/{doc_id}/chunks")
async def get_doc_chunks(doc_id: str):
    """
    List all chunks for a specific document.
    """
    try:
        chunks = await asyncio.to_thread(get_document_chunks, doc_id, get_graph_instance())
        return {"status": "success", "chunks": chunks, "count": len(chunks)}
    except Exception as e:
        logger.error(f"Error getting chunks for document {doc_id}: {e}")
        return {"status": "error", "message": str(e), "chunks": []}


class DocumentUpdateRequest(BaseModel):
    description: str = ""


@ingest_router.put("/documents/{doc_id}")
async def update_doc_metadata(doc_id: str, req: DocumentUpdateRequest):
    """
    Update a document's description/folder metadata.
    """
    try:
        updated = await asyncio.to_thread(update_document, doc_id, req.description, get_graph_instance())
        if not updated:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"status": "success", "message": f"Document '{doc_id}' metadata updated."}
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@ingest_router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """
    Delete a document and all its associated chunks from Neo4j.
    """
    try:
        deleted = await asyncio.to_thread(delete_document, doc_id, get_graph_instance())
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"status": "success", "message": f"Document '{doc_id}' and all chunks deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# ===========================================================================================================================================================
# Streaming Helpers
# ===========================================================================================================================================================

def _sse(data: dict) -> str:
    """Wrap a dict as a Server-Sent Event frame."""
    return f"data: {json.dumps(data)}\n\n"


def _parse_chunk(chunk) -> tuple[str, str]:
    """Extract (content, reasoning) text from a LangChain streaming chunk."""
    content = ""
    reasoning = ""

    raw = getattr(chunk, "content", "")
    if isinstance(raw, str):
        content = raw
    elif isinstance(raw, list):
        for part in raw:
            if isinstance(part, str):
                content += part
            elif isinstance(part, dict):
                p_type = part.get("type", "")
                if p_type == "text":
                    content += part.get("text", "")
                elif p_type in ("reasoning", "thinking"):
                    reasoning += part.get("reasoning") or part.get("thinking") or part.get("text", "")
    else:
        content = str(raw) if raw else ""

    # Additional reasoning from kwargs (e.g. Claude extended thinking)
    if hasattr(chunk, "additional_kwargs") and isinstance(chunk.additional_kwargs, dict):
        extra = chunk.additional_kwargs.get("reasoning_content") or chunk.additional_kwargs.get("thinking") or ""
        reasoning += extra

    # Strip inline <think>...</think> tags streamed inside content
    if "<think>" in content:
        prefix, rest = content.split("<think>", 1)
        content = prefix
        if "</think>" in rest:
            r_part, c_part = rest.split("</think>", 1)
            reasoning += r_part
            content += c_part
        else:
            reasoning += rest
    elif "</think>" in content:
        r_part, c_part = content.split("</think>", 1)
        reasoning += r_part
        content = c_part

    return content, reasoning


# Tool status message lookups
_TOOL_START_MSG: dict[str, str] = {
    "document_search_tool": "📄 Searching uploaded documents...",
}
_TOOL_END_MSG: dict[str, str] = {
    "document_search_tool": "✅ Document search completed",
}


# ===========================================================================================================================================================
# Streaming Logic
# ===========================================================================================================================================================
@chat_router.post("/agent/ask")
async def agent_ask(request: QueryRequest) -> StreamingResponse:
    """Endpoint to query the new LangChain Agent with SSE streaming."""

    async def agent_stream_generator() -> AsyncGenerator[str, None]:
        logger.info(
            f"Agent request: '{request.question[:50]}...' from user {request.user_id}"
        )

        # Reset tool call counter for this request
        reset_tool_call_count(request.session_id)
        response_chunks = []
        response_thought_chunks = []

        try:
            # 1. Prepare Input
            # Extract any attached files / images from payload
            attached = []
            if request.attached_files:
                attached.extend(request.attached_files)
            if request.files:
                attached.extend([f for f in request.files if f not in attached])
            if request.images:
                for img_item in request.images:
                    if isinstance(img_item, str) and img_item not in attached:
                        attached.append(img_item)
                    elif isinstance(img_item, dict) and img_item.get("filename") and img_item["filename"] not in attached:
                        attached.append(img_item["filename"])

            # Formulate effective question with attached file context if not already mentioned
            effective_question = request.question
            if attached:
                files_header = f"[Attached File(s): {', '.join(attached)}]"
                if not any(f in request.question for f in attached) and files_header not in request.question:
                    effective_question = f"{files_header}\n{request.question}"

            # Retrieve history
            history_response = await asyncio.to_thread(
                get_chat_messages, request.session_id
            )
            messages = []
            if history_response.get("status") == "success":
                for msg in history_response.get("messages", []):
                    if isinstance(msg, str):
                        try:
                            msg = json.loads(msg)
                        except Exception:
                            msg = {"role": "user", "content": msg}
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role in ("assistant", "ai"):
                            messages.append(AIMessage(content=content))

            # Construct input for Graph Agent (expects 'messages' key in state)
            input_messages = messages + [HumanMessage(content=effective_question)]
            input_data = {
                "messages": input_messages,
                "question": effective_question,
                "session_id": request.session_id,
            }

            # Save user message to DB
            try:
                # Ensure session is linked to user
                await asyncio.to_thread(
                    link_session_to_user,
                    request.session_id,
                    request.user_id,
                    request.question,
                )

                await asyncio.to_thread(
                    add_user_message_to_session, request.session_id, request.question
                )
            except Exception as e:
                logger.warning(f"Error saving user message: {e}")

            # 2. Stream Events from Agent Executor
            async for event in rag_agent.astream_events(
                input_data, version="v2"
            ):
                event_type = event["event"]
                event_name = event["name"]

                # --- A. Tool status ---
                if event_type == "on_tool_start":
                    msg = _TOOL_START_MSG.get(event_name, f"🛠️ Using tool: {event_name}...")
                    yield _sse({"type": "status", "stage": "tool_start", "status": "running", "message": msg})

                elif event_type == "on_tool_end":
                    msg = _TOOL_END_MSG.get(event_name, f"✅ Tool {event_name} completed")
                    yield _sse({"type": "status", "stage": "tool_end", "status": "complete", "message": msg})

                # --- B. Token streaming (final agent LLM only) ---
                elif event_type == "on_chat_model_stream":
                    if "answer_llm" not in event.get("tags", []):
                        continue
                    chunk = event["data"].get("chunk")
                    if chunk:
                        content, reasoning = _parse_chunk(chunk)
                        if content or reasoning:
                            yield _sse({"type": "token", "content": content, "reasoning_content": reasoning})
                            if content:
                                response_chunks.append(content)
                            if reasoning:
                                response_thought_chunks.append(reasoning)

        except Exception as e:
            logger.error(f"Error in agent stream: {e}")
            yield _sse({"type": "error", "content": str(e)})

        # Save AI response to DB
        try:
            full_response = "".join(response_chunks)
            full_thought = "".join(response_thought_chunks)
            if full_response:
                await asyncio.to_thread(
                    add_ai_message_to_session,
                    request.session_id,
                    full_response,
                    full_thought,
                )
                logger.info(f"Response saved to DB: {len(full_response)} chars")
        except Exception as e:
            logger.warning(f"Error saving AI response: {e}")

        # Signal clean stream end — prevents client "unexpected EOF"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        agent_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )

# ===========================================================================================================================================================
# Analytical & Visualization Endpoints
# ===========================================================================================================================================================

@stats_router.get("/summary")
def api_get_database_summary():
    return get_database_summary()


@stats_router.get("/history")
def api_get_import_history(limit: int = 20):
    return get_import_history(limit)


@stats_router.get("/entity_counts")
def api_get_entity_counts():
    return get_entity_counts()


@graph_router.get("/search")
def api_search_nodes(term: str, limit: int = 10):
    return search_nodes(term, limit)


class GraphSampleRequest(BaseModel):
    node_types: List[str]
    rel_types: List[str]
    limit: int = 50
    focus_node_id: str = ""


@graph_router.post("/sample")
def api_get_graph_sample(request: GraphSampleRequest):
    return get_graph_sample(
        request.node_types,
        request.rel_types,
        request.limit,
        request.focus_node_id,
    )

# Include routers
app.include_router(system_router)
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(repair_router)
app.include_router(ingest_router)
app.include_router(stats_router)
app.include_router(graph_router)

# uvicorn main:app --reload
if __name__ == "__main__":
    # Run the app with Uvicorn, specifying host and port here
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
