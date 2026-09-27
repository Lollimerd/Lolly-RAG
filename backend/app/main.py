import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from starlette.middleware import Middleware
import uvicorn

from agents.agent import rag_agent
from setup.init_config import (
    NEO4J_PASSWORD,
    NEO4J_URL,
    NEO4J_USERNAME,
    answer_LLM,
    create_constraints,
    embedding_model,
    get_graph_instance,
)
from tools.doc_processor import (
    SUPPORTED_EXTENSIONS,
    delete_document,
    get_document_chunks,
    list_documents,
    process_uploaded_file,
    process_uploaded_file_stream,
    update_document,
)
from utils.dashboard import (
    get_database_summary,
    get_entity_counts,
    get_graph_sample,
    get_import_history,
    search_nodes,
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
)
from utils.utils import find_container_by_port, reset_tool_call_count

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

_CHAT_HISTORY_QUERY = """
MATCH (s:Session {id: $session_id})-[:HAS_MESSAGE]->(m:Message)
RETURN m.type AS role,
       m.content AS content,
       m.thought AS thought
ORDER BY coalesce(m.created_at, elementId(m)) ASC
LIMIT 1000
"""

_AGENT_HISTORY_QUERY = """
MATCH (s:Session {id: $sid})-[:HAS_MESSAGE]->(m:Message)
RETURN m.type AS role,
       m.content AS content
ORDER BY coalesce(m.created_at, elementId(m)) ASC
LIMIT 50
"""

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Args: _: FastAPI application."""
    try:
        create_constraints(get_graph_instance())
        logger.info("Database constraints verified.")
        if (repaired := repair_missing_has_message_relationships()) > 0:
            logger.info("Checked %d sessions for missing relationships", repaired)
    except Exception as e:
        logger.error("Database constraint init failed: %s", e)
    yield

app = FastAPI(title="GraphRAG API", version="1.2.0", middleware=middleware, lifespan=lifespan)

system_router = APIRouter(tags=["System"])
users_router = APIRouter(tags=["Users"])
chat_router = APIRouter(tags=["Chat"])
repair_router = APIRouter(tags=["Repair"])
ingest_router = APIRouter(prefix="/ingest", tags=["Ingestion"])
stats_router = APIRouter(prefix="/stats", tags=["Analytics"])
graph_router = APIRouter(prefix="/graph", tags=["Graph"])

class QueryRequest(BaseModel):
    question: str
    session_id: str
    user_id: str = "test_user"
    mode: str = "auto"
    attached_files: Optional[List[str]] = None
    files: Optional[List[str]] = None
    images: Optional[List[Union[str, Dict[str, Any]]]] = None

class DocumentUploadResponse(BaseModel):
    status: str
    doc_id: str = ""
    filename: str = ""
    chunk_count: int = 0
    message: str = ""

class DocumentUpdateRequest(BaseModel):
    description: str = ""

class GraphSampleRequest(BaseModel):
    node_types: List[str]
    rel_types: List[str]
    limit: int = 50
    focus_node_id: str = ""

@system_router.get("/")
def index():
    """Args: None."""
    return {"status": "online", "message": "Welcome to the GraphRAG API"}

@system_router.get("/health")
def health_check():
    """Args: None."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@system_router.get("/config")
def get_configuration():
    """Args: None."""
    try:
        parsed = urlparse(NEO4J_URL)
        discovered = find_container_by_port(parsed.port or 7687)
        name = f"{parsed.hostname} (Configured Host)" if any(k in discovered for k in ["not mounted", "Error", "Invalid"]) else discovered
        return {
            "ollama_model": answer_LLM().model,
            "neo4j_url": NEO4J_URL,
            "container_name": name,
            "neo4j_user": NEO4J_USERNAME,
            "status": "success",
        }
    except Exception as e:
        logger.error("Error in get_configuration: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "ollama_model": "unknown",
            "neo4j_url": NEO4J_URL,
            "container_name": "unknown",
        }

@users_router.get("/users")
def get_users():
    """Args: None."""
    try:
        return {"users": get_all_users(), "status": "success"}
    except Exception as e:
        return {
            "users": [],
            "status": "error",
            "message": str(e),
        }

@users_router.get("/user/{user_id}/chats")
def get_user_chats(user_id: str):
    """Args: user_id: User identifier."""
    try:
        return {"chats": get_user_sessions(user_id), "status": "success"}
    except Exception as e:
        return {
            "chats": [],
            "status": "error",
            "message": str(e),
        }

@chat_router.get("/chat/{session_id}")
def get_chat_messages(session_id: str):
    """Args: session_id: Session identifier."""
    try:
        results = get_graph_instance().query(_CHAT_HISTORY_QUERY, params={"session_id": session_id}) or []
        formatted = [
            {"role": "assistant" if d.get("role") in ("ai", "assistant") else d.get("role", "user"),
             "content": d.get("content", ""), "thought": d.get("thought")}
            for r in results for d in [dict(r) if isinstance(r, dict) else {"role": "user", "content": str(r)}]
        ]
        return {"messages": formatted, "status": "success"}
    except Exception as e:
        logger.error("Error fetching chat history: %s", e)
        return {"messages": [], "status": "error", "message": str(e)}

@chat_router.delete("/chat/{session_id}")
def delete_user_session(session_id: str):
    """Args: session_id: Session identifier."""
    try:
        delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@repair_router.post("/repair-sessions")
def repair_sessions():
    """Args: None."""
    try:
        repaired = repair_missing_has_message_relationships()
        return {
            "status": "success",
            "message": f"Checked/repaired {repaired} sessions",
            "repaired_count": repaired,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to repair: {e}")

@users_router.delete("/user/{user_id}")
def delete_app_user(user_id: str):
    """Args: user_id: User identifier."""
    try:
        delete_user(user_id)
        return {"status": "success", "message": f"User {user_id} deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _validate_file_upload(file: UploadFile) -> str:
    """Args: file: Uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return ext

@ingest_router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    description: str = Form(default=""),
    force: bool = Form(default=False),
    engine: Optional[str] = Form(default=None),
):
    """Args: file: Uploaded file, user_id: User identifier, description: Document description, force: Force overwrite flag, engine: Ingestion engine."""
    _validate_file_upload(file)
    try:
        file_bytes = await file.read()
        res = await asyncio.to_thread(
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
            status=res.get("status", "success"),
            doc_id=res["doc_id"],
            filename=res["filename"],
            chunk_count=res["chunk_count"],
            message=res.get("message") or f"Ingested successfully with {res['chunk_count']} chunks.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

async def _stream_to_sse(sync_gen_fn: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
    """Args: sync_gen_fn: Generator callable."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    filename = kwargs.get("filename", "")

    def _worker():
        try:
            for event in sync_gen_fn(*args, **kwargs):
                loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception as exc:
            logger.error("Error in stream for '%s': %s", filename, exc, exc_info=True)
            loop.call_soon_threadsafe(
                q.put_nowait,
                {
                    "type": "error",
                    "status": "error",
                    "progress": 1.0,
                    "filename": filename,
                    "message": str(exc),
                },
            )
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=_worker, daemon=True).start()
    try:
        while (item := await q.get()) is not None:
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@ingest_router.post("/documents/stream")
async def upload_document_stream(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    description: str = Form(default=""),
    force: bool = Form(default=False),
    engine: Optional[str] = Form(default=None),
) -> StreamingResponse:
    """Args: file: Uploaded document, user_id: User identifier, description: Document description, force: Force overwrite flag, engine: Ingestion engine."""
    _validate_file_upload(file)
    file_bytes = await file.read()
    return StreamingResponse(
        _stream_to_sse(
            process_uploaded_file_stream,
            file_bytes=file_bytes,
            filename=file.filename,
            user_id=user_id,
            description=description,
            graph=get_graph_instance(),
            embedder=embedding_model(),
            force=force,
            engine=engine,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

@ingest_router.post("/apoc/csv", response_model=DocumentUploadResponse)
async def upload_csv_apoc(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    description: str = Form(default=""),
    force: bool = Form(default=False),
):
    """Args: file: Uploaded CSV file, user_id: User identifier, description: Document description, force: Force overwrite flag."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported for APOC ingestion.")
    try:
        file_bytes = await file.read()
        res = await asyncio.to_thread(
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
            status=res.get("status", "success"),
            doc_id=res["doc_id"],
            filename=res["filename"],
            chunk_count=res["chunk_count"],
            message=res.get("message") or f"Ingested via APOC with {res['chunk_count']} records.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"APOC Ingestion failed: {e}")

@ingest_router.get("/documents")
async def get_documents():
    """Args: None."""
    try:
        docs = await asyncio.to_thread(list_documents, get_graph_instance())
        return {"status": "success", "documents": docs, "count": len(docs)}
    except Exception as e:
        return {"status": "error", "message": str(e), "documents": []}

@ingest_router.get("/documents/{doc_id}/chunks")
async def get_doc_chunks(doc_id: str):
    """Args: doc_id: Document ID."""
    try:
        chunks = await asyncio.to_thread(get_document_chunks, doc_id, get_graph_instance())
        return {"status": "success", "chunks": chunks, "count": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e), "chunks": []}

@ingest_router.put("/documents/{doc_id}")
async def update_doc_metadata(doc_id: str, req: DocumentUpdateRequest):
    """Args: doc_id: Document ID, req: Update request."""
    try:
        if not await asyncio.to_thread(update_document, doc_id, req.description, get_graph_instance()):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"status": "success", "message": f"Document '{doc_id}' metadata updated."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

@ingest_router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Args: doc_id: Document ID."""
    try:
        if not await asyncio.to_thread(delete_document, doc_id, get_graph_instance()):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"status": "success", "message": f"Document '{doc_id}' and chunks deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

def _sse(data: dict) -> str:
    """Args: data: Event dictionary."""
    return f"data: {json.dumps(data)}\n\n"

def _parse_chunk(chunk) -> tuple[str, str]:
    """Args: chunk: Streaming chunk."""
    raw = getattr(chunk, "content", "")
    content = raw if isinstance(raw, str) else "".join(
        p if isinstance(p, str) else (p.get("text", "") if p.get("type") == "text" else "")
        for p in (raw if isinstance(raw, list) else [])
    ) if raw else ""
    reasoning = "".join(
        p.get("reasoning") or p.get("thinking") or p.get("text", "")
        for p in (raw if isinstance(raw, list) else []) if isinstance(p, dict) and p.get("type") in ("reasoning", "thinking")
    )
    if hasattr(chunk, "additional_kwargs") and isinstance(chunk.additional_kwargs, dict):
        reasoning += chunk.additional_kwargs.get("reasoning_content") or chunk.additional_kwargs.get("thinking") or ""

    if "<think>" in content:
        prefix, rest = content.split("<think>", 1)
        r_part, c_part = rest.split("</think>", 1) if "</think>" in rest else (rest, "")
        content, reasoning = prefix + c_part, reasoning + r_part
    elif "</think>" in content:
        r_part, c_part = content.split("</think>", 1)
        content, reasoning = c_part, reasoning + r_part
    return content, reasoning

_TOOL_START_MSG = {"document_search_tool": "📄 Searching uploaded documents..."}
_TOOL_END_MSG = {"document_search_tool": "✅ Document search completed"}

@chat_router.post("/agent/ask")
async def agent_ask(request: QueryRequest) -> StreamingResponse:
    """Args: request: Query request payload."""

    async def agent_stream_generator() -> AsyncGenerator[str, None]:
        reset_tool_call_count(request.session_id)
        resp_chunks, resp_thought = [], []
        try:
            attached = list(dict.fromkeys(
                [f for src in filter(None, [request.attached_files, request.files]) for f in src]
                + [img if isinstance(img, str) else img.get("filename") for img in (request.images or []) if img and (isinstance(img, str) or img.get("filename"))]
            ))
            eff_q = f"[Attached File(s): {', '.join(attached)}]\n{request.question}" if attached and not any(f in request.question for f in attached) else request.question
            messages = []
            try:
                rows = await asyncio.to_thread(lambda: get_graph_instance().query(_AGENT_HISTORY_QUERY, params={"sid": request.session_id}))
                messages = [
                    HumanMessage(content=r.get("content", "")) if r.get("role") == "user" else AIMessage(content=r.get("content", ""))
                    for r in (rows or []) if r.get("role") in ("user", "assistant", "ai")
                ]
            except Exception as e:
                logger.warning("Could not load history for %s: %s", request.session_id, e)

            input_data = {
                "messages": messages + [HumanMessage(content=eff_q)],
                "question": eff_q,
                "session_id": request.session_id,
            }
            try:
                await asyncio.to_thread(link_session_to_user, request.session_id, request.user_id, request.question)
                await asyncio.to_thread(add_user_message_to_session, request.session_id, request.question)
            except Exception as e:
                logger.warning("Error saving user message: %s", e)

            async for event in rag_agent.astream_events(input_data, version="v2"):
                etype, ename = event["event"], event["name"]
                if etype == "on_tool_start":
                    yield _sse({
                        "type": "status",
                        "stage": "tool_start",
                        "status": "running",
                        "message": _TOOL_START_MSG.get(ename, f"🛠️ Using tool: {ename}..."),
                    })
                elif etype == "on_tool_end":
                    yield _sse({
                        "type": "status",
                        "stage": "tool_end",
                        "status": "complete",
                        "message": _TOOL_END_MSG.get(ename, f"✅ Tool {ename} completed"),
                    })
                elif etype == "on_chat_model_stream" and "answer_llm" in event.get("tags", []):
                    if chunk := event["data"].get("chunk"):
                        c, r = _parse_chunk(chunk)
                        if c or r:
                            yield _sse({
                                "type": "token",
                                "content": c,
                                "reasoning_content": r,
                            })
                            if c: resp_chunks.append(c)
                            if r: resp_thought.append(r)
        except Exception as e:
            logger.error("Error in agent stream: %s", e, exc_info=True)
            yield _sse({"type": "error", "content": str(e)})
        finally:
            try:
                full_resp, full_th = "".join(resp_chunks), "".join(resp_thought)
                if full_resp:
                    await asyncio.to_thread(add_ai_message_to_session, request.session_id, full_resp, full_th)
            except Exception as e:
                logger.warning("Error saving AI response: %s", e)
            yield "data: [DONE]\n\n"

    return StreamingResponse(agent_stream_generator(), media_type="text/event-stream", headers=SSE_HEADERS)

@stats_router.get("/summary")
def api_get_database_summary():
    """Args: None."""
    return get_database_summary()

@stats_router.get("/history")
def api_get_import_history(limit: int = 20):
    """Args: limit: Maximum items."""
    return get_import_history(limit)

@stats_router.get("/entity_counts")
def api_get_entity_counts():
    """Args: None."""
    return get_entity_counts()

@graph_router.get("/search")
def api_search_nodes(term: str, limit: int = 10):
    """Args: term: Search query, limit: Maximum matches."""
    return search_nodes(term, limit)

@graph_router.post("/sample")
def api_get_graph_sample(request: GraphSampleRequest):
    """Args: request: Graph sample request."""
    return get_graph_sample(request.node_types, request.rel_types, request.limit, request.focus_node_id)

app.include_router(system_router)
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(repair_router)
app.include_router(ingest_router)
app.include_router(stats_router)
app.include_router(graph_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
