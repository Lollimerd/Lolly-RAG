# 🍭 Lolly RAG: Agentic GraphRAG System for Technical Documents & Q&A

**Lolly RAG** is an end-to-end, high-performance **Agentic Graph Retrieval-Augmented Generation (GraphRAG)** application. It integrates local LLMs (via Ollama), Neo4j graph database vector and hybrid indexing, and document knowledge bases to provide context-aware, verifiable engineering answers and dynamic graph visualizations.

---

## 🏗 Architecture Overview

```mermaid
flowchart TD
    subgraph UI ["Frontend (Streamlit)"]
        WebUI["web_ui.py (Chat Interface)"]
        Explorer["pages/neo4j_explorer.py (Graph Explorer)"]
        DocIngestion["pages/doc_injestion.py (Document Explorer & Ingestion)"]
    end

    subgraph Backend ["Backend Service (FastAPI)"]
        API["app/main.py"]
        Agent["agents/agent.py (RAG Agent)"]
        Tools["tools/document_search.py (Document Search Tool)"]
        Memory["utils/memory.py (Neo4j Session & User Memory)"]
        DocProc["utils/doc_processor.py (Doc Chunking & Graph Ingestion)"]
        Middleware["middleware/mermaid.py (Syntax Validation)"]
    end

    subgraph Infrastructure ["Local AI & Graph Infrastructure"]
        Neo4j[("Neo4j DB (5.26)\n- Vector Indexes\n- Fulltext & Text Indexes\n- Document & Chunk Graph")]
        Ollama[("Ollama Server\n- qwen3.5:4b\n- jina-embeddings-v2")]
        Reranker["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
    end

    WebUI <-->|HTTP / SSE| API
    Explorer <-->|REST API| API
    DocIngestion <-->|REST API| API

    API --> Agent
    API --> DocProc
    DocProc --> Neo4j
    Agent --> Tools
    Agent --> Middleware
    Tools --> Neo4j
    Tools --> Reranker
    Agent --> Ollama
    Memory --> Neo4j
```

---

## 🌟 Key Features

* **🤖 Autonomous Agentic GraphRAG**: Powered by [`deepagents`](file:///home/lolli/projects/agentic-graphrag/lolly-rag/backend/agents/agent.py) and LangChain, utilizing hierarchical tool execution protocols to search document knowledge graphs or fallback gracefully to internal model knowledge.
* **⚡ Vector + Graph Hybrid Search**: Combines Neo4j vector cosine similarity indexes on `DocumentChunk` nodes with fulltext keyword indexes on documents, text indexes, metadata matching, and GPU-accelerated Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`).
* **📥 Multi-Format Document Ingestion**: Ingestion pipeline for PDFs, Word `.docx`, Markdown `.md`, and plain text `.txt` with automatic node creation, chunking, vector embedding generation (`jina-embeddings-v2-base-en`), and relationship wiring `(Document)-[:HAS_CHUNK]->(DocumentChunk)`.
* **📊 Visual Graph Explorer & Analytics**: Interactive PyVis network visualizers, database summaries, entity count distribution metrics, and graph sampling directly in Streamlit.
* **🧠 Persistent Graph Memory & Session Repair**: Chat history and user sessions are stored directly in Neo4j graph nodes. Startup routines automatically repair missing session relationships (`HAS_MESSAGE`).
* **🔮 Robust Middleware Pipeline**: Custom [`MermaidValidationMiddleware`](file:///home/lolli/projects/agentic-graphrag/lolly-rag/backend/middleware/mermaid.py) ensures valid syntax for streamed workflow and architectural diagrams.

---

## 📁 Repository Structure

```
lolly-rag/
├── backend/
│   ├── agents/
│   │   └── agent.py              # Agent initialization & system prompts
│   ├── app/
│   │   └── main.py               # FastAPI application & REST endpoints
│   ├── middleware/
│   │   ├── in_built.py           # Context & tool call middleware
│   │   └── mermaid.py            # Mermaid diagram validation middleware
│   ├── setup/
│   │   └── init_config.py        # Ollama LLM, embedding & Neo4j vector index setups
│   ├── tools/
│   │   └── document_search.py    # Multi-index hybrid search & Cross-Encoder reranking
│   ├── utils/
│   │   ├── dashboard.py          # Neo4j query helpers & graph statistics
│   │   ├── doc_processor.py      # Document parser, chunker & graph builder
│   │   ├── memory.py             # User and chat session graph operations
│   │   └── utils.py              # Environment & diagnostic tools
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── pages/
│   │   ├── doc_injestion.py      # Document file explorer & upload page
│   │   └── neo4j_explorer.py     # Interactive Neo4j graph viewer
│   ├── utils/
│   │   └── ui_utils.py           # Custom Streamlit UI components & layout helpers
│   ├── web_ui.py                 # Main Streamlit chat app
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml             # Containerized services (Ollama, Neo4j, Backend, Frontend)
├── pyproject.toml                 # Project metadata & pyright setup
├── requirements.txt               # Full Python dependency specifications
└── run.sh                         # Unified launch script for FastAPI & Streamlit
```

---

## 🚀 Quick Start

### 1. Prerequisites

* **Python 3.13+** (or standard virtual environment manager like `uv` or `venv`)
* **Docker & Docker Compose** (with NVIDIA Container Toolkit for GPU acceleration if available)
* **Ollama** running locally or via Docker

### 2. Environment Setup

Create or update your `.env` file in the root directory:

```env
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"

OLLAMA_BASE_URL="http://localhost:11434"
EMBEDDING_MODEL="jina/jina-embeddings-v2-base-en:latest"

BACKEND_URL="http://localhost:8000"
```

### 3. Option A: Run Locally via `run.sh`

Activate your environment and run the startup script:

```bash
chmod +x run.sh
./run.sh
```

This starts:
* **FastAPI Backend**: `http://localhost:8000` (Docs available at `http://localhost:8000/docs`)
* **Streamlit Frontend**: `http://localhost:8501`

### 4. Option B: Run with Docker Compose

To spin up all services including Neo4j, Ollama, FastAPI, and Streamlit:

```bash
docker-compose up --build -d
```

Service Ports:
* **Streamlit Frontend**: `http://localhost:8501`
* **FastAPI Backend**: `http://localhost:8000`
* **Neo4j Browser**: `http://localhost:7474`
* **Ollama API**: `http://localhost:11434`

---

## 🛠 Model Configuration

Model definitions and LLM parameters are set in [`backend/setup/init_config.py`](file:///home/lolli/projects/agentic-graphrag/lolly-rag/backend/setup/init_config.py):

| Role | Default Model / Class | Function |
| :--- | :--- | :--- |
| **Answer LLM** | `qwen3.5:4b` | Agent reasoning, tool orchestration & answer generation |
| **Embedding Model** | `jina-embeddings-v2-base-en` | 768-dim vector embeddings for Neo4j Vector Indexes |
| **Reranker Model** | `ms-marco-MiniLM-L-6-v2` | PyTorch GPU cross-encoder candidate re-scoring |
| **Summarizer LLM** | `qwen3.5:0.8b` | Historical chat context condensation |

---

## 🌐 API Reference

### System & Health
* `GET /`: API status welcome message
* `GET /health`: Health check timestamp
* `GET /config`: Runtime configuration details (Ollama model, Neo4j status)

### Chat & Users
* `GET /users`: Retrieve all registered users
* `GET /user/{user_id}/chats`: Retrieve sessions for a specified user
* `GET /chat/{session_id}`: Fetch message history for a session
* `POST /agent/ask`: Primary agent query endpoint (supports SSE streaming)

### Document Ingestion & Management
* `POST /ingest/documents`: Upload and chunk document (`.pdf`, `.docx`, `.txt`, `.md`)
* `GET /ingest/documents`: List uploaded documents metadata
* `GET /ingest/documents/{doc_id}/chunks`: Retrieve chunks for a document
* `PUT /ingest/documents/{doc_id}`: Update document description / folder metadata
* `DELETE /ingest/documents/{doc_id}`: Delete document and associated chunks

### Analytics & Graph
* `GET /stats/summary`: Database document, user, session, and message metrics
* `GET /stats/entity_counts`: Entity and relationship type counts
* `GET /graph/search`: Search knowledge graph nodes
* `POST /graph/sample`: Graph network topology sample for visualization

---

## 📄 License

This project is open-source and available under the MIT License.

