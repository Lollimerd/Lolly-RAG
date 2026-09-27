# 🍭 Lolly RAG: Agentic GraphRAG System for Knowledge Base

**Lolly RAG** is an end-to-end, high-performance **Agentic Graph Retrieval-Augmented Generation (GraphRAG)** system. It integrates local LLMs (via Ollama), Neo4j graph database vector and hybrid indexing, GPU-accelerated Cross-Encoder reranking, and **NVIDIA Nemotron OCR v2** to provide context-aware, verifiable engineering answers, multi-modal file ingestion, and dynamic graph visualizations.

---

## 🏗 Architecture Overview

```mermaid
flowchart TD
    subgraph UI ["Frontend (Streamlit)"]
        WebUI["web_ui.py\n(Chat Interface & Direct File Attachments)"]
        Explorer["pages/neo4j_explorer.py\n(Graph Explorer & Topology)"]
        DocIngestion["pages/doc_ingestion.py\n(Document Explorer & Batch Ingestion)"]
    end

    subgraph Backend ["Backend Service (FastAPI)"]
        API["app/main.py (REST & SSE Stream API)"]
        Agent["agents/agent.py (Autonomous RAG Agent)"]
        Tools["tools/document_search.py\n(Hybrid Vector + Fulltext Search)"]
        Memory["utils/memory.py (Neo4j Session & User Memory)"]
        DocProc["tools/doc_processor.py\n(Document Pipeline Orchestrator)"]
        MediaProc["utils/media_processor.py\n(Nemotron OCR & PPTX Loader)"]
        TabularProc["utils/tabular_processor.py\n(CSV & Excel / APOC Ingestion)"]
        TextProc["utils/text_processor.py\n(PDF, Word, Markdown, TXT)"]
        Middleware["middleware/in_built.py\n(Tool Limit, Retry & Context Middleware)"]
    end

    subgraph Infrastructure ["Local AI & Graph Infrastructure"]
        Neo4j[("Neo4j DB (5.26)\n- Vector Cosine Indexes\n- Fulltext & Text Indexes\n- Document & Chunk Graph")]
        Ollama[("Ollama Server\n- qwen3.5:4b (Reasoner)\n- qwen3.5:0.8b (Summarizer)\n- jina-embeddings-v2")]
        Reranker["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
        OCR["NVIDIA Nemotron OCR v2\n(Layout & Visual Text Extraction)"]
    end

    WebUI <-->|HTTP / SSE| API
    Explorer <-->|REST API| API
    DocIngestion <-->|REST API| API

    API --> Agent
    API --> DocProc
    DocProc --> MediaProc
    DocProc --> TabularProc
    DocProc --> TextProc
    MediaProc --> OCR
    TextProc --> OCR

    DocProc --> Neo4j
    TabularProc --> Neo4j

    Agent --> Tools
    Agent --> Middleware
    Tools --> Neo4j
    Tools --> Reranker
    Agent --> Ollama
    Memory --> Neo4j
```

---

## 🌟 Key Features

| Feature                                        | Description                                                                                                                                                                                                                                                                                         | Key Tech & Highlights                                                 |
| :--------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **🤖 Autonomous Agentic GraphRAG**              | Powered by`deepagents` and LangChain, utilizing hierarchical tool execution protocols to search document knowledge graphs or fallback gracefully to internal model knowledge.                                                                                                                       | LangChain,`deepagents`, Ollama (`qwen3.5:4b`)                         |
| **👁️ NVIDIA Nemotron OCR v2**                   | High-accuracy deep learning OCR pipeline replacing legacy Tesseract. Automatically transcribes text, tables, diagrams, and visual layout from images and scanned PDFs into structured searchable chunks.                                                                                            | `nemotron-ocr-v2`, `torchvision`, `shapely`, GPU-accelerated          |
| **📎 Native In-Chat File Attachments**          | Drag-and-drop or select multiple documents/media directly inside the chat bar. Automatically ingests attachments into the knowledge graph under "Chat Uploads" before querying the agent.                                                                                                           | Streamlit`st.chat_input(accept_file="multiple")`, SSE status feedback |
| **📊 Multi-Modal Ingestion Pipeline**           | Specialized loaders for diverse formats: Spreadsheets (`.xlsx`, `.xls`), CSV (`.csv` via APOC/chunking), PowerPoint (`.pptx`, `.ppt` with slide/table/notes extraction), PDF (with OCR fallback), Word (`.docx`), Markdown (`.md`), and images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`). | `python-pptx`, `openpyxl`, `pandas`, `pypdf`, `Pillow`                |
| **⚡ Vector + Graph Hybrid Search**             | Combines Neo4j vector cosine similarity indexes on`DocumentChunk` nodes with fulltext keyword indexes, metadata matching, and GPU-accelerated Cross-Encoder reranking.                                                                                                                              | Neo4j Vector Indexes, Fulltext Search,`ms-marco-MiniLM-L-6-v2`        |
| **🔍 Smart Intent Detection**                   | Search tool automatically detects visual/diagram, presentation/slide, and tabular dataset intent from natural user queries, routing to specialized chunk types (`[Media: Image & OCR Text]`, `[Dataset: Tabular Records]`).                                                                         | Regex Query Classifier, Metadata Filtering                            |
| **📈 Visual Graph Explorer & Analytics**        | Interactive PyVis network visualizers, database summaries, entity count distribution metrics, and graph sampling directly in Streamlit.                                                                                                                                                             | PyVis Network Visualizer, Streamlit Analytics                         |
| **🧠 Persistent Graph Memory & Session Repair** | Chat history and user sessions are stored directly in Neo4j graph nodes. Startup routines automatically repair missing session relationships (`HAS_MESSAGE`).                                                                                                                                       | Neo4j Graph Sessions, Auto-Healing Graph Routines                     |
| **🔮 Robust Middleware Pipeline**               | Integrated`middleware/in_built.py` handling automatic summarization, tool call limits (3 runs), contextual trimming, and tool retry mechanisms.                                                                                                                                                     | Summarization, Context Editing, Tool Call Limits & Retries            |

---

## 📁 Repository Structure

```
lolly-rag/
├── backend/
│   ├── agents/
│   │   └── agent.py              # Agent initialization & system prompts (Document-First Rule)
│   ├── app/
│   │   └── main.py               # FastAPI application, REST endpoints & SSE streaming
│   ├── middleware/
│   │   └── in_built.py           # Context, tool limit & retry middleware
│   ├── setup/
│   │   └── init_config.py        # Ollama LLM, Nemotron OCR v2 & Neo4j vector configs
│   ├── tools/
│   │   ├── doc_processor.py      # Main document ingestion & dispatch orchestrator
│   │   └── document_search.py    # Multi-index hybrid search & Cross-Encoder reranking
│   ├── utils/
│   │   ├── dashboard.py          # Neo4j query helpers & graph statistics
│   │   ├── media_processor.py    # PowerPoint (.pptx) & Nemotron OCR image loaders
│   │   ├── tabular_processor.py  # CSV & Excel (.xlsx, .xls) chunking & APOC ingestion
│   │   ├── text_processor.py     # PDF (with OCR fallback), DOCX, MD & TXT loaders
│   │   ├── memory.py             # User and chat session graph operations
│   │   └── utils.py              # Environment & diagnostic tools
│   ├── Dockerfile                # Python 3.12 image with CUDA runtime dependencies
│   └── requirements.txt
├── frontend/
│   ├── pages/
│   │   ├── doc_ingestion.py      # Document explorer, folder manager & batch upload page
│   │   └── neo4j_explorer.py     # Interactive Neo4j graph viewer & analytics
│   ├── utils/
│   │   ├── doc_utils.py          # Frontend document API helper routines
│   │   └── ui_utils.py           # Streamlit UI styling & component helpers
│   ├── web_ui.py                 # Main Streamlit chat UI with native multi-file uploads
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml             # Full-stack composition (Ollama, Neo4j, Backend, Frontend)
├── pyproject.toml                 # Project dependencies & Python >=3.12 requirement
├── start.sh                       # Local development launcher (FastAPI + Streamlit via .venv)
└── .env.example                   # Environment configuration template
```

---

## 🚀 Setup & Getting Started

Choose the setup mode that best fits your workflow:

| Setup Mode                                                       | Best For                                   | Prerequisites                     | Key Command                                    |
| :--------------------------------------------------------------- | :----------------------------------------- | :-------------------------------- | :--------------------------------------------- |
| [**1. Docker Compose**](#1-docker-compose-quickest)              | Quickest complete stack deployment         | Docker & NVIDIA Container Toolkit | `docker compose up --build -d`                 |
| [**2. Local Bare-Metal**](#2-local-bare-metal-development)       | Rapid local iteration & active development | Python 3.12+, `uv`, local GPU     | `./start.sh`                                   |
| [**3. Air-Gapped / Offline**](#3-air-gapped--offline-deployment) | Environments without internet access       | Docker or `.venv` bundle          | `./setup-airgap.sh` → `./run-airgap-docker.sh` |

---

### 1. Docker Compose (Quickest)

Spins up all services (**FastAPI backend**, **Streamlit frontend**, **Neo4j 5.26**, and **Ollama**) in containers with GPU support:

```bash
# 1. Clone repository & configure environment
git clone <repository-url>
cd Lolly-RAG
cp .env.example .env

# 2. Build and launch all services in background
docker compose up --build -d
```

**Service Endpoints:**
* 🎨 **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501) (or `http://localhost:8511`)
* ⚡ **FastAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* 🗄️ **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) *(Credentials: `neo4j` / `password`)*
* 🦙 **Ollama API**: [http://localhost:11434](http://localhost:11434)

---

### 2. Local Bare-Metal Development

Run services directly on the host using [`uv`](https://docs.astral.sh/uv/) for fast virtual environment and package management.

#### Step 1: Environment & Virtualenv
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & enter directory
git clone <repository-url>
cd Lolly-RAG

# Create virtual environment & sync dependencies
uv venv --python 3.12 .venv
uv sync

# Configure environment variables
cp .env.example .env
```

#### Step 2: Start External Services (Ollama & Neo4j)
```bash
# Pull required Ollama models
ollama pull qwen3.5:4b
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b

# Run Neo4j with APOC plugin enabled
docker run -d \
  --name lolly-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.26
```

#### Step 3: Launch Local Services
```bash
chmod +x start.sh
./start.sh
```
*`./start.sh` automatically activates `.venv` and concurrently launches FastAPI (port 8000) and Streamlit (port 8501). Press `Ctrl+C` to gracefully stop both.*

---

### 3. Air-Gapped / Offline Deployment

Deploy Lolly RAG into secure environments with **no outbound internet access**.

#### Phase 1: Preparation (Online Machine)
Run the automated preparation script on an internet-connected machine:

```bash
chmod +x setup-airgap.sh

# Option A: Prepare for Docker deployment (default — fast)
./setup-airgap.sh

# Option B: Prepare for bare-metal host deployment (syncs .venv via uv)
./setup-airgap.sh --bare-metal

# Option C: Prepare both Docker & bare-metal artifacts
./setup-airgap.sh --all

# Optional: Verify all offline artifacts locally
./setup-airgap.sh --verify
```

**Self-Contained Artifacts Generated:**

| Artifact                | Local Location          | Purpose                                                   |
| :---------------------- | :---------------------- | :-------------------------------------------------------- |
| **Docker Images**       | `docker-images/*.tar`   | Pre-saved images for Backend, Frontend, Neo4j, Ollama     |
| **Ollama Models**       | `ollama-models/models/` | Direct bind-mounted model blobs & manifests               |
| **Hugging Face Models** | `.cache/huggingface/`   | Offline weights for Cross-Encoder & Nemotron OCR v2       |
| **PyTorch Backbone**    | `.cache/torch/`         | RegNet backbone weights for Nemotron OCR layout detection |
| **Python Virtualenv**   | `.venv/`                | Fully synced Python 3.12 virtual environment (bare-metal) |

#### Phase 2: Transfer to Air-Gapped Machine
Copy the **entire repository directory** to the target air-gapped machine via external SSD, USB, or secure scp/rsync:

```bash
rsync -avP Lolly-RAG/ user@airgap-machine:/home/user/Lolly-RAG/
```

#### Phase 3: Launch Offline Stack
On the target air-gapped machine:

* **Docker Mode (1-Command Launcher)**:
  ```bash
  chmod +x run-airgap-docker.sh
  ./run-airgap-docker.sh
  ```
  *(Automatically loads `docker-images/*.tar`, configures `.env`, and starts the complete containerized stack via `docker compose up -d`)*

* **Bare-Metal Mode**:
  ```bash
  # 1. Configure environment
  cp .env.example .env

  # 2. Ensure host Neo4j and Ollama services are running, then launch:
  ./start.sh
  ```

---

## 🛠 Model Configuration

Model definitions, OCR pipelines, and LLM parameters are managed in [`backend/setup/init_config.py`](backend/setup/init_config.py):

| Role                | Default Model / Class                  | Function                                                        |
| :------------------ | :------------------------------------- | :-------------------------------------------------------------- |
| **Answer LLM**      | `qwen3.5:4b`                           | Agent reasoning, tool orchestration & answer generation         |
| **Embedding Model** | `qwen3-embedding:0.6b`                 | 1024-dimensional vector embeddings for Neo4j Vector Indexes     |
| **Reranker Model**  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | PyTorch GPU cross-encoder candidate re-scoring                  |
| **OCR Engine**      | `nvidia/nemotron-ocr-v2`               | Deep learning OCR, layout segmentation & visual text extraction |
| **Summarizer LLM**  | `qwen3.5:0.8b`                         | Historical chat context condensation & token management         |

---

## 📄 License

This project is open-source and available under the MIT License.
