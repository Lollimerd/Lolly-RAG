## 🌐 API Reference

### System & Health

* `GET /`: API status welcome message
* `GET /health`: Health check timestamp
* `GET /config`: Runtime configuration details (Ollama model, Neo4j status)

### Chat & Users

* `GET /users`: Retrieve all registered users
* `GET /user/{user_id}/chats`: Retrieve sessions for a specified user
* `DELETE /user/{user_id}`: Delete user and all associated chat history
* `GET /chat/{session_id}`: Fetch message history for a session
* `DELETE /chat/{session_id}`: Delete a specific chat session
* `POST /repair-sessions`: Repair orphaned messages and missing graph relationships
* `POST /agent/ask`: Primary agent query endpoint (supports SSE streaming & `attached_files`)

### Document Ingestion & Management

* `POST /ingest/documents`: Upload and chunk documents, spreadsheets, presentations, and images (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.png`, `.jpg`, etc.)
* `POST /ingest/apoc/csv`: Ingest large CSV files directly using Neo4j APOC
* `GET /ingest/documents`: List uploaded documents metadata
* `GET /ingest/documents/{doc_id}/chunks`: Retrieve chunks for a document
* `PUT /ingest/documents/{doc_id}`: Update document description or folder metadata
* `DELETE /ingest/documents/{doc_id}`: Delete document and associated chunks

### Analytics & Graph

* `GET /stats/summary`: Database document, user, session, and message metrics
* `GET /stats/history`: Activity history log
* `GET /stats/entity_counts`: Entity and relationship type counts
* `GET /graph/search`: Search knowledge graph nodes
* `POST /graph/sample`: Graph network topology sample for PyVis visualization