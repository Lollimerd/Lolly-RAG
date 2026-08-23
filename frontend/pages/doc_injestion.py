import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from utils.advanced_rag import contextualize_chunk
from rank_bm25 import BM25Okapi
import os
import re

# Cap contextual-retrieval LLM calls so huge uploads don't hang forever
MAX_CONTEXTUAL_CHUNKS = 150


def reset_documents():
    st.session_state.documents_loaded = False
    st.session_state.retrieval_pipeline = None
    st.session_state.processing = False
    st.session_state.suggested_questions = []
    st.session_state.semantic_cache = []


def _apply_contextual_retrieval(texts, documents, llm_uri, llm_model):
    """Prepend an LLM-generated situating sentence to each chunk (Anthropic technique)."""
    # Build a short summary per source document
    summaries = {}
    per_source_text = {}
    for d in documents:
        src = d.metadata.get("source", "doc")
        per_source_text[src] = per_source_text.get(src, "") + "\n" + d.page_content
    for src, full in per_source_text.items():
        summaries[src] = full.strip()[:1500]
    global_summary = " ".join(per_source_text.values())[:1500]

    total = min(len(texts), MAX_CONTEXTUAL_CHUNKS)
    progress = st.progress(0.0, text="Contextualizing chunks…")
    for i, chunk in enumerate(texts[:total]):
        src = chunk.metadata.get("source", "doc")
        summary = summaries.get(src, global_summary)
        ctx = contextualize_chunk(chunk.page_content, summary, llm_uri, llm_model)
        if ctx:
            chunk.metadata["context"] = ctx
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
        progress.progress((i + 1) / total, text=f"Contextualizing chunks… {i + 1}/{total}")
    progress.empty()
    return texts


def process_documents(uploaded_files, reranker, embedding_model, base_url,
                      llm_model=None, enable_contextual=False):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []

    if not os.path.exists("temp"):
        os.makedirs("temp")

    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                os.remove(file_path)
                continue

            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            st.session_state.processing = False
            return

    if not documents:
        st.error("No valid documents could be loaded.")
        st.session_state.processing = False
        return

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)

    # 🚀 Contextual Retrieval — enrich each chunk before embedding/indexing
    if enable_contextual and llm_model:
        llm_uri = f"{base_url}/api/generate"
        texts = _apply_contextual_retrieval(texts, documents, llm_uri, llm_model)

    text_contents = [doc.page_content for doc in texts]

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    try:
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"Failed to build vector store: {str(e)}")
        st.session_state.processing = False
        return

    bm25_retriever = BM25Retriever.from_texts(
        text_contents,
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts),
        "doc_chunks": texts,
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False
