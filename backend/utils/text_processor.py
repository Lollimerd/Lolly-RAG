"""
text_processor.py
-----------------
Document loader and processor for text and document formats:
- PDF: .pdf (via PyPDFLoader with automated Nemotron OCR fallback for scanned/image-based PDFs)
- Word: .docx (via Docx2txtLoader)
- Text & Markdown: .txt, .md (via TextLoader with multi-encoding fallback)
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIN_PDF_TEXT_CHARS = 30

def _load_text_or_markdown(file_path: str, filename: str) -> List[Document]:
    """Args: file_path: File path, filename: File name."""
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            docs = TextLoader(file_path, encoding=enc).load()
            for doc in docs:
                doc.metadata.setdefault("filename", filename)
                doc.metadata.setdefault("source", filename)
            return docs
        except Exception:
            pass
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return [Document(page_content=f.read(), metadata={"filename": filename, "source": filename})]
    except Exception as exc:
        raise ValueError(f"Could not read text/markdown file '{filename}': {exc}") from exc

def _extract_page_ocr(page: Any, engine: Any) -> tuple[list[str], int]:
    """Args: page: PDF page object, engine: OCR engine."""
    texts = []
    images = getattr(page, "images", [])
    for img in images:
        data = getattr(img, "data", None)
        if data:
            extracted = engine.extract_text(data)
            if extracted and extracted.strip():
                texts.append(extracted.strip())
    return texts, len(images)

def _process_page_ocr(doc: Document, page: Any, engine: Any, filename: str, idx: int, total_pages: int) -> None:
    """Args: doc: Document object, page: PDF page, engine: OCR engine, filename: File name, idx: Page index, total_pages: Total pages."""
    try:
        content = (doc.page_content or "").strip()
        ocr_texts, img_cnt = _extract_page_ocr(page, engine)
        if ocr_texts:
            combined = "\n\n".join(ocr_texts)
            prefix = f"{content}\n\n" if content else f"# Document: {filename} (Page {idx}/{total_pages})\n"
            doc.page_content = f"{prefix}### Scanned Content (OCR):\n{combined}"
            doc.metadata.update({
                "is_ocr": True,
                "ocr_engine": getattr(engine, "name", "nemotron-ocr-v2"),
                "image_count": img_cnt,
            })
        elif not content:
            doc.page_content = f"# Document: {filename} (Page {idx}/{total_pages})\n*(Empty page or graphic objects without readable text)*"
    except Exception as err:
        logger.debug("Failed OCR processing on PDF page %d: %s", idx, err)

def _load_pdf_with_ocr(
    file_path: str,
    filename: str,
    ocr_engine: Optional[Any] = None,
    min_text_chars: int = MIN_PDF_TEXT_CHARS,
) -> List[Document]:
    """Args: file_path: PDF path, filename: File name, ocr_engine: Optional OCR engine, min_text_chars: Minimum text threshold."""
    try:
        docs = PyPDFLoader(file_path).load()
    except Exception as exc:
        raise ValueError(f"Could not load PDF file '{filename}': {exc}") from exc

    total_pages = len(docs)
    if total_pages == 0:
        empty_msg = f"# Document: {filename}\n*(Empty PDF document with no pages)*"
        return [Document(page_content=empty_msg, metadata={"filename": filename, "source": filename, "total_pages": 0})]

    engine, pdf_reader = None, None
    needs_ocr = any(len((d.page_content or "").strip()) < min_text_chars for d in docs)
    if needs_ocr:
        try:
            from setup.init_config import ocr_model
            engine = ocr_engine or ocr_model()
        except Exception as ocr_err:
            logger.debug("Could not init OCR model: %s", ocr_err)
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(file_path)
        except Exception as r_err:
            logger.debug("Could not open PdfReader: %s", r_err)

    for idx, doc in enumerate(docs, start=1):
        doc.metadata.setdefault("filename", filename)
        doc.metadata.setdefault("source", f"{filename} (Page {idx}/{total_pages})")
        doc.metadata.update({"page_number": idx, "total_pages": total_pages})
        content = (doc.page_content or "").strip()

        if len(content) < min_text_chars and engine and pdf_reader:
            page_idx = idx - 1
            if page_idx < len(pdf_reader.pages):
                _process_page_ocr(doc, pdf_reader.pages[page_idx], engine, filename, idx, total_pages)

    logger.info("Loaded %d page(s) from PDF '%s'", len(docs), filename)
    return docs

def load_text_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None) -> List[Document]:
    """Args: file_path: Document path, filename: File name, ocr_engine: Optional OCR engine."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _load_pdf_with_ocr(file_path, filename, ocr_engine=ocr_engine)
    if ext == ".docx":
        try:
            docs = Docx2txtLoader(file_path).load()  # type: ignore[abstract]
            for doc in docs:
                doc.metadata.setdefault("filename", filename)
                doc.metadata.setdefault("source", filename)
            logger.info("Loaded DOCX '%s' (%d sections)", filename, len(docs))
            return docs
        except Exception as exc:
            raise ValueError(f"Could not load Word file '{filename}': {exc}") from exc
    if ext in (".txt", ".md"):
        docs = _load_text_or_markdown(file_path, filename)
        logger.info("Loaded text/markdown file '%s' (%d sections)", filename, len(docs))
        return docs
    raise ValueError(f"Unsupported text file type '{ext}'. Supported: {', '.join(sorted(TEXT_EXTENSIONS))}")
