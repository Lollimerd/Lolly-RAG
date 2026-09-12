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

# Supported text & document extensions
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIN_PDF_TEXT_CHARS = 30  # Minimum digital characters before triggering OCR fallback


# ---------------------------------------------------------------------------
# Plain Text & Markdown Loader
# ---------------------------------------------------------------------------

def _load_text_or_markdown(file_path: str, filename: str) -> List[Document]:
    """
    Load a plain text or markdown file with multi-encoding fallback.
    Tries utf-8, utf-8-sig, latin1, cp1252, and finally utf-8 with error replacement.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for enc in encodings_to_try:
        try:
            loader = TextLoader(file_path, encoding=enc)
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = filename
            return docs
        except Exception:
            continue

    # Fallback to UTF-8 with replacement if all else fails
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return [
            Document(
                page_content=content,
                metadata={"filename": filename, "source": filename},
            )
        ]
    except Exception as exc:
        raise ValueError(f"Could not read text/markdown file '{filename}': {exc}") from exc


# ---------------------------------------------------------------------------
# PDF Loader with Scanned Image OCR Fallback
# ---------------------------------------------------------------------------

def _load_pdf_with_ocr(
    file_path: str,
    filename: str,
    ocr_engine: Optional[Any] = None,
    min_text_chars: int = MIN_PDF_TEXT_CHARS,
) -> List[Document]:
    """
    Load a PDF document into LangChain Documents.

    If pages contain digital selectable text, it extracts them directly.
    If a page contains sparse or no text (< min_text_chars, e.g. scanned PDFs or image pages),
    it automatically extracts embedded page images and runs OCR text extraction.
    """
    try:
        docs = PyPDFLoader(file_path).load()
    except Exception as exc:
        raise ValueError(f"Could not load PDF file '{filename}': {exc}") from exc

    total_pages = len(docs)
    if total_pages == 0:
        return [
            Document(
                page_content=f"# Document: {filename}\n*(Empty PDF document with no pages)*",
                metadata={"filename": filename, "source": filename, "total_pages": 0},
            )
        ]

    # Check if any pages need OCR fallback
    needs_ocr = any(len((d.page_content or "").strip()) < min_text_chars for d in docs)

    engine = None
    pdf_reader = None
    if needs_ocr:
        try:
            from setup.init_config import ocr_model
            engine = ocr_engine or ocr_model()
        except Exception as ocr_err:
            logger.debug("Could not initialize OCR model for PDF scanning: %s", ocr_err)
            engine = None

        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(file_path)
        except Exception as reader_err:
            logger.debug("Could not open PdfReader for image extraction: %s", reader_err)
            pdf_reader = None

    for idx, doc in enumerate(docs, start=1):
        doc.metadata.setdefault("filename", filename)
        if not doc.metadata.get("source"):
            doc.metadata["source"] = f"{filename} (Page {idx}/{total_pages})"
        doc.metadata["page_number"] = idx
        doc.metadata["total_pages"] = total_pages

        content = (doc.page_content or "").strip()

        # If page is sparse / scanned and OCR is available, extract images and OCR them
        if len(content) < min_text_chars and engine is not None and pdf_reader is not None:
            try:
                page_idx = idx - 1
                if page_idx < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_idx]
                    page_images_ocr = []
                    image_count = 0

                    if hasattr(page, "images"):
                        for img in page.images:
                            image_count += 1
                            img_data = getattr(img, "data", None)
                            if img_data:
                                try:
                                    ocr_text = engine.extract_text(img_data)
                                    if ocr_text and ocr_text.strip():
                                        page_images_ocr.append(ocr_text.strip())
                                except Exception as img_err:
                                    logger.debug("OCR failed for image on page %d in '%s': %s", idx, filename, img_err)

                    if page_images_ocr:
                        ocr_combined = "\n\n".join(page_images_ocr)
                        if content:
                            doc.page_content = f"{content}\n\n### Scanned / Image Content (OCR):\n{ocr_combined}"
                        else:
                            header = f"# Document: {filename} (Page {idx}/{total_pages})\n"
                            doc.page_content = f"{header}### Scanned Content (OCR):\n{ocr_combined}"

                        doc.metadata["is_ocr"] = True
                        doc.metadata["ocr_engine"] = getattr(engine, "name", "nemotron-ocr-v2")
                        doc.metadata["image_count"] = image_count
                        logger.info("Extracted OCR text for scanned PDF page %d of '%s'", idx, filename)

                    elif not content:
                        doc.page_content = (
                            f"# Document: {filename} (Page {idx}/{total_pages})\n"
                            "*(Empty page or graphic objects with no readable text detected)*"
                        )
            except Exception as page_ocr_err:
                logger.debug("Failed OCR processing on PDF page %d of '%s': %s", idx, filename, page_ocr_err)

    logger.info("Loaded %d page(s) from PDF '%s'", len(docs), filename)
    return docs


# ---------------------------------------------------------------------------
# High-Level Document Dispatcher
# ---------------------------------------------------------------------------

def load_text_document(
    file_path: str,
    filename: str,
    ocr_engine: Optional[Any] = None,
) -> List[Document]:
    """
    Load a text or standard document file (.pdf, .docx, .txt, .md) into LangChain Documents.

    Args:
        file_path: Absolute or relative filesystem path to the file.
        filename: Original or canonical filename with extension.
        ocr_engine: Optional OCR engine instance for scanned PDF fallback.

    Returns:
        List of Document objects extracted from the file.

    Raises:
        ValueError: If the file extension is not supported by this processor.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _load_pdf_with_ocr(file_path, filename, ocr_engine=ocr_engine)

    elif ext == ".docx":
        try:
            docs = Docx2txtLoader(file_path).load()  # type: ignore[abstract]
            for doc in docs:
                doc.metadata.setdefault("filename", filename)
                if not doc.metadata.get("source"):
                    doc.metadata["source"] = filename
            logger.info("Loaded DOCX '%s' (%d doc sections)", filename, len(docs))
            return docs
        except Exception as exc:
            raise ValueError(f"Could not load Word file '{filename}': {exc}") from exc

    elif ext in (".txt", ".md"):
        docs = _load_text_or_markdown(file_path, filename)
        logger.info("Loaded text/markdown file '%s' (%d doc sections)", filename, len(docs))
        return docs

    else:
        raise ValueError(
            f"Unsupported text file type '{ext}'. Supported: {', '.join(sorted(TEXT_EXTENSIONS))}"
        )
