"""
media_processor.py
------------------
Document loader and text extractor for presentations and images:
- PowerPoint presentations: .pptx, .ppt (via python-pptx with structured Markdown tables,
  speaker notes, and automated OCR text extraction for embedded slide images)
- Images: .png, .jpg, .jpeg, .webp, .bmp, .tiff (via Pillow and Nemotron OCR text extraction)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator, List, Optional

try:
    from langchain_core.document_loaders import BaseLoader
except ImportError:
    from langchain_community.document_loaders.base import BaseLoader  # type: ignore[no-redef]
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PRESENTATION_EXTENSIONS = {".pptx", ".ppt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MEDIA_EXTENSIONS = PRESENTATION_EXTENSIONS | IMAGE_EXTENSIONS

def _iter_shapes(shapes: Any) -> Iterator[Any]:
    """Args: shapes: Shape collection."""
    for shape in shapes:
        if hasattr(shape, "shapes") and shape.shapes:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape

def _extract_image_blob(shape: Any) -> Optional[bytes]:
    """Args: shape: Presentation shape."""
    try:
        if hasattr(shape, "image") and getattr(shape, "image", None) is not None:
            return getattr(shape.image, "blob", None)
    except Exception:
        pass
    return None

def _extract_slide_shapes(slide: Any, engine: Optional[Any], slide_idx: int) -> tuple[str, list[str], list[str], list[str], int]:
    """Args: slide: Slide object, engine: Optional OCR engine, slide_idx: 1-indexed slide number."""
    slide_title = ""
    text_blocks, tables_md, image_ocr_blocks, image_count = [], [], [], 0

    if slide.shapes.title and getattr(slide.shapes.title, "has_text_frame", False):
        t_frame = getattr(slide.shapes.title, "text_frame", None)
        if t_frame:
            slide_title = t_frame.text.strip()

    for shape in _iter_shapes(slide.shapes):
        if shape == slide.shapes.title:
            continue
        if getattr(shape, "has_text_frame", False) and getattr(shape, "text_frame", None):
            shape_text = shape.text_frame.text.strip()
            if shape_text:
                text_blocks.append(shape_text)
        elif getattr(shape, "has_table", False) and getattr(shape, "table", None):
            tbl = shape.table
            if len(tbl.rows) > 0 and len(tbl.columns) > 0:
                rows = ["| " + " | ".join(c.text.replace("\n", " ").replace("|", "\\|").strip() for c in r.cells) + " |" for r in tbl.rows]
                sep = "| " + " | ".join(["---"] * len(tbl.columns)) + " |"
                tables_md.append("\n".join([rows[0], sep] + rows[1:]))

        img_blob = _extract_image_blob(shape)
        if img_blob:
            image_count += 1
            if engine:
                try:
                    ocr_text = engine.extract_text(img_blob)
                    if ocr_text and ocr_text.strip():
                        image_ocr_blocks.append(f"- **[Image {image_count} OCR]:** {ocr_text.strip()}")
                except Exception as err:
                    logger.debug("OCR failed on slide %d: %s", slide_idx, err)

    return slide_title, text_blocks, tables_md, image_ocr_blocks, image_count

class PowerPointLoader(BaseLoader):
    """Loader for PowerPoint presentations (.pptx, .ppt)."""

    def __init__(self, file_path: str, filename: Optional[str] = None, ocr_engine: Optional[Any] = None, extract_image_ocr: bool = True):
        """Args: file_path: File path, filename: File name, ocr_engine: OCR engine, extract_image_ocr: OCR flag."""
        self.file_path = file_path
        self.filename = filename or os.path.basename(file_path)
        self.ocr_engine = ocr_engine
        self.extract_image_ocr = extract_image_ocr

    def lazy_load(self) -> Iterator[Document]:
        """Args: None."""
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError("Install python-pptx: `pip install python-pptx`") from exc
        try:
            prs = Presentation(self.file_path)
        except Exception as exc:
            raise ValueError(f"Could not open PowerPoint file '{self.filename}': {exc}") from exc

        total_slides = len(prs.slides)
        if total_slides == 0:
            yield Document(
                page_content=f"# Presentation: {self.filename}\n*(Empty presentation with no slides)*",
                metadata={"source": self.filename, "filename": self.filename, "slide_number": 0, "total_slides": 0, "is_presentation": True},
            )
            return

        engine = None
        if self.extract_image_ocr:
            try:
                from setup.init_config import ocr_model
                engine = self.ocr_engine or ocr_model()
            except Exception as e:
                logger.debug("Could not init OCR model: %s", e)

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_title, text_blocks, tables_md, image_ocr_blocks, image_count = _extract_slide_shapes(slide, engine, slide_idx)
            speaker_notes = slide.notes_slide.notes_text_frame.text.strip() if (slide.has_notes_slide and slide.notes_slide.notes_text_frame) else ""

            header = f"# Presentation: {self.filename} (Slide {slide_idx}/{total_slides})\n" + (f"## Title: {slide_title}\n\n" if slide_title else "\n")
            parts = [header]
            if text_blocks:
                parts.append("\n".join(text_blocks))
            if tables_md:
                parts.append("\n\n### Tables:\n" + "\n\n".join(tables_md))
            if image_ocr_blocks:
                parts.append("\n\n### Embedded Image Text (OCR):\n" + "\n\n".join(image_ocr_blocks))
            if speaker_notes:
                parts.append(f"\n\n**Speaker Notes:**\n{speaker_notes}")
            if not text_blocks and not tables_md and not image_ocr_blocks and not speaker_notes and not slide_title:
                parts.append(f"*({image_count} embedded objects without readable OCR text)*" if image_count > 0 else "*(Empty slide)*")

            yield Document(
                page_content="\n".join(parts).strip(),
                metadata={
                    "source": f"{self.filename} (Slide {slide_idx}/{total_slides})",
                    "filename": self.filename,
                    "slide_number": slide_idx,
                    "total_slides": total_slides,
                    "slide_title": slide_title,
                    "has_notes": bool(speaker_notes),
                    "has_images": bool(image_count > 0),
                    "image_count": image_count,
                    "has_image_ocr": bool(image_ocr_blocks),
                    "is_presentation": True,
                },
            )

class ImageOCRLoader(BaseLoader):
    """Loader for Image files (.png, .jpg, etc.)."""

    def __init__(self, file_path: str, filename: Optional[str] = None, ocr_engine: Optional[Any] = None):
        """Args: file_path: File path, filename: File name, ocr_engine: OCR engine."""
        self.file_path = file_path
        self.filename = filename or os.path.basename(file_path)
        self.ocr_engine = ocr_engine

    def lazy_load(self) -> Iterator[Document]:
        """Args: None."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Install Pillow: `pip install pillow`") from exc
        from setup.init_config import ocr_model
        engine = self.ocr_engine or ocr_model()
        try:
            with Image.open(self.file_path) as img:
                width, height = img.size[0], img.size[1]
                fmt = img.format or os.path.splitext(self.filename)[1].lstrip(".").upper()
                mode = img.mode
                try:
                    ocr_text = engine.extract_text(img)
                except Exception as err:
                    logger.warning("OCR error for '%s': %s", self.filename, err)
                    ocr_text = ""
                body = f"### Extracted Text (OCR):\n{ocr_text}" if ocr_text else "*(No readable text detected via OCR in this image)*"
                yield Document(
                    page_content=f"# Image Document: {self.filename}\n**Resolution:** {width}x{height} | **Format:** {fmt} | **Color Mode:** {mode}\n\n{body}",
                    metadata={
                        "source": f"{self.filename} (Image)",
                        "filename": self.filename,
                        "width": width,
                        "height": height,
                        "format": fmt,
                        "is_image": True,
                        "ocr_engine": getattr(engine, "name", "nemotron-ocr-v2"),
                    },
                )
        except Exception as exc:
            raise ValueError(f"Could not load image file '{self.filename}': {exc}") from exc

def load_presentation_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None, extract_image_ocr: bool = True) -> List[Document]:
    """Args: file_path: File path, filename: File name, ocr_engine: Optional OCR engine, extract_image_ocr: OCR flag."""
    return list(PowerPointLoader(file_path=file_path, filename=filename, ocr_engine=ocr_engine, extract_image_ocr=extract_image_ocr).lazy_load())

def load_image_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None) -> List[Document]:
    """Args: file_path: File path, filename: File name, ocr_engine: Optional OCR engine."""
    return list(ImageOCRLoader(file_path=file_path, filename=filename, ocr_engine=ocr_engine).lazy_load())

def load_media_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None, extract_image_ocr: bool = True) -> List[Document]:
    """Args: file_path: File path, filename: File name, ocr_engine: Optional OCR engine, extract_image_ocr: OCR flag."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in PRESENTATION_EXTENSIONS:
        return load_presentation_document(file_path, filename, ocr_engine, extract_image_ocr)
    if ext in IMAGE_EXTENSIONS:
        return load_image_document(file_path, filename, ocr_engine)
    raise ValueError(f"Unsupported media file type '{ext}'. Supported: {', '.join(sorted(MEDIA_EXTENSIONS))}")
