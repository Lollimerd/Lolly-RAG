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

# Supported presentation and image extensions
PRESENTATION_EXTENSIONS = {".pptx", ".ppt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MEDIA_EXTENSIONS = PRESENTATION_EXTENSIONS | IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# PowerPoint Presentation Loader (with Embedded Image OCR)
# ---------------------------------------------------------------------------

def _iter_shapes(shapes: Any) -> Iterator[Any]:
    """Recursively iterate over shapes, flattening any group shapes."""
    for shape in shapes:
        if hasattr(shape, "shapes") and shape.shapes:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _extract_image_blob(shape: Any) -> Optional[bytes]:
    """Safely extract raw image bytes from a shape if it contains an image."""
    try:
        if hasattr(shape, "image") and getattr(shape, "image", None) is not None:
            return getattr(shape.image, "blob", None)
    except Exception:
        pass
    return None


class PowerPointLoader(BaseLoader):
    """
    LangChain Document Loader for PowerPoint presentations (.pptx, .ppt).

    Extracts slide-level structured documents containing:
    - Slide titles and headings
    - Bullet points and body text
    - Tables formatted into clean Markdown tables
    - Speaker notes
    - OCR text extracted from embedded slide images and diagrams
    """

    def __init__(
        self,
        file_path: str,
        filename: Optional[str] = None,
        ocr_engine: Optional[Any] = None,
        extract_image_ocr: bool = True,
    ):
        self.file_path = file_path
        self.filename = filename or os.path.basename(file_path)
        self.ocr_engine = ocr_engine
        self.extract_image_ocr = extract_image_ocr

    def lazy_load(self) -> Iterator[Document]:
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError(
                "Could not import python-pptx. Please install it with `pip install python-pptx`."
            ) from exc

        try:
            prs = Presentation(self.file_path)
        except Exception as exc:
            raise ValueError(f"Could not open PowerPoint file '{self.filename}': {exc}") from exc

        total_slides = len(prs.slides)
        if total_slides == 0:
            yield Document(
                page_content=f"# Presentation: {self.filename}\n*(Empty presentation with no slides)*",
                metadata={
                    "source": self.filename,
                    "filename": self.filename,
                    "slide_number": 0,
                    "total_slides": 0,
                    "is_presentation": True,
                },
            )
            return

        # Lazily initialize OCR engine if image OCR is enabled
        engine = None
        if self.extract_image_ocr:
            try:
                from setup.init_config import ocr_model
                engine = self.ocr_engine or ocr_model()
            except Exception as ocr_init_err:
                logger.debug("Could not initialize OCR model for PPT embedded images: %s", ocr_init_err)
                engine = None

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_title = ""
            text_blocks: List[str] = []
            tables_md: List[str] = []
            image_ocr_blocks: List[str] = []
            image_count = 0

            # 1. Slide Title (if defined as title shape)
            if slide.shapes.title and getattr(slide.shapes.title, "has_text_frame", False):
                t_frame = getattr(slide.shapes.title, "text_frame", None)
                if t_frame:
                    slide_title = t_frame.text.strip()

            # Iterate over all shapes (including nested group shapes)
            for shape in _iter_shapes(slide.shapes):
                # Skip title shape already extracted
                if shape == slide.shapes.title:
                    continue

                has_text = getattr(shape, "has_text_frame", False)
                text_frame = getattr(shape, "text_frame", None)
                has_tbl = getattr(shape, "has_table", False)
                tbl = getattr(shape, "table", None)

                # 2. Text Frame (paragraphs, subtitles, bullet points)
                if has_text and text_frame:
                    shape_text = text_frame.text.strip()
                    if shape_text:
                        text_blocks.append(shape_text)

                # 3. Tables in slide
                elif has_tbl and tbl:
                    col_count = len(tbl.columns)
                    row_count = len(tbl.rows)
                    if row_count > 0 and col_count > 0:
                        tbl_rows = []
                        for row in tbl.rows:
                            row_cells = [
                                cell.text.replace("\n", " ").replace("|", "\\|").strip()
                                for cell in row.cells
                            ]
                            tbl_rows.append("| " + " | ".join(row_cells) + " |")

                        if tbl_rows:
                            sep = "| " + " | ".join(["---"] * col_count) + " |"
                            table_md = "\n".join([tbl_rows[0], sep] + tbl_rows[1:])
                            tables_md.append(table_md)

                # 4. Embedded Image / Picture OCR
                img_blob = _extract_image_blob(shape)
                if img_blob:
                    image_count += 1
                    if engine is not None:
                        try:
                            ocr_text = engine.extract_text(img_blob)
                            if ocr_text and ocr_text.strip():
                                image_ocr_blocks.append(f"- **[Image {image_count} OCR]:** {ocr_text.strip()}")
                        except Exception as img_ocr_err:
                            logger.debug("Failed OCR for slide %d image %d in '%s': %s", slide_idx, image_count, self.filename, img_ocr_err)

            # 5. Speaker Notes
            speaker_notes = ""
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    speaker_notes = notes_text

            # Construct structured slide Markdown representation
            header = f"# Presentation: {self.filename} (Slide {slide_idx}/{total_slides})\n"
            if slide_title:
                header += f"## Title: {slide_title}\n\n"
            else:
                header += "\n"

            content_parts = [header]
            if text_blocks:
                content_parts.append("\n".join(text_blocks))
            if tables_md:
                content_parts.append("\n\n### Tables:\n" + "\n\n".join(tables_md))
            if image_ocr_blocks:
                content_parts.append("\n\n### Embedded Image Text (OCR):\n" + "\n\n".join(image_ocr_blocks))
            if speaker_notes:
                content_parts.append(f"\n\n**Speaker Notes:**\n{speaker_notes}")

            if not text_blocks and not tables_md and not image_ocr_blocks and not speaker_notes and not slide_title:
                if image_count > 0:
                    content_parts.append(f"*({image_count} embedded image/graphic objects with no readable OCR text)*")
                else:
                    content_parts.append("*(Empty slide or graphic objects only)*")

            page_content = "\n".join(content_parts).strip()
            source_label = f"{self.filename} (Slide {slide_idx}/{total_slides})"

            yield Document(
                page_content=page_content,
                metadata={
                    "source": source_label,
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


# ---------------------------------------------------------------------------
# Image OCR Document Loader
# ---------------------------------------------------------------------------

class ImageOCRLoader(BaseLoader):
    """
    LangChain Document Loader for Image files (.png, .jpg, .jpeg, .webp, .bmp, .tiff).
    Extracts text content using the configured OCR model and captures image metadata.
    """

    def __init__(self, file_path: str, filename: Optional[str] = None, ocr_engine: Optional[Any] = None):
        self.file_path = file_path
        self.filename = filename or os.path.basename(file_path)
        self.ocr_engine = ocr_engine

    def lazy_load(self) -> Iterator[Document]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Could not import Pillow. Please install it with `pip install pillow`."
            ) from exc

        from setup.init_config import ocr_model

        engine = self.ocr_engine or ocr_model()

        try:
            with Image.open(self.file_path) as img:
                width, height = img.size
                img_format = img.format or os.path.splitext(self.filename)[1].lstrip(".").upper()
                mode = img.mode

                try:
                    ocr_text = engine.extract_text(img)
                except Exception as ocr_err:
                    logger.warning("OCR processing error for image '%s': %s", self.filename, ocr_err)
                    ocr_text = ""

                header = (
                    f"# Image Document: {self.filename}\n"
                    f"**Resolution:** {width}x{height} | **Format:** {img_format} | **Color Mode:** {mode}\n\n"
                )
                if ocr_text:
                    body = f"### Extracted Text (OCR):\n{ocr_text}"
                else:
                    body = "*(No readable text detected via OCR in this image)*"

                page_content = f"{header}{body}"
                source_label = f"{self.filename} (Image)"

                yield Document(
                    page_content=page_content,
                    metadata={
                        "source": source_label,
                        "filename": self.filename,
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "is_image": True,
                        "ocr_engine": getattr(engine, "name", "nemotron-ocr-v2"),
                    },
                )
        except Exception as exc:
            raise ValueError(f"Could not load image file '{self.filename}': {exc}") from exc


# ---------------------------------------------------------------------------
# High-Level Presentation & Image Dispatchers
# ---------------------------------------------------------------------------

def load_presentation_document(
    file_path: str,
    filename: str,
    ocr_engine: Optional[Any] = None,
    extract_image_ocr: bool = True,
) -> List[Document]:
    """Load a presentation file (.pptx, .ppt) into LangChain Documents with optional image OCR."""
    return list(
        PowerPointLoader(
            file_path=file_path,
            filename=filename,
            ocr_engine=ocr_engine,
            extract_image_ocr=extract_image_ocr,
        ).lazy_load()
    )


def load_image_document(file_path: str, filename: str, ocr_engine: Optional[Any] = None) -> List[Document]:
    """Load an image file (.png, .jpg, etc.) with OCR extraction into LangChain Documents."""
    return list(ImageOCRLoader(file_path=file_path, filename=filename, ocr_engine=ocr_engine).lazy_load())


def load_media_document(
    file_path: str,
    filename: str,
    ocr_engine: Optional[Any] = None,
    extract_image_ocr: bool = True,
) -> List[Document]:
    """
    Load a presentation or image file into LangChain Documents.

    Args:
        file_path: Filesystem path to the file.
        filename: Original filename.
        ocr_engine: Optional OCR engine instance for image/presentation files.
        extract_image_ocr: Whether to run OCR on images embedded inside presentations.

    Returns:
        List of Document objects extracted from the presentation or image.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in PRESENTATION_EXTENSIONS:
        return load_presentation_document(
            file_path=file_path,
            filename=filename,
            ocr_engine=ocr_engine,
            extract_image_ocr=extract_image_ocr,
        )
    elif ext in IMAGE_EXTENSIONS:
        return load_image_document(file_path=file_path, filename=filename, ocr_engine=ocr_engine)
    else:
        raise ValueError(
            f"Unsupported media file type '{ext}'. Supported: {', '.join(sorted(MEDIA_EXTENSIONS))}"
        )
