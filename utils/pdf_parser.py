"""
pdf_parser.py
─────────────
Extract plain text from PDF files using PyMuPDF (fitz).
Handles:
  - Normal text PDFs
  - Multi-column PDFs (best-effort)
  - Image-only / scanned PDFs (detected and reported)
  - Corrupted / empty PDFs
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

try:
    import pymupdf as fitz  # PyMuPDF ≥ 1.24 prefers the `pymupdf` namespace
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz  # fallback for older PyMuPDF versions
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class PDFParseResult:
    text: str = ""
    page_count: int = 0
    is_image_only: bool = False
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and not self.is_image_only and bool(self.text.strip())


def extract_text_from_pdf(file_bytes: bytes) -> PDFParseResult:
    """
    Extract readable text from a PDF file supplied as raw bytes.

    Returns a PDFParseResult with:
        .text          – extracted plain text (empty string on failure)
        .page_count    – number of pages
        .is_image_only – True when every page is an embedded image with no selectable text
        .warnings      – non-fatal issues
        .error         – fatal error message (None on success)
    """
    result = PDFParseResult()

    # ── Dependency check ───────────────────────────────────────────────────
    if not FITZ_AVAILABLE:
        result.error = (
            "PyMuPDF is not installed. Run: pip install pymupdf"
        )
        return result

    # ── Size guard ─────────────────────────────────────────────────────────
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        result.error = (
            f"File is too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
        )
        return result

    if len(file_bytes) == 0:
        result.error = "The uploaded PDF file is empty (0 bytes)."
        return result

    # ── Open document ──────────────────────────────────────────────────────
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        result.error = f"Could not open PDF: {exc}. The file may be corrupted."
        return result

    result.page_count = len(doc)

    if result.page_count == 0:
        result.error = "PDF has no pages."
        doc.close()
        return result

    # ── Extract text page by page ──────────────────────────────────────────
    pages_with_text = 0
    pages_image_only = 0
    all_text: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        # get_text("text") returns plain UTF-8 text, preserving line breaks
        page_text = page.get_text("text")

        if page_text.strip():
            pages_with_text += 1
            all_text.append(f"--- Page {page_num} ---\n{page_text}")
        else:
            # Check if there are images on this page (possible scan)
            images = page.get_images(full=True)
            if images:
                pages_image_only += 1
            else:
                result.warnings.append(
                    f"Page {page_num} has no text and no images (blank or unsupported content)."
                )

    doc.close()

    # ── Classify result ────────────────────────────────────────────────────
    if pages_image_only > 0 and pages_with_text == 0:
        result.is_image_only = True
        result.error = (
            "This PDF appears to be a scanned document (image-only). "
            "No selectable text could be extracted. "
            "Please use an OCR tool (e.g. Adobe Acrobat, Tesseract) to convert it to "
            "a text-based PDF before uploading, or paste the resume text manually."
        )
        return result

    if pages_image_only > 0:
        result.warnings.append(
            f"{pages_image_only} page(s) are image-only and were skipped. "
            "Some content may be missing from the analysis."
        )

    result.text = "\n\n".join(all_text)

    if not result.text.strip():
        result.error = "No readable text could be extracted from the PDF."

    return result
