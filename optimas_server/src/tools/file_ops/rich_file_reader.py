"""
Rich file reader — extract text from non-plaintext file formats.

Supported formats:
  .pdf   — text + optional image extraction  (pdfplumber)
  .docx  — paragraphs + tables               (python-docx)
  .pptx  — slides with shapes/tables          (python-pptx)
  .xlsx  — sheets → text tables               (openpyxl)
  .csv   — standard CSV                       (stdlib csv)
  .html / .htm — visible text                 (BeautifulSoup)

Every public helper returns ``(text, metadata)`` where *text* is a plain
UTF-8 string and *metadata* is a small dict with format-specific info.
All helpers are **sync** — the caller should run them in an executor when
needed to keep the event loop free.

Design principle: each extractor catches its own exceptions internally and
returns a clear error string instead of raising, so the tool layer never
crashes on a malformed file.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RICH_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".csv", ".html", ".htm",
}


def is_rich_format(path: Path) -> bool:
    """Return True if *path*'s suffix is a supported rich format."""
    return path.suffix.lower() in _RICH_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def read_pdf(
    path: Path,
    *,
    extract_images: bool = True,
    image_output_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Extract text (and optionally images) from a PDF file.

    Parameters
    ----------
    path : Path
        Absolute path to the PDF.
    extract_images : bool
        If True, attempt to save embedded images alongside the PDF.
    image_output_dir : Path | None
        Directory to write extracted images into.  Defaults to
        ``<pdf_parent>/<pdf_stem>_images/``.

    Returns
    -------
    (text, metadata)
        *text* contains one ``--- Page N ---`` header per page followed by the
        page text.  *metadata* includes ``pages``, ``images`` (list of saved
        paths), and ``format``.
    """
    try:
        import pdfplumber
    except ImportError:
        return (
            "[Error] pdfplumber is not installed. Cannot read PDF files.",
            {"format": "pdf", "error": "pdfplumber not installed"},
        )

    metadata: Dict = {"format": "pdf", "pages": 0, "images": []}
    sections: List[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            metadata["pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []

                parts: List[str] = [f"--- Page {i} ---"]
                if page_text.strip():
                    parts.append(page_text)

                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    parts.append(f"\n[Table {t_idx + 1} on page {i}]")
                    for row in table:
                        cells = [str(c) if c is not None else "" for c in row]
                        parts.append(" | ".join(cells))

                sections.append("\n".join(parts))

        if extract_images:
            saved = _extract_pdf_images(path, image_output_dir)
            metadata["images"] = saved
            if saved:
                sections.append(
                    "\n--- Extracted images ---\n"
                    + "\n".join(
                        f"  [{i+1}] {p}" for i, p in enumerate(saved)
                    )
                )

    except Exception as exc:
        logger.warning("PDF read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read PDF: {exc}",
            {"format": "pdf", "error": str(exc)},
        )

    text = "\n\n".join(sections) if sections else "(empty PDF)"
    return text, metadata


def _extract_pdf_images(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
) -> List[str]:
    """Best-effort image extraction from a PDF.  Returns list of saved paths."""
    try:
        import pdfplumber
    except ImportError:
        return []

    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_images"

    saved: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                images = page.images or []
                for img_idx, img_meta in enumerate(images):
                    try:
                        img_obj = page.crop(
                            (
                                img_meta["x0"],
                                img_meta["top"],
                                img_meta["x1"],
                                img_meta["bottom"],
                            )
                        )
                        pil_img = img_obj.to_image(resolution=150).original
                        output_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"page{page_idx + 1}_img{img_idx + 1}.png"
                        out_path = output_dir / fname
                        pil_img.save(str(out_path), format="PNG")
                        saved.append(str(out_path))
                    except Exception:
                        pass
    except Exception:
        pass

    return saved


# ---------------------------------------------------------------------------
# DOCX (Word)
# ---------------------------------------------------------------------------

def read_docx(path: Path) -> Tuple[str, Dict]:
    try:
        from docx import Document
    except ImportError:
        return (
            "[Error] python-docx is not installed. Cannot read DOCX files.",
            {"format": "docx", "error": "python-docx not installed"},
        )

    metadata: Dict = {"format": "docx", "paragraphs": 0, "tables": 0}
    parts: List[str] = []

    try:
        doc = Document(str(path))
        metadata["paragraphs"] = len(doc.paragraphs)
        metadata["tables"] = len(doc.tables)

        for para in doc.paragraphs:
            text = para.text
            if text.strip():
                parts.append(text)

        for t_idx, table in enumerate(doc.tables):
            rows_text: List[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows_text.append(" | ".join(cells))
            if rows_text:
                parts.append(f"\n[Table {t_idx + 1}]\n" + "\n".join(rows_text))

    except Exception as exc:
        logger.warning("DOCX read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read DOCX: {exc}",
            {"format": "docx", "error": str(exc)},
        )

    text = "\n".join(parts) if parts else "(empty document)"
    return text, metadata


# ---------------------------------------------------------------------------
# PPTX (PowerPoint)
# ---------------------------------------------------------------------------

def read_pptx(path: Path) -> Tuple[str, Dict]:
    try:
        from pptx import Presentation
    except ImportError:
        return (
            "[Error] python-pptx is not installed. Cannot read PPTX files.",
            {"format": "pptx", "error": "python-pptx not installed"},
        )

    metadata: Dict = {"format": "pptx", "slides": 0}
    sections: List[str] = []

    try:
        prs = Presentation(str(path))
        metadata["slides"] = len(prs.slides)

        for slide_idx, slide in enumerate(prs.slides, start=1):
            parts: List[str] = [f"--- Slide {slide_idx} ---"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            parts.append(text)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        parts.append(" | ".join(cells))
            sections.append("\n".join(parts))

    except Exception as exc:
        logger.warning("PPTX read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read PPTX: {exc}",
            {"format": "pptx", "error": str(exc)},
        )

    text = "\n\n".join(sections) if sections else "(empty presentation)"
    return text, metadata


# ---------------------------------------------------------------------------
# XLSX / XLS (Excel via openpyxl)
# ---------------------------------------------------------------------------

def read_xlsx(path: Path) -> Tuple[str, Dict]:
    try:
        import openpyxl
    except ImportError:
        return (
            "[Error] openpyxl is not installed. Cannot read XLSX files.",
            {"format": "xlsx", "error": "openpyxl not installed"},
        )

    metadata: Dict = {"format": "xlsx", "sheets": 0}
    sections: List[str] = []

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        metadata["sheets"] = len(wb.sheetnames)

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: List[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(
                    f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows)
                )

        wb.close()
    except Exception as exc:
        logger.warning("XLSX read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read XLSX: {exc}",
            {"format": "xlsx", "error": str(exc)},
        )

    text = "\n\n".join(sections) if sections else "(empty spreadsheet)"
    return text, metadata


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> Tuple[str, Dict]:
    metadata: Dict = {"format": "csv", "rows": 0}
    rows_text: List[str] = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sniffer = csv.Sniffer()
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = sniffer.sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            for row in reader:
                rows_text.append(" | ".join(row))

        metadata["rows"] = len(rows_text)
    except Exception as exc:
        logger.warning("CSV read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read CSV: {exc}",
            {"format": "csv", "error": str(exc)},
        )

    text = "\n".join(rows_text) if rows_text else "(empty CSV)"
    return text, metadata


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def read_html(path: Path) -> Tuple[str, Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return (
            "[Error] beautifulsoup4 is not installed. Cannot read HTML files.",
            {"format": "html", "error": "beautifulsoup4 not installed"},
        )

    metadata: Dict = {"format": "html"}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        soup = BeautifulSoup(raw, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
    except Exception as exc:
        logger.warning("HTML read failed for %s: %s", path, exc, exc_info=True)
        return (
            f"[Error] Failed to read HTML: {exc}",
            {"format": "html", "error": str(exc)},
        )

    return text or "(empty HTML)", metadata


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXT_DISPATCH = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".pptx": read_pptx,
    ".xlsx": read_xlsx,
    ".xls": read_xlsx,
    ".csv": read_csv,
    ".html": read_html,
    ".htm": read_html,
}


def read_rich_file(
    path: Path,
    *,
    extract_images: bool = True,
    image_output_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Dispatch to the right extractor based on file extension.

    Returns ``(text, metadata)``.  Never raises — errors are embedded in *text*.
    """
    ext = path.suffix.lower()
    handler = _EXT_DISPATCH.get(ext)
    if handler is None:
        return (
            f"[Error] Unsupported rich file format: {ext}",
            {"format": ext, "error": "unsupported"},
        )

    if ext == ".pdf":
        return read_pdf(path, extract_images=extract_images, image_output_dir=image_output_dir)
    return handler(path)
