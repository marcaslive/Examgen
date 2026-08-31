# designer/services/pdf_service.py

import logging
import os
import random
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Soft cap so one huge page cannot dominate Gemini context alone.
# Full-document capping still happens in question_generator (GEMINI_INPUT_CHARS).
MAX_CHARS_PER_PAGE = 12000


def _try_import_pypdf2():
    try:
        from PyPDF2 import PdfReader
        return PdfReader
    except ImportError:
        logger.error("PyPDF2 is not installed. Run: pip install PyPDF2")
        return None


class PDFService:
    """Service for extracting text and metadata from PDF files."""

    @staticmethod
    def _validate_path(file_path: str) -> bool:
        if not file_path or not isinstance(file_path, str):
            logger.error("PDF path is empty or invalid")
            return False
        if not os.path.isfile(file_path):
            logger.error("PDF file not found: %s", file_path)
            return False
        return True

    @staticmethod
    def _open_reader(file_path: str):
        """Open a PdfReader, handling missing file / encryption / corrupt PDF."""
        if not PDFService._validate_path(file_path):
            return None

        PdfReader = _try_import_pypdf2()
        if PdfReader is None:
            return None

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            logger.error("Error opening PDF %s: %s", file_path, e)
            return None

        try:
            if getattr(reader, "is_encrypted", False):
                # Try empty password (common for "open" encrypted PDFs)
                try:
                    ok = reader.decrypt("")
                    if not ok:
                        logger.warning("PDF is encrypted and cannot be decrypted: %s", file_path)
                        return None
                except Exception as e:
                    logger.warning("PDF decrypt failed for %s: %s", file_path, e)
                    return None
        except Exception:
            pass

        return reader

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize extracted PDF text."""
        if not text:
            return ""
        # Replace nulls / form-feed / weird separators common in PDFs
        text = text.replace("\x00", " ")
        text = text.replace("\x0c", "\n")  # form feed
        # Collapse excessive blank lines / spaces but keep paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def get_page_count(file_path: str) -> int:
        """Get the number of pages in a PDF file."""
        reader = PDFService._open_reader(file_path)
        if reader is None:
            return 0
        try:
            return len(reader.pages)
        except Exception as e:
            logger.error("Error getting page count for %s: %s", file_path, e)
            return 0

    @staticmethod
    def extract_text_from_pages(
        file_path: str,
        pages: Optional[List[int]] = None,
        max_chars_per_page: int = MAX_CHARS_PER_PAGE,
    ) -> Dict[int, str]:
        """
        Extract text from specified pages of a PDF.

        Args:
            file_path: Path to the PDF file
            pages: List of 1-based page numbers. If None, extract all pages.
            max_chars_per_page: Truncate each page to this many chars (0 = no limit)

        Returns:
            Dictionary mapping page numbers (1-based) to extracted text.
            Pages with no usable text are omitted.
        """
        result: Dict[int, str] = {}
        reader = PDFService._open_reader(file_path)
        if reader is None:
            return result

        try:
            total_pages = len(reader.pages)
            if total_pages == 0:
                logger.warning("PDF has 0 pages: %s", file_path)
                return result

            if pages is None:
                page_list = list(range(1, total_pages + 1))
            else:
                # Unique, valid, sorted 1-based pages
                page_list = sorted({
                    int(p) for p in pages
                    if isinstance(p, (int, float, str))
                    and str(p).strip().lstrip("-").isdigit()
                    and 1 <= int(p) <= total_pages
                })

            for page_num in page_list:
                try:
                    page = reader.pages[page_num - 1]  # 0-based index
                    raw = page.extract_text() or ""
                    text = PDFService._clean_text(raw)
                    if not text:
                        continue
                    if max_chars_per_page and len(text) > max_chars_per_page:
                        text = text[:max_chars_per_page].rsplit(" ", 1)[0] + "…"
                    result[page_num] = text
                except Exception as e:
                    logger.warning(
                        "Error extracting page %s from %s: %s", page_num, file_path, e
                    )

        except Exception as e:
            logger.error("Error reading PDF %s: %s", file_path, e)

        return result

    @staticmethod
    def extract_all_text(file_path: str) -> str:
        """Extract all text from a PDF file, concatenated with page markers."""
        pages = PDFService.extract_text_from_pages(file_path)
        if not pages:
            return ""
        return "\n\n".join(
            f"[Page {num}]\n{text}" for num, text in sorted(pages.items())
        )

    @staticmethod
    def get_pages_for_range(file_path: str, start: int, end: int) -> List[int]:
        """
        Get a list of page numbers within an inclusive range.
        Swapped start/end are normalized. Out-of-bounds values are clamped.
        """
        total = PDFService.get_page_count(file_path)
        if total <= 0:
            return []

        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return []

        # Allow user to put higher number in "from"
        if start > end:
            start, end = end, start

        start = max(1, start)
        end = min(total, end)

        if start > end:
            return []

        return list(range(start, end + 1))

    @staticmethod
    def get_random_pages(file_path: str, count: int) -> List[int]:
        """Get random page numbers from a PDF (sorted ascending)."""
        total = PDFService.get_page_count(file_path)
        if total <= 0:
            return []

        try:
            count = int(count)
        except (TypeError, ValueError):
            return []

        if count <= 0:
            return []

        count = min(count, total)
        return sorted(random.sample(range(1, total + 1), count))

    @staticmethod
    def parse_specific_pages(pages_str: str, max_page: Optional[int] = None) -> List[int]:
        """
        Parse page numbers from a user string.

        Supports:
          - "3, 7, 12"
          - "5-10"
          - mixed: "1, 3-5, 8, 12-15"

        If max_page is given, pages above it are dropped.
        """
        if not pages_str or not str(pages_str).strip():
            return []

        pages = set()
        # Split on comma or whitespace
        tokens = re.split(r"[,\s]+", str(pages_str).strip())

        for token in tokens:
            if not token:
                continue
            token = token.strip()

            # Range: 5-10 or 5–10 (en-dash)
            m = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", token)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    a, b = b, a
                for n in range(a, b + 1):
                    if n >= 1 and (max_page is None or n <= max_page):
                        pages.add(n)
                continue

            # Single page
            if token.isdigit():
                n = int(token)
                if n >= 1 and (max_page is None or n <= max_page):
                    pages.add(n)

        return sorted(pages)

    @staticmethod
    def clamp_pages(pages: List[int], file_path: str) -> List[int]:
        """Drop page numbers that are outside 1..total for this PDF."""
        total = PDFService.get_page_count(file_path)
        if total <= 0 or not pages:
            return []
        return sorted({int(p) for p in pages if 1 <= int(p) <= total})

    @staticmethod
    def is_scanned_pdf(file_path: str, sample_pages: int = 3, min_chars: int = 50) -> bool:
        """
        Heuristic: True if sampled pages have almost no extractable text
        (typical of image-only / scanned PDFs).

        Returns True also if the file cannot be read (safer default for callers
        that want to warn the user).
        """
        reader = PDFService._open_reader(file_path)
        if reader is None:
            return True

        try:
            total_pages = len(reader.pages)
            if total_pages == 0:
                return True

            check_pages = min(max(1, sample_pages), total_pages)
            text_found = 0

            for i in range(check_pages):
                try:
                    raw = reader.pages[i].extract_text() or ""
                    text = PDFService._clean_text(raw)
                    if len(text) > min_chars:
                        text_found += 1
                except Exception as e:
                    logger.debug("is_scanned_pdf page %s error: %s", i + 1, e)

            # If none of the sampled pages had real text → likely scanned
            return text_found == 0
        except Exception as e:
            logger.warning("is_scanned_pdf failed for %s: %s", file_path, e)
            return True

    @staticmethod
    def get_text_stats(file_path: str, pages: Optional[List[int]] = None) -> dict:
        """
        Lightweight stats for UI / capacity checks without building huge strings twice.
        """
        extracted = PDFService.extract_text_from_pages(file_path, pages)
        char_count = sum(len(t) for t in extracted.values())
        word_count = sum(len(re.findall(r"[A-Za-z0-9]{2,}", t)) for t in extracted.values())
        return {
            "pages_with_text": len(extracted),
            "page_numbers": sorted(extracted.keys()),
            "char_count": char_count,
            "word_count": word_count,
            "is_empty": len(extracted) == 0,
        }