# designer/services/pdf_service.py

import os
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PDFService:
    """Service for extracting text and metadata from PDF files."""

    @staticmethod
    def get_page_count(file_path: str) -> int:
        """Get the number of pages in a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception as e:
            logger.error(f"Error getting page count for {file_path}: {e}")
            return 0

    @staticmethod
    def extract_text_from_pages(file_path: str, pages: Optional[List[int]] = None) -> Dict[int, str]:
        """
        Extract text from specified pages of a PDF.
        
        Args:
            file_path: Path to the PDF file
            pages: List of 1-based page numbers. If None, extract all pages.
            
        Returns:
            Dictionary mapping page numbers (1-based) to extracted text.
        """
        result = {}
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)

            if pages is None:
                pages = list(range(1, total_pages + 1))

            for page_num in pages:
                if 1 <= page_num <= total_pages:
                    try:
                        page = reader.pages[page_num - 1]  # 0-based index
                        text = page.extract_text() or ''
                        text = text.strip()
                        if text:
                            result[page_num] = text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")

        return result

    @staticmethod
    def extract_all_text(file_path: str) -> str:
        """Extract all text from a PDF file, concatenated."""
        pages = PDFService.extract_text_from_pages(file_path)
        return '\n\n'.join(f"[Page {num}]\n{text}" for num, text in sorted(pages.items()))

    @staticmethod
    def get_pages_for_range(file_path: str, start: int, end: int) -> List[int]:
        """Get a list of page numbers within a range."""
        total = PDFService.get_page_count(file_path)
        start = max(1, start)
        end = min(total, end)
        return list(range(start, end + 1))

    @staticmethod
    def get_random_pages(file_path: str, count: int) -> List[int]:
        """Get random page numbers from a PDF."""
        import random
        total = PDFService.get_page_count(file_path)
        if total == 0:
            return []
        count = min(count, total)
        return sorted(random.sample(range(1, total + 1), count))

    @staticmethod
    def parse_specific_pages(pages_str: str) -> List[int]:
        """Parse a comma-separated string of page numbers."""
        pages = []
        for part in pages_str.split(','):
            part = part.strip()
            if part.isdigit():
                pages.append(int(part))
        return sorted(set(pages))

    @staticmethod
    def is_scanned_pdf(file_path: str, sample_pages: int = 3) -> bool:
        """Check if a PDF appears to be scanned (no extractable text)."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            check_pages = min(sample_pages, total_pages)

            text_found = 0
            for i in range(check_pages):
                text = reader.pages[i].extract_text() or ''
                if len(text.strip()) > 50:
                    text_found += 1

            return text_found == 0
        except Exception:
            return True