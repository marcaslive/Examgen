# designer/services/document_converter.py

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentConverter:
    """
    Service for converting non-PDF documents to PDF format.
    
    Supported formats:
    - TXT: Converted using reportlab (if available) or stored as-is
    - DOCX: Converted using python-docx for text extraction
    
    For full conversion support (DOC, PPTX, PPT, ODT), 
    LibreOffice must be installed on the server:
        sudo apt install libreoffice  (Linux)
        choco install libreoffice     (Windows)
    
    Set LIBREOFFICE_PATH in settings if not on system PATH.
    """

    SUPPORTED_EXTENSIONS = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'odt', 'txt']

    @staticmethod
    def is_pdf(filename: str) -> bool:
        return filename.lower().endswith('.pdf')

    @staticmethod
    def get_extension(filename: str) -> str:
        return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    @staticmethod
    def is_supported(filename: str) -> bool:
        ext = DocumentConverter.get_extension(filename)
        return ext in DocumentConverter.SUPPORTED_EXTENSIONS

    @staticmethod
    def convert_to_pdf(input_path: str, original_filename: str) -> Optional[str]:
        """
        Convert a document to PDF format.
        
        Returns the path to the converted PDF, or None if conversion failed.
        If the file is already a PDF, returns the input path.
        """
        ext = DocumentConverter.get_extension(original_filename)

        if ext == 'pdf':
            return input_path

        if ext == 'txt':
            return DocumentConverter._convert_txt_to_pdf(input_path)

        if ext == 'docx':
            return DocumentConverter._convert_docx_to_pdf(input_path)

        # For doc, pptx, ppt, odt - try LibreOffice
        return DocumentConverter._convert_with_libreoffice(input_path)

    @staticmethod
    def _convert_txt_to_pdf(input_path: str) -> Optional[str]:
        """Convert a text file to PDF."""
        try:
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Try using reportlab
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet

                output_path = input_path.rsplit('.', 1)[0] + '.pdf'
                doc = SimpleDocTemplate(output_path, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                for line in content.split('\n'):
                    if line.strip():
                        story.append(Paragraph(line, styles['Normal']))
                    else:
                        story.append(Spacer(1, 12))

                if story:
                    doc.build(story)
                    return output_path
            except ImportError:
                pass

            # Fallback: create a minimal PDF manually
            output_path = input_path.rsplit('.', 1)[0] + '.pdf'
            lines = content.split('\n')
            pdf_content = DocumentConverter._create_simple_pdf(lines)
            with open(output_path, 'wb') as f:
                f.write(pdf_content)
            return output_path

        except Exception as e:
            logger.error(f"Error converting TXT to PDF: {e}")
            return None

    @staticmethod
    def _convert_docx_to_pdf(input_path: str) -> Optional[str]:
        """Convert a DOCX file to PDF using python-docx for text extraction + reportlab."""
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument(input_path)
            text_lines = []
            for para in doc.paragraphs:
                text_lines.append(para.text)

            # Try reportlab
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet

                output_path = input_path.rsplit('.', 1)[0] + '.pdf'
                pdf_doc = SimpleDocTemplate(output_path, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                for line in text_lines:
                    if line.strip():
                        # Escape XML special characters
                        safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(safe_line, styles['Normal']))
                    else:
                        story.append(Spacer(1, 12))

                if not story:
                    story.append(Paragraph("(Empty document)", styles['Normal']))

                pdf_doc.build(story)
                return output_path

            except ImportError:
                # Without reportlab, save as text content alongside
                output_path = input_path.rsplit('.', 1)[0] + '.pdf'
                content = '\n'.join(text_lines)
                pdf_content = DocumentConverter._create_simple_pdf(content.split('\n'))
                with open(output_path, 'wb') as f:
                    f.write(pdf_content)
                return output_path

        except Exception as e:
            logger.error(f"Error converting DOCX to PDF: {e}")
            return None

    @staticmethod
    def _convert_with_libreoffice(input_path: str) -> Optional[str]:
        """Convert using LibreOffice command line."""
        try:
            import subprocess
            output_dir = os.path.dirname(input_path)

            result = subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_path],
                capture_output=True, text=True, timeout=120
            )

            if result.returncode == 0:
                expected_output = input_path.rsplit('.', 1)[0] + '.pdf'
                if os.path.exists(expected_output):
                    return expected_output

            logger.error(f"LibreOffice conversion failed: {result.stderr}")
            return None

        except FileNotFoundError:
            logger.error("LibreOffice is not installed. Install it for DOC/PPTX/PPT/ODT conversion.")
            return None
        except Exception as e:
            logger.error(f"Error converting with LibreOffice: {e}")
            return None

    @staticmethod
    def _create_simple_pdf(lines: list) -> bytes:
        """Create a very basic PDF from text lines without external libraries."""
        # Minimal valid PDF
        objects = []
        # Catalog
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        # Pages
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        # Page
        objects.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       b"/Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>\nendobj\n")
        # Font
        objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        # Content stream
        y = 750
        content_lines = []
        content_lines.append(b"BT\n/F1 10 Tf\n")
        for line in lines[:70]:  # Limit to ~1 page
            safe = line.encode('latin-1', errors='replace')
            safe = safe.replace(b'(', b'\\(').replace(b')', b'\\)')
            content_lines.append(f"1 0 0 1 50 {y} Tm\n".encode())
            content_lines.append(b"(" + safe + b") Tj\n")
            y -= 12
            if y < 50:
                break
        content_lines.append(b"ET\n")
        stream = b"".join(content_lines)
        objects.append(f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode() +
                       stream + b"\nendstream\nendobj\n")

        # Build PDF
        pdf = b"%PDF-1.4\n"
        offsets = []
        for obj in objects:
            offsets.append(len(pdf))
            pdf += obj

        xref_offset = len(pdf)
        pdf += f"xref\n0 {len(objects) + 1}\n".encode()
        pdf += b"0000000000 65535 f \n"
        for offset in offsets:
            pdf += f"{offset:010d} 00000 n \n".encode()

        pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        pdf += f"startxref\n{xref_offset}\n%%EOF\n".encode()

        return pdf