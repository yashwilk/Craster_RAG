"""
Each PDF page becomes one Document object.
Uses pdfplumber for extraction:
    better than pypdf for text extraction
    handles tables and columns better
    preserves more structure

"""

import logging
import re
from pathlib import Path
from typing import List

import pdfplumber

from craster_rag.ingestion.base_loader import BaseLoader, Document
from config import DOCUMENT_CATEGORIES

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):

    SUPPORTED_EXTENSION = ".pdf"

    def __init__(self, clean_text: bool = True):
        self.clean_text = clean_text


    def validate(self, source: str) -> bool:

        path = Path(source)

        if not path.exists():
            logger.error(f"Path does not exist: '{source}'")
            return False

        if path.is_file():
            if path.suffix.lower() != self.SUPPORTED_EXTENSION:
                logger.error(f"Not a PDF file: '{source}'")
                return False
            return True

        if path.is_dir():
            pdfs = list(path.glob("*.pdf"))
            if not pdfs:
                logger.error(f"No PDF files found in directory: '{source}'")
                return False
            return True

        return False


    def _build_title(self, filename: str) -> str:
        """Build a clean human readable title from filename.

        Examples:
            "Maternity Policy.pdf" -> "Maternity Policy"
        """
        # remove extension
        title = Path(filename).stem
        title = title.replace("_", " ").replace("-", " ")
        # clean up multiple spaces
        title = re.sub(r"\s+", " ", title).strip()
        return title


    def _clean_page_text(self, text: str) -> str:
        """Clean extracted PDF text.

        PDFs often have:
            headers and footers repeated on every page
            extra whitespace and line breaks
            page numbers embedded in text
            hyphenated words split across lines
        """
        # "employ-\nee" → "employee"
        text = re.sub(r"-\n", "", text)
        # lines that are just a number
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # multiple blank lines → single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)
        # normalise whitespace within lines
        lines = []
        for line in text.split("\n"):
            cleaned = re.sub(r"\s+", " ", line).strip()
            lines.append(cleaned)
        text = "\n".join(lines)

        return text.strip()


    def _load_single_pdf(self, file_path: Path) -> List[Document]:
        """Load a single PDF file into Document objects.

        One Document per page. Each Document carries page number and category.
        """
        documents = []
        category = DOCUMENT_CATEGORIES.get(file_path.name, "general")
        # build clean title from filename
        title = self._build_title(file_path.name)
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    raw_text = page.extract_text()
                    if not raw_text or not raw_text.strip():
                        logger.debug(
                            f"Skipping empty page {page_num} "
                            f"in '{file_path.name}'"
                        )
                        continue

                    content = (
                        self._clean_page_text(raw_text) if self.clean_text else raw_text
                    )

                    if not content.strip():
                        continue

                    # build Document for this page
                    document = Document(
                        content  = content,
                        source   = str(file_path.resolve()),
                        title    = title,
                        doc_type = "pdf",
                        metadata = {
                            "filename"    : file_path.name,
                            "category"    : category,
                            "page_number" : page_num,
                            "total_pages" : total_pages,
                            "file_size_bytes": file_path.stat().st_size,
                        },
                    )
                    documents.append(document)

        except Exception as e:
            logger.error(
                f"Failed to load '{file_path.name}': {e}"
            )
            return []

        return documents


    def load(self, source: str) -> List[Document]:
        """Load PDF files from a file path or folder."""
        path = Path(source)

        if not self.validate(source):
            raise FileNotFoundError(
                f"Invalid source: '{source}'. "
                f"Must be a .pdf file or folder of PDFs."
            )

        if path.is_file():
            pdf_files = [path]
        else:
            pdf_files = sorted(path.glob("*.pdf"))

        logger.info(
            f"Found {len(pdf_files)} PDF(s) to load from '{source}'"
        )

        all_documents = []
        for pdf_file in pdf_files:
            documents = self._load_single_pdf(pdf_file)
            all_documents.extend(documents)
            logger.info(
                f"Loaded '{pdf_file.name}' — "
                f"{len(documents)} page(s)"
            )

        self._log_loaded(all_documents, source)
        return all_documents
