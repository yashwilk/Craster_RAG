import logging
from pathlib import Path
from craster_rag.ingestion.base_loader import BaseLoader, Document

logger = logging.getLogger(__name__)


class TxtLoader(BaseLoader):
    SUPPORT_EXTENSION = ".txt"

    def validate(self, source: str) -> bool:
        path = Path(source)

        if not path.exists():
            logger.error(f"Source path does not exist: '{source}'")
            return False

        if path.is_file():
            if path.suffix.lower() != self.SUPPORT_EXTENSION:
                logger.error(
                    f"File '{source}' is not a .txt file. "
                    f"Got extension: '{path.suffix}'"
                )
                return False
            return True

        if path.is_dir():
            txt_files = list(path.glob("*.txt"))
            if not txt_files:
                logger.warning(f"No .txt files found in folder: '{source}'")
                return False
            return True

        return False
    

    def load(self,source:str)->list[Document]:
        path=Path(source)


        if not self.validate(source):
            raise FileNotFoundError(
                f"Invalid source: '{source}'. "
                f"Must be a .txt file or folder containing .txt files."
            )
        
        if path.is_file():
            txt_files = [path]
        else:
            txt_files = sorted(path.glob("*.txt"))

        logger.info(f"Found {len(txt_files)} .txt file(s) to load from '{source}'")

        documents = []

        for txt_file in txt_files:
            document = self._load_single_file(txt_file)
            if document is not None:
                documents.append(document)

        self._log_loaded(documents, source)
        return documents


    def _load_single_file(self, file_path: Path) -> Document | None:
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            logger.warning(f"Skipping empty file: '{file_path.name}'")
            return None

        # "Procedure-001-Arcus-Support-Ticket.txt" → "Procedure 001 Arcus Support Ticket"
        title = file_path.stem.replace("-", " ").replace("_", " ")

        document = Document(
            content=content,
            source=str(file_path.resolve()),
            title=title,
            doc_type="txt",
            metadata={
                "filename": file_path.name,
                "file_stem": file_path.stem,
                "file_size_bytes": file_path.stat().st_size,
                "folder": str(file_path.parent.resolve())
            }
        )

        logger.info(
            f"Loaded '{file_path.name}' "
            f"({len(content)} characters)"
        )
        return document


