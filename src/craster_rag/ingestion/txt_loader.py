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
        try:
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

        except UnicodeDecodeError:
            # file has unusual encoding, try latin-1 as fallback
            logger.warning(
                f"UTF-8 decode failed for '{file_path.name}', "
                f"trying latin-1 encoding"
            )
            try:
                content = file_path.read_text(encoding="latin-1").strip()
                title = file_path.stem.replace("-", " ").replace("_", " ")
                return Document(
                    content=content,
                    source=str(file_path.resolve()),
                    title=title,
                    doc_type="txt",
                    metadata={
                        "filename": file_path.name,
                        "file_stem": file_path.stem,
                        "file_size_bytes": file_path.stat().st_size,
                        "folder": str(file_path.parent.resolve()),
                        "encoding": "latin-1",
                    },
                )
            except Exception as e:
                logger.error(
                    f"Failed to load '{file_path.name}': {e}"
                )
                return None

        except Exception as e:
            logger.error(f"Unexpected error loading '{file_path.name}': {e}")
            return None


"""TxtLoader-inherits form base __loader__
loads single file
|
take file path as input-reads content of the file
|
reads the file name -strips - and _
|
return document

----------------------------------------------------------

|
calls abstract load
|
runs the validate first
|
checks if the path is a sing file or fodler and laods the entirre path into array
|
for every path -load singel file and return document
|
append all document objects to Documents

--------------------------------------------------------
 |
 call abstractmethod validate - add what it has to do here
 |
 check if link exists
 |
 check if is a File-check if its is a .txt file
 |
 check if path is FOlder- List of .txt files
"""