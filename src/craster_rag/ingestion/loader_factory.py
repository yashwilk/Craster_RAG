
"""Factory that picks the right loader for each file type."""


import logging
from pathlib import Path
from craster_rag.ingestion.base_loader import BaseLoader, Document
from craster_rag.ingestion.pdf_loader import PdfLoader
from craster_rag.ingestion.txt_loader import TxtLoader


logger = logging.getLogger(__name__)


class LoadFactory:
    """
    Factory class that creates the right loader for each file type.

    All methods are class methods — you never
    need to instantiate LoaderFactory directly.
    """

    _REGISTRY: dict[str, type[BaseLoader]] = {
        ".pdf": PdfLoader,
        ".txt": TxtLoader,
    }

    @classmethod
    def get_loader(cls, extension: str) -> BaseLoader:
        ext = extension.lower().strip()
        if ext not in cls._REGISTRY:
            supported = list(cls._REGISTRY.keys())
            raise ValueError(
                f"No loader for '{ext}'. "
                f"Supported: {supported}"
            )
        loader_class = cls._REGISTRY[ext]
        logger.debug(f"Using {loader_class.__name__} for '{ext}'")
        return loader_class()

    @classmethod
    def get_loader_for_files(cls, file_path: str) -> BaseLoader:
        extension = Path(file_path).suffix.lower()
        return cls.get_loader(extension)

    @classmethod
    def load_all(cls, folder_path: str) -> list[Document]:
        """
        Load all supported files from a folder.

        Scans folder for all supported file types.
        Picks right loader for each type.
        Returns all documents in one flat list.
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: '{folder_path}'"
            )

        if not folder.is_dir():
            raise ValueError(
                f"'{folder_path}' is not a folder"
            )

        all_files = cls._find_supported_files(folder)

        if not all_files:
            logger.warning(
                f"No supported files found in '{folder_path}'. "
                f"Supported types: {list(cls._REGISTRY.keys())}"
            )
            return []

        logger.info(
            f"Found {len(all_files)} supported file(s) "
            f"in '{folder_path}'"
        )

        files_by_ext: dict[str, list[Path]] = {}
        for file in all_files:
            ext = file.suffix.lower()
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(file)

        all_documents: list[Document] = []
        for ext, files in files_by_ext.items():
            try:
                loader = cls.get_loader(ext)
                logger.info(
                    f"Loading {len(files)} {ext} file(s) "
                    f"with {loader.__class__.__name__}"
                )
                for file in files:
                    try:
                        docs = loader.load(str(file))
                        all_documents.extend(docs)
                    except Exception as e:
                        logger.error(f"Failed to load '{file.name}': {e}")
                        continue
            except ValueError as e:
                logger.warning(f"Skipping {ext} files: {e}")
                continue

        logger.info(
            f"LoaderFactory loaded {len(all_documents)} "
            f"document(s) from '{folder_path}'"
        )
        return all_documents

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return list of supported file extensions."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if a file type is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls._REGISTRY

    @classmethod
    def _find_supported_files(cls, folder: Path) -> list[Path]:
        """Find all supported files in a folder."""
        supported_files = []
        for ext in cls._REGISTRY.keys():
            files = list(folder.glob(f"*{ext}"))
            supported_files.extend(files)
        return sorted(supported_files)
