#Every loader (TXT, PDF, SharePoint, Word) must inherit from this class and implement the load() and validate() methods.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging


logger=logging.getLogger(__name__)

@dataclass
class Document:
    content:str
    source:str #source: Path to file/folder or URL depending on loader type
    title:str
    doc_type:str
    metadata:dict =field(default_factory=dict)
    doc_id:Optional[str]=None

    def __post_init__(self):
        """Auto generate doc_id from source if not provided."""
        if self.doc_id is None:
            # use the source path/url as the ID
            self.doc_id = self.source

    def __repr__(self)->str:
        return (
            f"Document(title='{self.title}', "
            f"type='{self.doc_type}', "
            f"chars={len(self.content)}, "
            f"source='{self.source}')"
        )



class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Document]:
        pass

    @abstractmethod
    def validate(self, source: str) -> bool:
        pass

    def _log_loaded(self, documents: list[Document], source: str) -> None:
        logger.info(
            f"Loaded {len(documents)} document(s) from '{source}'"
        )

    def _is_valid_path(self, path: str) -> bool:
        """
        Helper to check if a file/folder path exists.
        Used by file-based loaders (TXT, PDF, Word).
        """
        return Path(path).exists()


"""
Class Document
|
define datype
|
_post_init_ - to check doc_id
|
__repr__ - to se how each file looks like and capture all info
|
base loader
|
@abstractmethod load - dont define anything here. returns list[Document] 
|
@abstractmethod valid - dont define anything here 
|
_log_loaded- - to to log the length of each file in folder (applicable for all)
|
is_valid_path - to check if file path is valid (applicable for all)


"""