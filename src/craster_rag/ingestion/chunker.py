"""

OLD CODE- MANUAL CHUNKING


Splits Document objects into smaller Chunk objects.

Chunking strategy:
1) Split by paragraphs first.
2)Group paragraphs into chunks up to chunk_size tokens
3) Add overlap between chunks
4) If single paragraph exceeds chunk_size, split by sentences
5) Add overlap between chunks so context is not lost at boundaries.


import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
import tiktoken

from craster_rag.ingestion.base_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    content:str
    source:str
    title:str
    doc_type:str
    chunk_index:int  #Position of this chunk in the document (0, 1, 2...)
    total_chunks:int
    token_count: int
    metadata:dict=field(default_factory=dict)
    chunk_id:Optional[str]=None

    #A chunk is a section of a document.It contains hundreds of tokens
    #chunk_size = 500 words
    #"Acumatica" = 1 word but 3 tokens
    #
    #500 words might actually be 800 tokens


    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id=f"{self.source}::chunk_{self.chunk_index}"

    def __repr__(self)->str:
        return (
            f"Chunk("
            f"title='{self.title}', "
            f"chunk={self.chunk_index + 1}/{self.total_chunks}, "
            f"tokens={self.token_count})"
        )

class Chunker:
    #Splits Document objects into Chunk objects.
    def __init__(self,chunk_size:int=500,chunk_overlap: int=50):
        if chunk_overlap>=chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be "
                f"less than chunk_size ({chunk_size})"
            )
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.tokenizer=tiktoken.get_encoding("cl100k_base")
        logger.info(
            f"Chunker initialised — "
            f"chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> list[Chunk]:
        # Loops through every Document and chunks each one.
        if not documents:
            logger.warning("No documents provided to chunker")
            return []

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_single_document(doc)
            all_chunks.extend(chunks)
            logger.info(
                f"'{doc.title}' → {len(chunks)} chunk(s)"
            )

        logger.info(
            f"Chunking complete — "
            f"{len(documents)} document(s) → "
            f"{len(all_chunks)} total chunk(s)"
        )
        return all_chunks


    def chunk_single_document(self,document:Document)->List[Chunk]:
        #        Split a single Document into Chunks.
        #Strategy:

        #    2. Group paragraphs into chunks up to chunk_size
        #    3. Add overlap between consecutive chunks

        if not document.content.strip():
            logger.warning(f"Empty document skipped: '{document.title}'")
            return []
        #1. Split content into paragraphs
        paragraphs=self.split_into_paragraphs(document.content)
        raw_chunks = self._group_into_chunks(paragraphs)
        overlapped_chunks = self._add_overlap(raw_chunks)
        total = len(overlapped_chunks)
        chunks = []

        for index, chunk_text in enumerate(overlapped_chunks):
            chunk = Chunk(
                content      = chunk_text.strip(),
                source       = document.source,
                title        = document.title,
                doc_type     = document.doc_type,
                chunk_index  = index,
                total_chunks = total,
                token_count  = self._count_tokens(chunk_text),
                metadata     = {
                    **document.metadata,        # carry over all doc metadata
                    "chunk_index"  : index,
                    "total_chunks" : total,
                },
            )
            chunks.append(chunk)

        return chunks


    def split_into_paragraphs(self, text: str) -> list[str]:
        #"Paragraph" here means any block of text separated by one or more blank lines.

        paragraphs = re.split(r"\n\s*\n", text)  # \n new line -s* space-\n new line
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs


    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        current_chunk = []
        current_tokens = 0
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)  # split on sentence endings (. ! ?)
        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _group_into_chunks(self, paragraphs: List[str]) -> list[str]:
        #Group paragraphs into chunks up to chunk_size tokens.
        #If a single paragraph is too long, splits it by sentences.
        chunks = []
        current_chunk = []
        current_tokens = 0
        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)
            if para_tokens > self.chunk_size:
                sentence_chunks = self._split_long_paragraph(paragraph)
                chunks.extend(sentence_chunks)
                continue

            if current_tokens + para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks


    def _add_overlap(self,chunks:list[str])->list[str]:
        #Overlap ensures each chunk has a bit of context
        if len(chunks)<=1:
            return chunks

        overlapped=[chunks[0]]

        for i in range(1,len(chunks)):
            current_chunk=chunks[i]
            previous_chunk=chunks[i-1]

            overlap_text=self._get_last_n_tokens(previous_chunk,self.chunk_overlap)
            overlapped_chunk=overlap_text+"\n\n "+current_chunk
            overlapped.append(overlapped_chunk)

        return overlapped


    def _get_last_n_tokens(self,text:str,n:int)->str:

        #Get the last n tokens of a string as text.
        tokens=self.tokenizer.encode(text)
        if len(tokens)<=n:
            return text
        last_n_token=tokens[-n:]
        return self.tokenizer.decode(last_n_token)
"""



"""NEW CODE-LANGCHAIN"""
"""Splits Document objects into Chunk objects using LangChain RecursiveCharacterTextSplitter.never splits mid word or mid sentence
RecursiveCharacterTextSplitter strategy:
    1. try splitting on paragraph breaks first
    2. if still too big split on line breaks
    3. if still too big split on spaces
    4. last resort split on characters

"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from craster_rag.ingestion.base_loader import Document

# logger
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    content      : str
    source       : str
    title        : str
    category     : str
    doc_type     : str
    chunk_index  : int
    total_chunks : int
    token_count  : int
    page_number  : int                   = 0
    metadata     : dict                  = field(default_factory=dict)
    chunk_id     : Optional[str]         = None

    def __post_init__(self):
        """Auto generate chunk_id from source and chunk_index."""
        if self.chunk_id is None:
            self.chunk_id = f"{self.source}::chunk_{self.chunk_index}"

    def __repr__(self) -> str:
        return (
            f"Chunk("
            f"title='{self.title}', "
            f"category='{self.category}', "
            f"page={self.page_number}, "
            f"chunk={self.chunk_index + 1}/{self.total_chunks}, "
            f"tokens={self.token_count})"
        )

class Chunker:
    """ LangChain measures in CHARACTERS not tokens
     chunk_size    : max characters per chunk (default 1000)
       1000 characters ≈ 250 tokens
     chunk_overlap : overlap between chunks (default 100)
    """

    def __init__(self, chunk_size:int=1000, chunk_overlap:int=100):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be "
                f"less than chunk_size ({chunk_size})"
            )

        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,       # measure by characters
            is_separator_regex=False,
        )
        logger.info(
            f"Chunker initialised — "
            f"chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )


    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:

        all_chunks: list[Chunk] = []

        for document in documents:
            chunks = self._chunk_single_document(document)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunking complete — "
            f"{len(documents)} document(s) → "
            f"{len(all_chunks)} total chunk(s)"
        )
        return all_chunks


    def _chunk_single_document(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            logger.warning(
                f"Empty document skipped: '{document.title}'"
            )
            return []
        # split content using LangChain
        text_chunks = self.splitter.split_text(document.content)
        #sample text_chunks= ["aaaaaaaaa bbbbbbbb","cccccccc","sssssss bbbb"]

        if not text_chunks:
            logger.warning(
                f"No chunks produced for: '{document.title}'"
            )
            return []

        # get metadata from document
        category    = document.metadata.get("category",    "general")
        page_number = document.metadata.get("page_number", 0)

        total = len(text_chunks)
        chunks = []
        for index, text in enumerate(text_chunks):
            if not text.strip():
                continue

            chunk = Chunk(
                content      = text.strip(),
                source       = document.source,
                title        = document.title,
                category     = category,
                doc_type     = document.doc_type,
                chunk_index  = index,
                total_chunks = total,
                token_count  = self._estimate_tokens(text),
                page_number  = page_number,
                metadata     = {
                    **document.metadata,
                    "chunk_index"  : index,
                    "total_chunks" : total,
                },
            )
            chunks.append(chunk)

        logger.debug(
            f"'{document.title}' page {page_number} "
            f"→ {len(chunks)} chunk(s)"
        )
        return chunks


    def _estimate_tokens(self, text: str) -> int:
        """ 1 token ≈ 4 characters for English text

            Estimated token count as integer
        """
        return len(text) // 4
