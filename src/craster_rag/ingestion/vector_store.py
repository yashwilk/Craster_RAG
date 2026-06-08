"""input for this file
embedded_chunks = [
    EmbeddedChunk(
        content      = "Step 1: Go to User Management...",
        source       = "C:/craster-rag/data/procedures/Procedure-001.txt",
        title        = "How to create a new user account",
        doc_type     = "txt",
        chunk_index  = 0,
        total_chunks = 3,
        token_count  = 487,
        chunk_id     = "C:/craster-rag/.../Procedure-001.txt::chunk_0",
        embedding    = [0.21, 0.84, 0.11, 0.63, 0.44, ...]  ← 768 numbers
    ),(),()"""

"""
Two main operations:
    1. STORE  — save embedded chunks during ingestion
    2. SEARCH — find similar chunks at query time
"""

import logging
from dataclasses import dataclass
from typing import Optional

from supabase import create_client,Client

from config import settings

from craster_rag.ingestion.embedder import EmbeddedChunk

logger=logging.getLogger(__name__)


@dataclass
class SearchResult:
    #A single search result returned from vector store.
    chunk_id:str
    content:str
    source:str
    title:str
    doc_type:str
    chunk_index:int
    total_chunks:int
    token_count:int
    metadata:dict
    score:float

    def __repr__(self)->str:
        return (
            f"SearchResult("
            f"title='{self.title}', "
            f"chunk={self.chunk_index + 1}/{self.total_chunks}, "
            f"score={self.score:.3f})"
        )

class VectorStore:
    """
    creating the chunks table if it doesnt exist
    storing embedded chunks
    searching by vector similarity
    checking for duplicate chunks
    deleting chunks by source file

    """
    def __init__(self,supabase_url:str="",supabase_key:str="",table_name:str="chunks"):
        self.supabase_url=supabase_url or settings.SUPABASE_URL
        self.supabase_key=supabase_key or settings.SUPABASE_KEY
        self.table_name=table_name
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key must be provided")

        # create Supabase client
        self.client:Client=create_client(self.supabase_url,self.supabase_key)
        logger.info(
            f"VectorStore initialised — "
            f"table='{table_name}'"
        )

    #Insert a single EmbeddedChunk into Supabase.
    def _insert_chunk(self,chunk:EmbeddedChunk)->None:
        (self.client
            .table(self.table_name)
            .upsert({
                "chunk_id"    : chunk.chunk_id,
                "content"     : chunk.content,
                "source"      : chunk.source,
                "title"       : chunk.title,
                "doc_type"    : chunk.doc_type,
                "chunk_index" : chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "token_count" : chunk.token_count,
                "metadata"    : chunk.metadata,
                "embedding"   : chunk.embedding,
            }).execute())

    #Check if a chunk already exists in Supabase.
    def _chunk_exists(self,chunk_id:str)->bool:
        response=(
            self.client
            .table(self.table_name)
            .select("chunk_id")
            .eq("chunk_id",chunk_id)
            .execute()
        )
        return len(response.data)>0

    #store a list of EmbeddedChunk objects in Supabase.
    def add_chunk(self,chunks:list[EmbeddedChunk],skip_existing:bool=True)->int:
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return 0
        stored_count=0

        for chunk in chunks:
            try:
                if skip_existing and self._chunk_exists(chunk.chunk_id):
                    logger.debug(
                        f"Skipping existing chunk: '{chunk.chunk_id}'"
                    )
                    continue
                self._insert_chunk(chunk)
                stored_count+=1
            except Exception as e:
                logger.error(
                    f"Failed to store chunk '{chunk.chunk_id}': {e}"
                )
                continue
        logger.info(
            f"Stored {stored_count}/{len(chunks)} chunk(s) in Supabase"
        )
        return stored_count
