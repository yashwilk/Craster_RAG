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
from typing import List, Optional

from supabase import create_client, Client

from config import settings

from craster_rag.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    source: str
    title: str
    doc_type: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: dict
    score: float

    def __repr__(self) -> str:
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
    def __init__(self, supabase_url: str = "", supabase_key: str = "", table_name: str = "chunks"):
        self.supabase_url = supabase_url or settings.SUPABASE_URL
        self.supabase_key = supabase_key or settings.SUPABASE_KEY
        self.table_name = table_name
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key must be provided")

        self._client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info(
            f"VectorStore initialised — "
            f"table='{table_name}'"
        )

    def _insert_chunk(self, chunk: EmbeddedChunk) -> None:
        (self._client
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

    def _chunk_exists(self, chunk_id: str) -> bool:
        response = (
            self._client
            .table(self.table_name)
            .select("chunk_id")
            .eq("chunk_id", chunk_id)
            .execute()
        )
        return len(response.data) > 0

    def add_chunk(self, chunks: List[EmbeddedChunk], skip_existing: bool = True) -> int:
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return 0
        stored_count = 0

        for chunk in chunks:
            try:
                if skip_existing and self._chunk_exists(chunk.chunk_id):
                    logger.debug(
                        f"Skipping existing chunk: '{chunk.chunk_id}'"
                    )
                    continue
                self._insert_chunk(chunk)
                stored_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to store chunk '{chunk.chunk_id}': {e}"
                )
                continue
        logger.info(
            f"Stored {stored_count}/{len(chunks)} chunk(s) in Supabase"
        )
        return stored_count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        # Find the most similar chunks to a query embedding. Uses cosine similarity via pgvector.
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        try:
            response = self._client.rpc(
                "match_chunks", {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                    "filter": {"doc_type": doc_type_filter} if doc_type_filter else {}
                }
            ).execute()

            results = []
            for row in response.data:
                result = SearchResult(
                    chunk_id     = row["chunk_id"],
                    content      = row["content"],
                    source       = row["source"],
                    title        = row["title"],
                    doc_type     = row["doc_type"],
                    chunk_index  = row["chunk_index"],
                    total_chunks = row["total_chunks"],
                    token_count  = row["token_count"],
                    metadata     = row["metadata"] or {},
                    score        = row["similarity"],
                )
                results.append(result)

            logger.info(f"Search returned {len(results)} result(s)")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise




    def delete_by_source(self, source: str) -> int:
        try:
            response = (
                self._client
                .table(self.table_name)
                .delete()
                .eq("source", source)
                .execute()
            )
            deleted = len(response.data) if response.data else 0
            logger.info(f"Deleted {deleted} chunk(s) for source '{source}'")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise


    def get_stats(self) -> dict:
        try:
            total_response = (
                self._client
                .table(self.table_name)
                .select("chunk_id", count="exact")
                .execute()
            )
            total = total_response.count or 0

            sources_response = (
                self._client
                .table(self.table_name)
                .select("source")
                .execute()
            )
            unique_sources = len(set(row["source"] for row in sources_response.data))

            doc_type_response = (
                self._client
                .table(self.table_name)
                .select("doc_type")
                .execute()
            )
            doc_types: dict = {}
            for row in doc_type_response.data:
                dt = row["doc_type"]
                doc_types[dt] = doc_types.get(dt, 0) + 1

            return {
                "total_chunks"   : total,
                "unique_sources" : unique_sources,
                "doc_types"      : doc_types,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
