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

"""Document = 10,000 tokens

Chunk size = 500 tokens

Result = 20 chunks-total chunk/cunkindex is index withing this 20 chunks"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from supabase import create_client, Client

from config import settings

from craster_rag.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    A single search result returned from vector store.

    Attributes:
        chunk_id     : unique chunk identifier
        content      : the chunk text
        source       : original file path
        title        : document title
        category     : document category (leave_family etc)
        doc_type     : pdf, txt etc
        chunk_index  : position in document
        total_chunks : total chunks in document
        token_count  : number of tokens
        page_number  : page in original PDF
        metadata     : extra info as dict
        score        : similarity score 0-1
    """
    chunk_id    : str
    content     : str
    source      : str
    title       : str
    category    : str
    doc_type    : str
    chunk_index : int
    total_chunks: int
    token_count : int
    page_number : int
    metadata    : dict
    score       : float

    def __repr__(self) -> str:
        return (
            f"SearchResult("
            f"title='{self.title}', "
            f"category='{self.category}', "
            f"page={self.page_number}, "
            f"score={self.score:.3f})"
        )


class VectorStore:
    """
     Handles:
        storing embedded chunks with category metadata
        vector similarity search
        BM25 keyword search
        hybrid search combining both
        category filtering
        chunk deletion for re-indexing

    """
    def __init__(self, supabase_url: str = "", supabase_key: str = "", table_name: str = "chunks"):
        self.supabase_url = supabase_url or settings.supabase_url
        self.supabase_key = supabase_key or settings.supabase_key
        self.table_name = table_name
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key must be provided")

        # create Supabase client
        self._client: Client = create_client(
            self.supabase_url,
            self.supabase_key,
        )

        logger.info(
            f"VectorStore initialised — "
            f"table='{table_name}'"
        )


    def _insert_chunk(self, chunk: EmbeddedChunk) -> None:
        # get category and page_number from metadata
        category    = chunk.metadata.get("category", "general")
        page_number = chunk.metadata.get("page_number", 0)

        self._client.table(self.table_name).upsert({
            "chunk_id"    : chunk.chunk_id,
            "content"     : chunk.content,
            "source"      : chunk.source,
            "title"       : chunk.title,
            "category"    : category,
            "doc_type"    : chunk.doc_type,
            "chunk_index" : chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "token_count" : chunk.token_count,
            "page_number" : page_number,
            "metadata"    : chunk.metadata,
            "embedding"   : chunk.embedding,
        }).execute()


    def _chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if chunk already exists in Supabase.

        """
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
            logger.info(f"Deleted {deleted} chunks for '{source}'")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise


    def _rows_to_results(
        self,
        rows       : list[dict],
        score_field: str = "similarity",
    ) -> list[SearchResult]:
        """Convert raw Supabase rows to SearchResult objects."""
        results = []
        for row in rows:
            result = SearchResult(
                chunk_id    = row.get("chunk_id", ""),
                content     = row.get("content", ""),
                source      = row.get("source", ""),
                title       = row.get("title", ""),
                category    = row.get("category", "general"),
                doc_type    = row.get("doc_type", "pdf"),
                chunk_index = row.get("chunk_index", 0),
                total_chunks= row.get("total_chunks", 1),
                token_count = row.get("token_count", 0),
                page_number = row.get("page_number", 0),
                metadata    = row.get("metadata") or {},
                score       = float(row.get(score_field, 0.0)),
            )
            results.append(result)
        return results


    def search(
        self,
        query_embedding : list[float],
        top_k           : int            = 5,
        category        : Optional[str]  = None,
    ) -> list[SearchResult]:
        """
        Pure vector similarity search.

        Used when hybrid search is disabled or as part of hybrid search."""

        # Find the most similar chunks to a query embedding. Uses cosine similarity via pgvector.
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        try:

            response = self._client.rpc(
                "match_chunks",
                {
                    "query_embedding" : query_embedding,
                    "match_count"     : top_k,
                    "filter_category" : category or "",
                }
            ).execute()

            results = self._rows_to_results(
                response.data,
                score_field="similarity"
            )

            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise



    def bm25_search(
        self,
        query_text : str,
        top_k      : int           = 5,
        category   : Optional[str] = None,
    ) -> list[SearchResult]:
        """BM25 full text keyword search. Finds chunks containing exact keywords from query."""

        if not query_text.strip():
            raise ValueError("query_text cannot be empty")

        try:
            response = self._client.rpc(
                "bm25_search_chunks",
                {
                    "query_text"      : query_text,
                    "match_count"     : top_k,
                    "filter_category" : category or "",
                }
            ).execute()

            results = self._rows_to_results(
                response.data,
                score_field="rank"
            )

            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise


    def _reciprocal_rank_fusion(
        self,
        vector_results : list[SearchResult],
        bm25_results   : list[SearchResult],
        vector_weight  : float = 0.7,
        bm25_weight    : float = 0.3,
        k              : int   = 60,
    ) -> list[SearchResult]:
        """Formula:
            score = vector_weight * 1/(vector_rank + k)
                  + bm25_weight   * 1/(bm25_rank   + k)  k=60 is standard RRF constant.
        Prevents top rank from dominating too much."""

        all_chunks: dict[str, SearchResult] = {}

        for result in vector_results:
            all_chunks[result.chunk_id] = result

        for result in bm25_results:
            if result.chunk_id not in all_chunks:
                all_chunks[result.chunk_id] = result

        # calculate RRF score for each chunk
        rrf_scores: dict[str, float] = {}

        # vector rankings
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += vector_weight * (1.0 / (rank + k))

        # bm25 rankings
        for rank, result in enumerate(bm25_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += bm25_weight * (1.0 / (rank + k))

        # sort by combined RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key    = lambda x: x[1],
            reverse= True,
        )

        final_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = all_chunks[chunk_id]
            result.score = round(rrf_score, 4)
            final_results.append(result)

        return final_results


    def hybrid_search(
        self,
        query_embedding : list[float],
        query_text      : str,
        top_k           : int           = 5,
        category        : Optional[str] = None,
        vector_weight   : float         = 0.0,
        bm25_weight     : float         = 0.0,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector and BM25 results.

        Uses Reciprocal Rank Fusion (RRF) to combine
        ranked lists from both search methods.

        RRF formula for each chunk:
            score = vector_weight * 1/(vector_rank + 60)
                  + bm25_weight   * 1/(bm25_rank   + 60)

        60 is the RRF constant (standard value)
        prevents very high scores for rank 1"""
        # use config weights if not provided
        v_weight = vector_weight or settings.vector_weight
        b_weight = bm25_weight   or settings.bm25_weight

        vector_results = []
        bm25_results   = []

        try:
            vector_results = self.search(
                query_embedding,
                top_k    = top_k,
                category = category,
            )
        except Exception as e:
            logger.warning(f"Vector search failed in hybrid: {e}")

        try:
            bm25_results = self.bm25_search(
                query_text,
                top_k    = top_k,
                category = category,
            )
        except Exception as e:
            logger.warning(f"BM25 search failed in hybrid: {e}")

        # if one search failed use the other
        if not vector_results and not bm25_results:
            logger.error("Both searches failed")
            return []

        if not vector_results:
            return bm25_results[:top_k]

        if not bm25_results:
            return vector_results[:top_k]

        # combine with RRF
        combined = self._reciprocal_rank_fusion(
            vector_results = vector_results,
            bm25_results   = bm25_results,
            vector_weight  = v_weight,
            bm25_weight    = b_weight,
        )

        logger.info(
            f"Hybrid search returned {len(combined[:top_k])} results"
        )
        return combined[:top_k]


    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dict with total_chunks, sources, categories
        """
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
            unique_sources = len(set(
                row["source"] for row in sources_response.data
            ))

            categories_response = (
                self._client
                .table(self.table_name)
                .select("category")
                .execute()
            )
            categories: dict = {}
            for row in categories_response.data:
                cat = row["category"]
                categories[cat] = categories.get(cat, 0) + 1

            return {
                "total_chunks"   : total,
                "unique_sources" : unique_sources,
                "categories"     : categories,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
