"""# INPUT
searcher = Searcher()
results  = searcher.search("how do I create a new user?")

# OUTPUT
results = [
    SearchResult(
        content = "Step 1: Go to User Management...",
        title   = "How to create a new user account",
        score   = 0.92
    ),
    SearchResult(
        content = "Click Add New User button...",
        title   = "How to create a new user account",
        score   = 0.87
    ),
]"""

#Takes a user question and returns the most relevant chunks from the vector store.


"""What it does:
    1. embeds the user question
    2. searches vector store for similar chunks
    3. filters out low quality results
    4. returns clean SearchResult objects
 """

import logging
from typing import List, Optional
from craster_rag.ingestion.embedder import Embedder
from craster_rag.retrieval.vector_store import VectorStore, SearchResult
from config import settings

# logger
logger = logging.getLogger(__name__)


class Searcher:
    #Takes a plain text question from the user.
    #Returns the most relevant SearchResult objects.

    def __init__(
            self,
            embedder     : Optional[Embedder]    = None,
            vector_store : Optional[VectorStore] = None,
            min_score    : float                 = 0.7,
            top_k        : int                   = 0,
        ):

        self._embedder     = embedder     or Embedder()
        self._vector_store = vector_store or VectorStore()
        self.min_score     = min_score
        self.top_k         = top_k or settings.top_k_results

        logger.info(
            f"Searcher initialised — "
            f"top_k={self.top_k}, "
            f"min_score={self.min_score}"
        )

    def search(
            self,
            query:str,
            doc_type_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        # step 1 — validate query
        query = query.strip()
        if not query:
            logger.warning("Empty query received")
            return []

        logger.info(f"Searching for: '{query[:50]}...'")
        # step 2 — embed query
        query_embedding = self._embed_query(query)
        if not query_embedding:
            return []

        # step 3 — search vector store
        results = self._search_vector_store(
            query_embedding,
            doc_type_filter,
        )
        if not results:
            logger.info("No results found in vector store")
            return []

        # step 4 — filter low quality results
        results = self._filter_by_score(results)

        logger.info(
            f"Search complete — "
            f"{len(results)} result(s) returned"
        )

        return results

    def _embed_query(self, query: str) -> List[float]:
        try:
            embedding = self._embedder.embed_query(query)
            logger.debug(
                f"Query embedded — "
                f"dimension={len(embedding)}"
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

    def _search_vector_store(
            self, query_embedding: List[float], doc_type_filter: Optional[str]
    ) -> List[SearchResult]:
        try:
            results = self._vector_store.search(
                query_embedding = query_embedding,
                top_k           = self.top_k,
                doc_type_filter = doc_type_filter,
            )
            logger.debug(
                f"Vector store returned {len(results)} result(s)"
            )
            return results

        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []

    def _filter_by_score(
            self, results: List[SearchResult]) -> List[SearchResult]:
        filtered = [result for result in results if result.score >= self.min_score]

        if len(filtered) < len(results):
            logger.info(
                f"Filtered {len(results) - len(filtered)} "
                f"low quality result(s) "
                f"below score {self.min_score}"
            )

        return filtered
