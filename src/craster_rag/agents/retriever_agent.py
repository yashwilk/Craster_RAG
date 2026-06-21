"""Retrieves the most relevant chunks from Supabase
using hybrid search (vector + BM25)."""


import logging
 
from craster_rag.agents.state import RAGState
from craster_rag.ingestion.embedder import Embedder
from craster_rag.retrieval.vector_store import VectorStore
from config import settings
 
# logger
logger = logging.getLogger(__name__)


# avoids reloading embedding model on every query
_embedder     = Embedder(
    model_name = settings.embedding_model,
    batch_size = settings.embedding_batch_size,
)
_vector_store = VectorStore()


def _retrieve_chunks(
        query,
        category:str|None
) -> list:

    """Run hybrid search and return top chunks."""
    # embed the query
    query_embedding = _embedder.embed_query(query)
    if settings.enable_hybrid_search:
        # hybrid search combines vector + BM25
        chunks = _vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=settings.top_k_results,
            category=category
        )
    else:
        # vector only search
        chunks = _vector_store.search(
            query_embedding = query_embedding,
            top_k           = settings.top_k_results,
            category        = category,
        )
 
    return chunks


def retriever_agent(state: RAGState) -> RAGState:

    query           = state["rewritten_query"] or state["question"]
    category        = state["category"]
    retry_count     = state["retry_count"]
    search_category = None if retry_count > 0 else category

    if retry_count > 0:
        logger.info(
            f"Retriever: retry {retry_count} — "
            f"broadening search to all categories"
        )
    else:
        logger.info(
            f"Retriever: searching category '{category}' "
            f"for query '{query[:50]}...'"
        )

    try:
        chunks = _retrieve_chunks(query, search_category)

    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        chunks = []
 
    # extract scores for state
    hybrid_scores = [chunk.score for chunk in chunks]
 
    logger.info(
        f"Retriever: found {len(chunks)} chunk(s)"
    )

    return {
        **state,
        "chunks"        : chunks,
        "hybrid_scores" : hybrid_scores,
        "vector_scores" : [],   # individual scores from VectorStore
        "bm25_scores"   : [],   # individual scores from VectorStore
    }