"""
Reranks retrieved chunks using a cross-encoder model.
Vector search encodes question and chunk SEPARATELY
    then compares vectors — fast but loses nuance.

  Cross-encoder reranking encodes question and chunk
    TOGETHER in one forward pass — slower but the model
    directly sees their relationship, giving much more
    accurate relevance scoring.

    cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
from typing import TYPE_CHECKING, Optional

from craster_rag.agents.state import RAGState
from config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)

_reranker_model: Optional["CrossEncoder"] = None


def _get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker model: "
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )

    return _reranker_model


def reranker_agent(state: RAGState) -> RAGState:

    question = state["question"]
    chunks   = state["chunks"]

    if not chunks:
        logger.info("Reranker: no chunks to rerank")
        return state

    # feature flag — no-op pass-through when disabled
    if not settings.enable_reranking:
        logger.debug("Reranker: disabled — truncating to top_k_results")
        return {
            **state,
            "chunks": chunks[: settings.top_k_results],
        }

    logger.info(
        f"Reranker: scoring {len(chunks)} chunk(s) "
        f"against question"
    )

    try:
        reranked_chunks = _rerank_chunks(question, chunks)

    except Exception as e:
        # never let reranking crash the pipeline
        # fall back to original retriever order
        logger.warning(
            f"Reranking failed: {e}. "
            f"Falling back to original retriever order."
        )
        reranked_chunks = chunks[: settings.top_k_results]

    logger.info(
        f"Reranker: narrowed to {len(reranked_chunks)} chunk(s)"
    )

    return {
        **state,
        "chunks"        : reranked_chunks,
        "hybrid_scores" : [c.score for c in reranked_chunks],
    }


def _rerank_chunks(question: str, chunks: list) -> list:

    pairs = [(question, chunk.content) for chunk in chunks]

    model = _get_reranker_model()

    # predict returns raw logit scores
    raw_scores = model.predict(pairs)

    scored_chunks = list(zip(chunks, raw_scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    top_chunks = scored_chunks[:settings.top_k_results]
    reranked = []
    for chunk, raw_score in top_chunks:
        # normalise raw logit to 0-1 via sigmoid
        normalised_score = _sigmoid(raw_score)
        chunk.score = round(float(normalised_score), 4)
        reranked.append(chunk)

    return reranked


def _sigmoid(x: float) -> float:
    """Convert raw cross-encoder logit score to 0-1 range."""
    import math
    return 1 / (1 + math.exp(-x))
