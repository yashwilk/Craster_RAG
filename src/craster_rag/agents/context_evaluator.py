"""
aluates whether retrieved chunks are good enough    to answer the user question accurately.
It prevents hallucination by refusing to answer    when context quality is poor.

Confidence levels:
    high    score > 0.8   answer directly
    medium  score 0.5-0.8 answer with caveat
    low     score < 0.5   retry retrieval
    none    no chunks     cannot answer


On low confidence:
    increments retry_count
    sends pipeline back to retriever
    retriever broadens search on retry
    maximum 2 retries before giving up


Scoring approach:
    1. checks if any chunks were retrieved
    2. looks at hybrid similarity scores
    3. checks how many chunks are above threshold
    4. checks if chunks are from correct category
    5. combines into final confidence score

"""

import logging

from craster_rag.agents.state import RAGState
from config import settings

# logger
logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD   = settings.high_confidence   # 0.8
MEDIUM_CONFIDENCE_THRESHOLD = settings.min_confidence    # 0.5
MAX_RETRIES                 = 2

def context_evaluator_agent(state:RAGState)->RAGState:

    chunks=state["chunks"]
    question=state["question"]
    category=state["category"]
    retry_count=state["retry_count"]

    logger.info(
        f"Evaluator: assessing {len(chunks)} chunk(s) "
        f"for question '{question[:50]}...'"
    )

    # calculate context score
    score = _calculate_context_score(
        chunks   = chunks,
        category = category,
    )

 # determine confidence level
    confidence_level = _determine_confidence(
        score       = score,
        retry_count = retry_count,
    )

  # determine if pipeline can answer
    can_answer = confidence_level in ("high", "medium")

    logger.info(
        f"Evaluator: score={score:.2f}, "
        f"confidence={confidence_level}, "
        f"can_answer={can_answer}"
    )

 # increment retry count if low confidence
    new_retry_count = retry_count
    if confidence_level == "low":
        new_retry_count = retry_count + 1
        logger.info(
            f"Evaluator: low confidence — "
            f"retry {new_retry_count}/{MAX_RETRIES}"
        )




    return {
        **state,
        "context_score": score,
        "confidence_level": confidence_level,
        "can_answer": can_answer,
        "retry_count": new_retry_count,

    }




def _calculate_context_score(chunks:list,category:str)->float:
    """ Scoring factors:
        1. no chunks retrieved          → 0.0
        2. average hybrid similarity    → main factor
        3. proportion above threshold   → quality factor
        4. category match bonus         → accuracy bonus"""

    # no chunks retrieved
    if not chunks:
        logger.warning("Evaluator: no chunks retrieved")
        return 0.0

  # factor 1 — average similarity score
    scores =[chunk.score for chunk in chunks]
    avg_score=sum(scores)/len(scores)


   # factor 2 — proportion of chunks above 0.6 threshold
    good_chunks=sum(1    for s in scores if s>=0.6)
    quality=good_chunks/len(chunks)

# factor 3 — category match bonus
# chunks from correct category get a small bonus

    if category and category!="general":
        matching=sum(1  for chunk in chunks if chunk.category==category)
        category_match=matching/len(chunks)
        category_bonus=category_match*0.1
    else:
        category_bonus = 0.05   # small bonus for general

  # combine factors/weighted avg
    raw_score = (avg_score * 0.7) + (quality * 0.3) + category_bonus
    # clamp to 0.0 - 1.0
    final_score = min(1.0, max(0.0, round(raw_score, 3)))

    logger.debug(
        f"Evaluator: avg_score={avg_score:.2f}, "
        f"quality={quality:.2f}, "
        f"category_bonus={category_bonus:.2f}, "
        f"final={final_score:.2f}"
    )

    return final_score



def _determine_confidence( score      : float, retry_count: int) -> str:


    """ Convert numeric score to confidence level string. """

  # too many retries → give up
    if retry_count >= MAX_RETRIES:
        return "none"

  # no context at all
    if score == 0.0:
        return "none"

  # high confidence → answer directly
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"

    # medium confidence → answer with caveat
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"

    # low confidence → retry
    return "low"



def should_retry(state: RAGState) -> str:
    """gGraph conditional edge function.

    Called by LangGraph after context_evaluator_agent
    to decide which node to go to next."""

    confidence_level = state["confidence_level"]
    retry_count      = state["retry_count"]

    if confidence_level in ("high", "medium"):
        logger.info("Evaluator: → proceeding to generator")
        return "generate"
    if confidence_level == "low" and retry_count <= MAX_RETRIES:
        logger.info(
            f"Evaluator: → retrying retrieval "
            f"({retry_count}/{MAX_RETRIES})"
        )
        return "retry"

    # no chunks or too many retries → cannot answer
    logger.info("Evaluator: → cannot answer")
    return "cannot_answer"
