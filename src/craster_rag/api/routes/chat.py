"""
Chat endpoint for the RAG API.
Endpoint:
    POST /api/v1/chat

Request:
    {
        "question": "how much maternity leave do I get?",
        "user_id" : "employee_123"
    }

Response:
    {
        "answer"          : "You are entitled to 52 weeks...",
        "final_answer"    : "You are entitled to 52 weeks...\n\nSources:\n[1]...",
        "sources"         : ["Maternity Policy"],
        "citations"       : [{...}],
        "category"        : "leave_family",
        "confidence_level": "high",
        "can_answer"      : true,
        "question"        : "how much maternity leave do I get?"
    }
"""


import logging
import uuid


from slowapi import Limiter
from fastapi import APIRouter, Request, HTTPException
from slowapi.util import get_remote_address


from craster_rag.agents.graph import run_pipeline
from craster_rag.api.models.request import ChatRequest
from craster_rag.api.models.response import ChatResponse, CitationResponse

from craster_rag.monitoring.langfuse_client import monitor
from config import settings


# logger
logger = logging.getLogger(__name__)


# router
router = APIRouter(prefix="/api/v1", tags=["chat"])


# rate limiter
limiter = Limiter(key_func=get_remote_address)



@router.post(
    "/chat",
    response_model = ChatResponse,
    summary        = "Ask a question about company policies",
    description    = "Runs the full multi-agent RAG pipeline and returns an answer with citations",
)
@limiter.limit("10/minute")
async def chat(
    http_request: Request,
    request     : ChatRequest,
) -> ChatResponse:
    """
    Receives employee question.
    Runs through multi-agent RAG pipeline:
        router → rewriter → retriever →
        evaluator → generator → verifier

    Returns answer with source citations.

    Args:
        http_request : FastAPI request (needed for rate limiter)
        request      : ChatRequest with question and user_id

    Returns:
        ChatResponse with answer citations and metadata
    """
    session_id = str(uuid.uuid4())

    logger.info(
        f"Chat request — "
        f"user={request.user_id}, "
        f"question='{request.question[:50]}...'"
    )

    with monitor.trace(
        name       = "hr_policy_query",
        input_data = {"question": request.question},
        user_id    = request.user_id,
        session_id = session_id,
        tags       = ["rag", "hr-policy", settings.environment.value],
        metadata   = {
            "environment"      : settings.environment.value,
            "app_version"      : settings.app_version,
        },
    ) as trace:

        try:
            # run full multi-agent pipeline
            result = run_pipeline(request.question)

            # build citations response
            citations = [
                CitationResponse(**citation)
                for citation in result.get("citations", [])
            ]

            monitor.score(
                name    = "context_relevance",
                value   = result.get("context_score", 0.0),
                comment = f"Category: {result.get('category', 'unknown')}",
            )

            monitor.score(
                name    = "can_answer",
                value   = 1.0 if result.get("can_answer") else 0.0,
                comment = f"Confidence: {result.get('confidence_level', 'none')}",
            )

            # update trace with final output
            if trace:
                trace.update(
                    output = {
                        "answer"           : result.get("answer", ""),
                        "can_answer"       : result.get("can_answer"),
                        "confidence_level" : result.get("confidence_level"),
                        "category"         : result.get("category"),
                        "citations_count"  : len(citations),
                    }
                )

            monitor.flush()

            return ChatResponse(
                answer           = result.get("answer", ""),
                final_answer     = result.get("final_answer", ""),
                sources          = result.get("sources", []),
                citations        = citations,
                category         = result.get("category", ""),
                confidence_level = result.get("confidence_level", "none"),
                can_answer       = result.get("can_answer", False),
                question         = request.question,
            )

        except ValueError as e:
            logger.warning(f"Invalid request: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            # log error to Langfuse trace
            if trace:
                trace.update(
                    level          = "ERROR",
                    status_message = str(e),
                )
            monitor.flush()

            logger.error(f"Pipeline error: {e}")
            raise HTTPException(
                status_code = 500,
                detail      = "Internal server error. Please try again.",
            )
