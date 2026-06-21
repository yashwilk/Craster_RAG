"""Generates an answer using Claude based on
retrieved chunks and confidence level.
hat it does:
    1. reads chunks and question from state
    2. builds context from chunks with page numbers
    3. selects appropriate prompt based on confidence
    4. calls Claude API
    5. writes answer to state


    Confidence aware generation:
    high    → answer directly and confidently
    medium  → answer but add caveat to verify with HR

"""


import logging
import time

import anthropic
import mlflow

from craster_rag.agents.state import RAGState
from config import settings

# logger
logger = logging.getLogger(__name__)

# Anthropic client — created once reused every call
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ── System Prompt ──────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful HR policy assistant for Craster company.
You answer employee questions about company policies and procedures.

Rules:
- Answer ONLY based on the provided context
- Never make up information not in the context
- Always mention the policy name the answer comes from
- Format steps as numbered lists when applicable
- Be clear concise and professional
- Use plain English not legal jargon
- If you mention a specific entitlement include the exact number"""



def answer_generator_agent(state: RAGState) -> RAGState:
    chunks           = state["chunks"]
    question         = state["question"]
    confidence_level = state["confidence_level"]

    logger.info(
        f"Generator: generating answer "
        f"with {len(chunks)} chunks "
        f"at {confidence_level} confidence"
    )

    try:
        answer = _generate_answer(
            question         = question,
            chunks           = chunks,
            confidence_level = confidence_level,
        )
        logger.info(
            f"Generator: answer generated "
            f"({len(answer)} chars)"
        )

    except Exception as e:
        logger.error(f"Generator failed: {e}")
        answer = (
            "I apologise but I was unable to generate an answer. "
            "Please contact HR directly for assistance."
        )

    return {
        **state,
        "answer": answer,
    }

def _generate_answer(
    question        : str,
    chunks          : list,
    confidence_level: str,
) -> str:

    """
    Call Claude to generate answer from context.
    """

    # build context from chunks
    context = _build_context(chunks)
    # build prompt based on confidence level
    prompt = _build_prompt(
        question         = question,
        context          = context,
        confidence_level = confidence_level,
    )
    # call Claude with timing
    start_time = time.time()
    response = _client.messages.create(
        model      = settings.anthropic_model,
        max_tokens = settings.anthropic_max_tokens,
        system     = SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": prompt}
        ],
    )

    latency = round(time.time() - start_time, 2)


    # log metrics to MLflow
    _log_metrics(response, latency)

    return response.content[0].text.strip()



def _build_context(chunks: list) -> str:
    """
    Format retrieved chunks into context string.

    Each chunk becomes a numbered block with
    source document and page number. """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
         part = (
            f"[{i}] {chunk.content}\n"
            f"    Source: {chunk.title} | "
            f"Page: {chunk.page_number} | "
            f"Relevance: {round(chunk.score * 100)}%"
        )

         context_parts.append(part)


    return "\n\n".join(context_parts)



def _build_prompt(
    question        : str,
    context         : str,
    confidence_level: str,
) -> str:

    """
    Build prompt based on confidence level.

    High confidence:
        answer directly and confidently

    Medium confidence:
        answer but recommend verifying with HR """



    if confidence_level == "high":
        instruction = (
            "Answer the question directly and confidently "
            "based on the policy information above."
        )
    else:
        # medium confidence
        instruction = (
            "Answer the question based on the available "
            "policy information above. Since the information "
            "may not fully cover this topic, end your answer with: "
            "'For complete guidance please confirm with HR directly.'"
        )


    prompt = f"""POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

INSTRUCTION:
{instruction}

Include the policy name and page number when referencing specific information."""

    return prompt


def _log_metrics(response: anthropic.types.Message, latency: float) -> None:
    """
    Log generation metrics to MLflow.

    Args:
        response : Claude API response
        latency  : generation time in seconds
    """
    try:
        mlflow.log_metrics({
            "generation_latency_seconds" : latency,
            "generation_input_tokens"    : response.usage.input_tokens,
            "generation_output_tokens"   : response.usage.output_tokens,
            "generation_total_tokens"    : (
                response.usage.input_tokens +
                response.usage.output_tokens
            ),
        })
    except Exception as e:
        # MLflow logging never crashes pipeline
        logger.warning(f"MLflow logging failed: {e}")
