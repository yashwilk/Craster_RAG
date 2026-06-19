"""
    1. builds a prompt with all category descriptions
    2. asks Claude which category fits the question
    3. parses Claude response to extract category
    4. writes category to RAGState
    5. falls back to general if parsing fails

"""


import logging
import re

import anthropic

from config import settings, CATEGORY_DESCRIPTIONS, DocumentCategory
from craster_rag.agents.state import RAGState

# logger
logger = logging.getLogger(__name__)

# ── Anthropic client ───────────────────────────────────
_client=anthropic.Anthropic(api_key=settings.anthropic_api_key)

VALID_CATEGORIES=[cat.value for cat in DocumentCategory]

def _build_routing_prompt(question:str)->str:
    #  Build the routing prompt with all category descriptions.
    categoryes_text=""
    for category,description in CATEGORY_DESCRIPTIONS.items():
        if category==DocumentCategory.GENERAL:
            continue
        categoryes_text+=f"{category}:{description}\n"
    prompt=f"""Classify this employee question into exactly one category.
 CATEGORIES:
{categoryes_text}
 Respond with ONLY the category name from this list:
{", ".join(VALID_CATEGORIES)}

Question: {question}

If the question clearly spans multiple categories or
does not fit any category respond with: general"""

    return prompt



def _build_system_prompt()->str:
    return """You are a document router for Craster company HR policies.
Your job is to classify employee questions into exactly one category.
Respond with ONLY the category name. Nothing else. No explanation."""


def _parse_category(raw_response:str)->str:
    cleaned=raw_response.strip().lower()

    if cleaned in VALID_CATEGORIES:
        return cleaned

    for category in VALID_CATEGORIES:
        if category in cleaned:
            return category

    cleaned_spaces=cleaned.replace(" ","_")
    if cleaned_spaces in VALID_CATEGORIES:
        return cleaned_spaces

    # fallback to general
    logger.warning(
        f"Could not parse category from: '{raw_response}'. "
        f"Falling back to 'general'"
    )
    return DocumentCategory.GENERAL.value



def _classify_question(question):
    prompt=_build_routing_prompt(question)

    response=_client.messages.create(

        model=settings.anthropic_model,
        max_tokens=50, # category name is short
        system=_build_system_prompt(),
        messages=[{"role":"user","content": prompt}]
    )
    raw_response = response.content[0].text.strip().lower()
    category     = _parse_category(raw_response)

    return category


def router_agent(state:RAGState)->RAGState:
    """
       Reads:
        state["question"]   user question

    Writes:
        state["category"]   classified category
    """
    question = state["question"]

    logger.info(f"Router: classifying question '{question[:50]}...'")

    try:
        category = _classify_question(question)
        logger.info(f"Router: classified as '{category}'")

    except Exception as e:
        logger.error(f"Router failed: {e}")
        category = DocumentCategory.GENERAL.value

    return {
        **state,
        "category": category,
    }
