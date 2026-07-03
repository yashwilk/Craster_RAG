"""First agent in the pipeline — prompt injection protection.
Two layer defence:
    Layer 1 — pattern matching (instant, free)
        detects known injection phrases immediately
        catches "ignore previous instructions",
        "you are now DAN", "forget you are", etc.

    Layer 2 — Claude classification (cheap, robust)
        for questions that pass pattern check
        Claude reads the question and classifies it
        as legitimate or an attack
        catches novel/creative injection attempts
        that pattern matching misses
        """

import logging
import re
from typing import Optional

import anthropic

from craster_rag.agents.state import RAGState
from config import settings


# logger
logger = logging.getLogger(__name__)

# ── Anthropic client (lazy) ────────────────────────────
_client: Optional[anthropic.Anthropic] = None

def _get_client() -> anthropic.Anthropic:
    """Lazily create and cache the Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client

# Layer 1 fast pattern matching


INJECTION_PATTERNS = [
    # instruction override attacks
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",

    # identity override attacks
    r"you\s+are\s+now\s+\w+",       # "you are now DAN"
    r"act\s+as\s+(if\s+you\s+are)",  # "act as if you are"
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
    r"forget\s+you\s+are",
    r"you\s+are\s+no\s+longer",

    # system prompt extraction attacks
    r"(show|reveal|print|display|output|repeat)\s+(your\s+)?(system\s+prompt|instructions|prompt)",
    r"what\s+(are|were)\s+your\s+instructions",
    r"what\s+is\s+your\s+(system\s+prompt|prompt)",

    # known jailbreak names
    r"\bDAN\b",                      # Do Anything Now
    r"\bAIM\b",                      # Always Intelligent and Machiavellian
    r"\bjailbreak\b",
    r"developer\s+mode",
    r"sudo\s+mode",
    r"god\s+mode",

    # data exfiltration attempts
    r"(employee|staff|salary|wage|payroll)\s+(data|records|list|database)",
    r"(personal|private|confidential)\s+(data|information|details)\s+of",
    r"(show|list|tell me)\s+(all\s+)?(employee|staff)\s+",

    # prompt injection via payload
    r"\[INST\]",                     # LLaMA instruction tag
    r"<\|im_start\|>",               # ChatML format
    r"<\|system\|>",
    r"Human:\s*\n",                  # conversation injection
]

# compile patterns once for efficiency
_COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in INJECTION_PATTERNS
]

HR_TOPICS = """
- Leave policies (maternity, paternity, parental, compassionate, sick leave)
- Health and safety at work
- Employment terms, contracts, redundancy, grievance, disciplinary
- Data protection, GDPR, whistleblowing
- Workplace conduct, harassment, anti-bribery
- Benefits, pay, expenses, pension
- Equality and diversity
- Company handbook and general procedures
"""

def guard_agent(state: RAGState) -> RAGState:
    question = state["question"]

    logger.info(f"Guard: checking question '{question[:50]}...'")

    pattern_hit = _check_patterns(question)

    if pattern_hit:
        logger.warning(
            f"Guard: injection detected (pattern) — "
            f"'{question[:80]}'"
        )
        return _block(state, f"Pattern match: {pattern_hit}")

    try:
        is_legitimate = _classify_with_claude(question)

        if not is_legitimate:
            logger.warning(
                f"Guard: injection detected (Claude) — "
                f"'{question[:80]}'"
            )
            return _block(state, "Claude classifier: off-topic or malicious")

    except Exception as e:
        # because of a transient API issue
        logger.warning(f"Guard: Claude classification failed ({e}) — allowing")

    logger.debug(f"Guard: question passed — '{question[:50]}...'")
    return state


def _check_patterns(question: str) -> Optional[str]:
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        if pattern.search(question):
            return INJECTION_PATTERNS[i]
    return None


def _classify_with_claude(question: str) -> bool:
    prompt = f"""You are a security classifier for an HR policy chatbot.
Determine if this is a legitimate employee question about HR policies.

Legitimate HR topics:
{HR_TOPICS}

Question: "{question}"

Reply with only YES (legitimate HR question) or NO (off-topic or malicious).
Single word answer only."""

    response = _get_client().messages.create(
        model      = settings.anthropic_model,
        max_tokens = 5,     # only need YES or NO
        messages   = [{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip().upper()
    return answer.startswith("Y")   # YES → legitimate


def _block(state: RAGState, reason: str) -> RAGState:
    return {
        **state,
        "can_answer"      : False,
        "confidence_level": "none",
        "error"           : (
            f"Question blocked by security guard. "
            f"Reason: {reason}"
        ),
    }
