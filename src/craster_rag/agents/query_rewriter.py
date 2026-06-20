"""
Rewrites vague user questions into better search queries.
    1.takes original question and detected category
    2. asks Claude to rewrite for better search
    3. Claude returns optimised search query
    4. writes rewritten_query to RAGState
"""

"""
Replaced claude with ollama Qwen"""


import logging
import requests
import anthropic
from config import settings
from craster_rag.agents.state import RAGState

# logger
logger = logging.getLogger(__name__)

#  _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


CATEGORY_CONTEXT: dict[str, str] = {
    "leave_family"      : "maternity paternity parental leave entitlement policy",
    "health_safety"     : "health safety workplace policy procedure",
    "employment"        : "employment terms conditions procedure policy",
    "data_compliance"   : "data protection GDPR compliance whistleblowing policy",
    "conduct"           : "workplace conduct behaviour harassment bullying policy",
    "rewards_benefits"  : "employee benefits rewards compensation expenses pension",
    "equality_diversity": "equality diversity inclusion discrimination policy",
    "company_general"   : "company policy handbook procedure general information",
    "general"           : "HR policy employee rights procedure",
}


def query_rewriter_agent(state: RAGState) -> RAGState:

    question = state["question"]
    category = state["category"]
    if not settings.enable_query_rewriting:
        logger.info("Query rewriting disabled — using original question")
        return {
            **state,
            "rewritten_query": question,
        }

    # check if Ollama is running
    if not _ollama_is_running():
        logger.warning(
            "Ollama not running — using original question. "
            "Start with: ollama serve"
        )
        return {**state, "rewritten_query": question}

    logger.info(
        f"Rewriter: rewriting for category '{category}'"
    )

    try:
        rewritten = _rewrite_with_qwen(question, category)
        logger.info(
            f"Rewriter: "
            f"'{question[:40]}' "
            f"→ '{rewritten[:40]}'"
        )

    except Exception as e:
        # never let rewriting crash the pipeline
        # fall back to original question
        logger.warning(
            f"Query rewriting failed: {e}. "
            f"Using original question."
        )
        rewritten = question

    return {
        **state,
        "rewritten_query": rewritten,
    }



def _ollama_is_running() -> bool:
    try:
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=3,
        )
        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        return False

    except Exception:
        return False



def _clean_qwen_response(response: str) -> str:
    if response.lower().startswith("query:"):
        response = response[6:].strip()
    # remove quotes
    response = response.strip('"').strip("'")
    # take only first line if multiple lines
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if lines:
        response = lines[0]

    return response.strip()



def _build_rewrite_prompt(question: str, category: str) -> str:
    #  Build the rewrite prompt with category context.

    context = CATEGORY_CONTEXT.get(category, "HR policy")
    prompt = f"""You are a search query optimiser for HR policy documents.
Rewrite the question as a concise search query.

Rules:
- Return ONLY the rewritten query
- Maximum 15 words
- Use specific HR terminology
- No explanation or preamble

Context: {context}

Examples:
Question: what about my leave?
Query: employee annual leave entitlement days policy

Question: can I work from home?
Query: flexible remote working policy employee rights home office

Question: what if I witness something wrong?
Query: whistleblowing reporting misconduct concern procedure

Now rewrite this:
Question: {question}
Query:"""

    return prompt


def _build_system_prompt():
    return """You are a search query optimiser for an HR policy document system.
Your job is to rewrite employee questions into better search queries.

Rules:
- Return ONLY the rewritten query
- No explanation or preamble
- Keep it concise 10-20 words maximum
- Use specific HR and policy terminology
- Include relevant keywords for the topic
- Make it suitable for semantic document search"""


def _rewrite_with_qwen(question: str, category: str) -> str:
    """Ask Claude to rewrite the question for better search.
    REPLACED WITH= Send question to local Ollama Qwen model for rewriting.Uses Ollama REST API at localhost:11434.
"""
    prompt = _build_rewrite_prompt(question, category)

    """  response = _client.messages.create(
        model=settings.anthropic_model,
        max_tokens=100,
        system=_build_system_prompt(),
        messages=[{"role": "user", "content": prompt}]
    )"""

    payload = {
        "model"  : settings.ollama_model,
        "prompt" : prompt,
        "stream" : False,       # get complete response at once
        "options": {
            "temperature" : 0.1,    # low temp for consistent output
            "num_predict" : 50,     # max tokens in response
            "top_p"       : 0.9,
        },
    }

    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json    = payload,
        timeout = 30,           # 30 second timeout
    )
    response.raise_for_status()
    result    = response.json()
    rewritten = result.get("response", "").strip()

    # clean up common Qwen response patterns
    rewritten = _clean_qwen_response(rewritten)

    # safety check
    if not rewritten:
        logger.warning("Qwen returned empty response")
        return question

    return rewritten
