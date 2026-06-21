"""
What it does:
    1. takes generated answer and retrieved chunks
    2. matches answer claims to source chunks
    3. builds structured citation list
    4. formats final answer with citations appended
    5. writes final_answer sources citations to state
"""



import logging
from dataclasses import dataclass

from craster_rag.agents.state import RAGState

# logger
logger = logging.getLogger(__name__)


@dataclass
class Citation:

    index      : int
    title      : str
    page_number: int
    category   : str
    excerpt    : str
    score      : float


    def format(self) -> str:
        """Format citation as readable string."""
        return (
            f"[{self.index}] {self.title} — "
            f"Page {self.page_number}"
        )


def citation_verifier_agent(state: RAGState) -> RAGState:

    answer    = state["answer"]
    chunks    = state["chunks"]
    question  = state["question"]
    can_answer = state["can_answer"]

    # handle cannot answer case
    if not can_answer or not answer:
        logger.info("Verifier: cannot answer — returning fallback")
        final_answer = _build_cannot_answer_message(question)
        return {
            **state,
            "citations"   : [],
            "sources"     : [],
            "final_answer": final_answer,
        }
    logger.info(
        f"Verifier: verifying citations for "
        f"{len(chunks)} chunk(s)"
    )

    # build citations from chunks
    citations = _build_citations(chunks)
    # extract unique source titles
    sources = _extract_unique_sources(citations)
    # format final answer with citations
    final_answer = _format_final_answer(answer, citations)


    logger.info(
        f"Verifier: {len(citations)} citation(s) "
        f"from {len(sources)} source(s)"
    )

    return {
        **state,
        "citations"   : [_citation_to_dict(c) for c in citations],
        "sources"     : sources,
        "final_answer": final_answer,
    }



def _build_citations(chunks: list) -> list[Citation]:

    citations  = []
    seen_pages = set()   # track title+page combos
    for chunk in chunks:
        key = (chunk.title, chunk.page_number)
        if key in seen_pages:
            continue
        seen_pages.add(key)

        # build excerpt — first 150 chars of chunk
        excerpt = chunk.content[:150].strip()
        if len(chunk.content) > 150:
            excerpt += "..."

        citation = Citation(
            index       = len(citations) + 1,
            title       = chunk.title,
            page_number = chunk.page_number,
            category    = chunk.category,
            excerpt     = excerpt,
            score       = chunk.score,
        )
        citations.append(citation)

    return citations


def _extract_unique_sources(citations: list[Citation]) -> list[str]:
    seen    = set()
    sources = []

    for citation in citations:
        if citation.title not in seen:
            seen.add(citation.title)
            sources.append(citation.title)

    return sources


def _citation_to_dict(citation: Citation) -> dict:
    return {
        "index"      : citation.index,
        "title"      : citation.title,
        "page_number": citation.page_number,
        "category"   : citation.category,
        "excerpt"    : citation.excerpt,
        "score"      : citation.score,
        "formatted"  : citation.format(),
    }


def _build_cannot_answer_message(question: str) -> str:
    return f"""I was unable to find sufficient information in the
company policies to answer your question about:

"{question}"

**What to do next:**
- Contact HR directly at hr@craster.com
- Speak with your line manager
- Check the full Employee Handbook on the company intranet
- Raise a formal query through the HR portal

If you believe this information should be in our policy
documents please let the HR team know so they can review."""


def _format_final_answer(answer: str, citations: list[Citation],) -> str:

    """Format answer with citations appended."""
    if not citations:
        return answer


    # format citation list
    citation_lines = [
        citation.format()
        for citation in citations
    ]

    citations_text = "\n".join(citation_lines)

    final = f"""{answer}

---
**Sources:**
{citations_text}"""

    return final
