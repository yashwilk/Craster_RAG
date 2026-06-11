import logging
from typing import List

from craster_rag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for the Claude API from search results and a user question."""

    NO_CONTEXT_TEMPLATE = """QUESTION:
{question}

No relevant procedures were found for this question.
Please inform the user that you could not find relevant
information in the company procedures and suggest they
contact their manager or check the wiki directly."""

    RAG_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer based ONLY on the context provided above
- If the answer involves steps format as a numbered list
- Always mention which procedure the answer comes from
- Be clear and concise"""

    def build(self, question: str, results: List[SearchResult]) -> str:
        if not results:
            logger.warning("No search results — building fallback prompt")
            return self._build_no_context_question(question)

        context = self._build_context(results)
        prompt = self.RAG_TEMPLATE.format(context=context, question=question)

        logger.debug(
            f"Prompt built — "
            f"chunks={len(results)}, "
            f"length={len(prompt)}"
        )

        return prompt

    def _build_context(self, results: List[SearchResult]) -> str:
        context_parts = []

        for i, result in enumerate(results, 1):
            score_pct = round(result.score * 100)
            part = (
                f"[{i}] {result.content}\n"
                f"    Source: {result.title} | "
                f"Relevance: {score_pct}%"
            )
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def _build_no_context_question(self, question: str) -> str:
        return self.NO_CONTEXT_TEMPLATE.format(question=question)
