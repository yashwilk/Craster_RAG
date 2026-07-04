import logging
import time
from dataclasses import dataclass
from typing import List

import anthropic
import mlflow

from config import settings
from craster_rag.generation.prompt_builder import PromptBuilder
from craster_rag.retrieval.vector_store import SearchResult
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    sources: List[str]
    query: str
    latency: float
    input_tokens: int
    output_tokens: int

    def __repr__(self) -> str:
        return (
            f"Answer("
            f"chars={len(self.text)}, "
            f"sources={len(self.sources)}, "
            f"latency={self.latency:.2f}s)"
        )


class ClaudeClient:
    """Wrapper around the Anthropic Claude API.
    Handles:
        building prompts from chunks and question
        calling Claude API with retries
        parsing response
        logging to MLflow
    """

    SYSTEM_PROMPT = """You are a helpful assistant for Craster company.
You answer questions about internal company procedures and processes.

Rules:
- Only answer based on the provided context
- If the context does not contain the answer say:
  "I could not find information about that in the procedures.
   Please contact your manager or check the wiki directly."
- Always mention which procedure the answer comes from
- Be concise and clear
- Format steps as numbered lists when applicable
- Never make up information not in the context"""

    def __init__(
        self,
        model      : str = "",
        max_tokens : int = 0,
    ):
        self.model      = model      or settings.anthropic_model
        self.max_tokens = max_tokens or settings.anthropic_max_tokens

        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

        self._prompt_builder = PromptBuilder()

        logger.info(
            f"ClaudeClient initialised — "
            f"model='{self.model}', "
            f"max_tokens={self.max_tokens}"
        )

    def answer(
        self,
        question: str,
        results : list[SearchResult],
    ) -> Answer:
        """
        Generate an answer using retrieved chunks.

        Steps:
            1. PromptBuilder formats the prompt
            2. Claude API generates the answer
            3. Answer object built with metadata
            4. Metrics logged to MLflow

        Args:
            question : user question as plain text
            results  : list of SearchResult from searcher

        Returns:
            Answer object with text and source citations
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        prompt = self._prompt_builder.build(question, results)

        start_time = time.time()
        response   = self._call_claude(prompt)
        latency    = time.time() - start_time

        answer_text = response.content[0].text

        sources = list(set(
            result.title for result in results
        ))

        answer = Answer(
            text          = answer_text,
            sources       = sources,
            query         = question,
            latency       = round(latency, 2),
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
        )

        self._log_to_mlflow(answer)

        logger.info(
            f"Answer generated — "
            f"latency={answer.latency}s, "
            f"tokens={answer.input_tokens + answer.output_tokens}"
        )

        return answer

    @retry(
        stop    = stop_after_attempt(3),
        wait    = wait_exponential(min=1, max=10),
        retry   = retry_if_exception_type(Exception),
        reraise = True,
    )
    def _call_claude(self, prompt: str) -> anthropic.types.Message:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response

        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {e}")
            raise

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _log_to_mlflow(self, answer: Answer) -> None:
        try:
            mlflow.log_metrics({
                "claude_latency_seconds" : answer.latency,
                "claude_input_tokens"    : answer.input_tokens,
                "claude_output_tokens"   : answer.output_tokens,
                "claude_total_tokens"    : answer.input_tokens + answer.output_tokens,
                "query_length_chars"     : len(answer.query),
                "answer_length_chars"    : len(answer.text),
                "num_sources"            : len(answer.sources),
            })
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
