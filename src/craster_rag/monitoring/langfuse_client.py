"""
Combines two approaches:
    1. CallbackHandler   automatic LangGraph node tracing
    2. Manual spans      custom business metrics per query

 CallbackHandler (automatic):
        every agent node latency
        input and output per node
        LLM token usage from Claude

    Manual spans (our code):
        user_id         which employee asked
        session_id      conversation grouping
        tags            category environment
        context_score   0.0 to 1.0
        context_relevance scored per query
        citation_validity 1.0 or 0.0
        can_answer      True or False
        errors          with full stack trace


"""


import logging
from contextlib import contextmanager
from typing import Any, Optional

from config import settings


# logger
logger = logging.getLogger(__name__)


def get_langfuse_handler():
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse CallbackHandler ready")
        return handler
    except ImportError:
        logger.warning("langfuse not installed — pip install langfuse")
        return None
    except Exception as e:
        logger.warning(f"Langfuse CallbackHandler setup failed: {e}")
        return None


class LangfuseMonitor:
    def __init__(self):
        self.client  = None
        self.enabled = False
        self._setup()

    def _setup(self) -> None:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.info(
                "Langfuse keys not set — manual monitoring disabled. "
                "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env"
            )
            return

        try:
            from langfuse import get_client
            self.client  = get_client()
            self.enabled = True
            logger.info("Langfuse manual monitoring enabled")
        except ImportError:
            logger.warning("langfuse not installed — pip install langfuse")
        except Exception as e:
            logger.warning(f"Langfuse setup failed: {e}")

    @contextmanager
    def trace(
        self,
        name      : str,
        input_data: Any,
        user_id   : str            = "anonymous",
        session_id: str            = "default",
        tags      : Optional[list] = None,
        metadata  : Optional[dict] = None,
    ):
        if not self.enabled or not self.client:
            yield None
            return
        try:
            from langfuse import propagate_attributes

            with self.client.start_as_current_observation(
                as_type  = "span",
                name     = name,
                input    = input_data,
                metadata = metadata or {},
            ) as root_trace:
                with propagate_attributes(
                    user_id    = user_id,
                    session_id = session_id,
                    tags       = tags or [],
                    metadata   = metadata or {},
                ):
                    yield root_trace
        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")
            yield None

    @contextmanager
    def span(
        self,
        name      : str,
        input_data: Any            = None,
        metadata  : Optional[dict] = None,
    ):
        if not self.enabled or not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_observation(
                as_type  = "span",
                name     = name,
                input    = input_data,
                metadata = metadata or {},
            ) as span:
                yield span
        except Exception as e:
            logger.warning(f"Langfuse span failed: {e}")
            yield None

    @contextmanager
    def generation(
        self,
        name      : str,
        model     : str,
        input_data: Any,
        metadata  : Optional[dict] = None,
    ):
        if not self.enabled or not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_observation(
                as_type  = "generation",
                name     = name,
                model    = model,
                input    = input_data,
                metadata = metadata or {},
            ) as generation:
                yield generation
        except Exception as e:
            logger.warning(f"Langfuse generation failed: {e}")
            yield None

    def score(
        self,
        name   : str,
        value  : float,
        comment: str = "",
    ) -> None:
        if not self.enabled or not self.client:
            return

        try:
            self.client.score_current_trace(
                name    = name,
                value   = value,
                comment = comment,
            )
        except Exception as e:
            logger.warning(f"Langfuse score failed: {e}")

    def flush(self) -> None:
        """
        Flush all pending traces to Langfuse.

        Call this:
            at end of short lived scripts
            before application shutdown
            in finally blocks

        Not needed for long running FastAPI server
        as Langfuse batches automatically.
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.flush()
            logger.debug("Langfuse flushed")
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")


# ── Single instances ───────────────────────────────────

monitor = LangfuseMonitor()
