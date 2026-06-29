"""
Graph structure:
    START
      ↓
    router              classifies question to category
      ↓
    query_rewriter      rewrites for better search
      ↓
    retriever           fetches relevant chunks
      ↓
    context_evaluator   scores context quality
      ↓ (conditional routing)
      ├── high/medium → generator → verifier → END
      ├── low + retries left → back to retriever
      └── low + no retries → verifier → END (cannot answer)


"""

import logging

from langgraph.graph import StateGraph, END

from craster_rag.agents.state import RAGState, create_initial_state
from craster_rag.agents.router_agent import router_agent
from craster_rag.agents.query_rewriter import query_rewriter_agent
from craster_rag.agents.retriever_agent import retriever_agent

from craster_rag.agents.context_evaluator import (
    context_evaluator_agent,
    should_retry,
)
from craster_rag.agents.answer_generator import answer_generator_agent
from craster_rag.agents.citation_verifier import citation_verifier_agent
from craster_rag.monitoring.langfuse_client import get_langfuse_handler

# logger
logger = logging.getLogger(__name__)


def _build_graph() -> StateGraph:

    """Build and compile the LangGraph pipeline. Called once at module level. Returns compiled graph ready to run."""

    graph = StateGraph(RAGState)

    # ── Add nodes ──────────────────────────────────────
    graph.add_node("router",     router_agent)
    graph.add_node("rewriter",   query_rewriter_agent)
    graph.add_node("retriever",  retriever_agent)
    graph.add_node("evaluator",  context_evaluator_agent)
    graph.add_node("generator",  answer_generator_agent)
    graph.add_node("verifier",   citation_verifier_agent)


    # entry point
    graph.set_entry_point("router")


    # router → rewriter → retriever → evaluator
    graph.add_edge("router",    "rewriter")
    graph.add_edge("rewriter",  "retriever")
    graph.add_edge("retriever", "evaluator")

    # evaluator has conditional routing
    # should_retry() decides which node to go to next

    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "generate"     : "generator",
            "retry"        : "retriever",   # retry retrieval
            "cannot_answer": "verifier",    # give up gracefully
        }
    )

    # generator → verifier → END
    graph.add_edge("generator", "verifier")
    graph.add_edge("verifier",  END)

    # compile the graph
    compiled = graph.compile()

    logger.info("RAG pipeline graph compiled successfully")
    return compiled

# ── Compiled graph ─────────────────────────────────────
# built once when module loads
_graph = _build_graph()

# Langfuse handler — None if not configured
_langfuse_handler = get_langfuse_handler()
 
 
def run_pipeline(question: str) -> RAGState:
    """Flow:
        router → rewriter → retriever → evaluator
        → (conditional) generator → verifier"""

    if not question.strip():
        raise ValueError("Question cannot be empty")
 
    logger.info(
        f"Pipeline: starting for '{question[:50]}...'"
    )
 
    initial_state = create_initial_state(question)
 
    try:
        # build invoke config
        # add Langfuse callback if available
        invoke_config = {}
        if _langfuse_handler:
            invoke_config["callbacks"] = [_langfuse_handler]
 
        # run the graph
        final_state = _graph.invoke(
            initial_state,
            config=invoke_config if invoke_config else None,
        )
 
        logger.info(
            f"Pipeline complete — "
            f"category={final_state.get('category')}, "
            f"confidence={final_state.get('confidence_level')}, "
            f"can_answer={final_state.get('can_answer')}"
        )
 
        return final_state
 
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
 
        # return safe error state
        return {
            **initial_state,
            "can_answer"      : False,
            "confidence_level": "none",
            "final_answer"    : (
                "I apologise but something went wrong. "
                "Please try again or contact HR directly."
            ),
            "sources"         : [],
            "citations"       : [],
            "error"           : str(e),
        }