"""
Defines the RAGState that flows through every agent
in the LangGraph pipeline.

    A shared data container.
    Every agent reads from it and writes to it.
    LangGraph passes it automatically between agents.


"""


from typing import TypedDict

class RAGState(TypedDict):
    # ── Input ───────────────────────────────────────────
    # set by the user before pipeline starts
    question          : str

    # ── Router Agent Output ─────────────────────────
    # which category does this question belong to?
    category          : str

    # ── Query Rewriter Output ───────────────────────────
    # improved version of question for better search
    rewritten_query   : str

    # ── Retriever Agent Output ──────────────────────────
    # chunks retrieved from Supabase
    chunks            : list          # list of SearchResult objects
    vector_scores     : list[float]   # scores from vector search
    bm25_scores       : list[float]   # scores from BM25 search
    hybrid_scores     : list[float]   # combined RRF scores


    # ── Context Evaluator Output ────────────────────────
    # how good is the retrieved context?
    context_score     : float         # 0.0 to 1.0
    confidence_level  : str           # high medium low none


    # ── Answer Generator Output ─────────────────────────
    answer            : str           # raw answer from Claude

    # ── Citation Verifier Output ────────────────────────
    citations         : list[dict]    # verified source citations
    final_answer      : str           # answer + formatted citations
    sources           : list[str]     # unique source document titles


    # ── Pipeline Control ────────────────────────────────
    can_answer        : bool          # False if cannot answer
    retry_count       : int           # how many retrieval retries
    error             : str           # error message if failed


def create_initial_state(question: str) -> RAGState:
    """ Sets all fields to safe empty defaults.
    Only question is set from user input."""

    return RAGState(
        question          = question.strip(),
        category          = "",
        rewritten_query   = "",
        chunks            = [],
        vector_scores     = [],
        bm25_scores       = [],
        hybrid_scores     = [],
        context_score     = 0.0,
        confidence_level  = "",
        answer            = "",
        citations         = [],
        final_answer      = "",
        sources           = [],
        can_answer        = True,
        retry_count       = 0,
        error             = "",
    )
