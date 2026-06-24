"""
# frontend always gets same structure
ChatResponse(
    answer, final_answer, sources,
    citations, category, confidence_level,
    can_answer, question
)

"""


from pydantic import BaseModel, Field


class CitationResponse(BaseModel):
    """
  A single citation in the response."""

    index      : int   = Field(description="Citation number")
    title      : str   = Field(description="Document title")
    page_number: int   = Field(description="Page number in PDF")
    category   : str   = Field(description="Document category")
    excerpt    : str   = Field(description="Relevant excerpt")
    score      : float = Field(description="Similarity score 0-1")
    formatted  : str   = Field(description="Formatted citation string")



class ChatResponse(BaseModel):
    answer          : str                    = Field(
        description = "Generated answer text"
    )
    final_answer    : str                    = Field(
        description = "Answer with formatted citations"
    )
    sources         : list[str]              = Field(
        default     = [],
        description = "Source document titles"
    )
    citations       : list[CitationResponse] = Field(
        default     = [],
        description = "Detailed citation objects"
    )
    category        : str                    = Field(
        description = "Detected question category"
    )
    confidence_level: str                    = Field(
        description = "high medium low or none"
    )
    can_answer      : bool                   = Field(
        description = "True if answer was generated"
    )
    question        : str                    = Field(
        description = "Original question echoed back"
    )



class HealthResponse(BaseModel):
    """
    Response model for health check endpoint."""

    status : str = Field(description="ok or error")
    version: str = Field(description="App version")

class AdminStatsResponse(BaseModel):
    """
    Response model for admin stats endpoint.

    Attributes:
        total_chunks   : total chunks in vector store
        unique_sources : number of unique documents
        categories     : chunk count per category
    """
    total_chunks   : int        = Field(description="Total chunks stored")
    unique_sources : int        = Field(description="Unique source documents")
    categories     : dict       = Field(description="Chunks per category")

class ErrorResponse(BaseModel):
    """
    Response model for error responses."""
    error : str = Field(description="Error message")
    detail: str = Field(default="", description="Error detail")
