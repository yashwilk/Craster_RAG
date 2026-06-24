"""Wrong format → clear error returned immediately
Correct format → data passed to route handler"""


from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    """
    # user sends this to the API
    {
        "question": "how much maternity leave do I get?",
        "user_id" : "employee_123"   # optional
    }
    """

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Employee question about company policies",
        examples=["How much maternity leave do I get?"],
    )

    user_id: str = Field(
        default="anonymous",
        max_length=100,
        description="Optional employee identifier",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        """Strip whitespace and validate not empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace")
        return stripped


 
class AdminStatsRequest(BaseModel):
    """
    Request model for admin stats endpoint.
    Currently no fields needed.
    Placeholder for future auth fields.
    """
    pass



class ReindexRequest(BaseModel):
    """
    Request model for reindex endpoint.
 
    Attributes:
        source : optional specific file to reindex
                 if empty reindexes all documents
    """
    source: str = Field(
        default     = "",
        description = "Optional specific file path to reindex",
    )