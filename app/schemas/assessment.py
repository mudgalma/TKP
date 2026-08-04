from typing import List, Optional

from pydantic import BaseModel, Field
from app.schemas.core import Citation


class AssessmentQuestion(BaseModel):
    question_type: str = Field(description="'MCQ', 'ShortAnswer', 'LongAnswer', 'Numerical'")
    question_text: str = Field(description="The question itself")
    options: Optional[List[str]] = Field(default=None, description="For MCQs: List of options")
    correct_answer: str = Field(description="The correct answer or ideal response")
    rubric: str = Field(description="Grading rubric or success criteria for this question")
    source_refs: List[Citation] = Field(default_factory=list, description="Citations from the text that hold the answer")


class Assessment(BaseModel):
    """An assessment designed for a specific period."""
    period_id: str = Field(description="The period this assessment belongs to")
    questions: List[AssessmentQuestion] = Field(description="List of questions")
