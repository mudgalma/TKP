from typing import List

from pydantic import BaseModel, Field
from app.schemas.core import Citation


class TeachingPeriod(BaseModel):
    """The core teaching period containing scripts and notes."""
    period_id: str = Field(description="Unique ID for the period, e.g., 'period_1'")
    title: str = Field(description="Title of the period/lesson")
    duration_minutes: int = Field(description="Duration in minutes (e.g., 40)")
    objectives: List[str] = Field(description="Specific learning objectives for this period")
    entry_ticket: str = Field(description="Warm-up activity or question")
    teacher_script: str = Field(description="Detailed script for the teacher to follow")
    blackboard_notes: str = Field(description="Bullet points to write on the board")
    checkpoint_questions: List[str] = Field(description="Quick questions to check understanding mid-lesson")
    exit_ticket: str = Field(description="Closing activity or question")
    homework: str = Field(description="Homework assignment")
    mentor_moment: str = Field(description="A motivational anecdote or real-world connection")
    source_refs: List[Citation] = Field(default_factory=list, description="Citations supporting the factual content")
