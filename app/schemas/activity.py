from typing import List

from pydantic import BaseModel, Field
from app.schemas.core import Citation


class Activity(BaseModel):
    """An educational activity (role-play, experiment, etc.)"""
    period_id: str = Field(description="The period this activity belongs to")
    title: str = Field(description="Name of the activity")
    activity_type: str = Field(description="Type: e.g., Demonstration, Role Play, Experiment")
    duration_minutes: int = Field(description="How long the activity takes")
    materials_needed: List[str] = Field(description="List of materials required")
    instructions: str = Field(description="Step-by-step instructions for the teacher")
    success_criteria: List[str] = Field(description="How to know if the activity succeeded")
    safety_notes: str = Field(description="Any safety constraints or notes")
    source_refs: List[Citation] = Field(default_factory=list, description="Citations supporting the activity context")
