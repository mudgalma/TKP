from typing import List

from pydantic import BaseModel, Field
from app.schemas.core import Citation


class LearningGap(BaseModel):
    """A potential student misconception and how to address it."""
    period_id: str = Field(description="The period this gap is likely to occur in")
    misconception: str = Field(description="The specific misconception")
    diagnostic_question: str = Field(description="A quick question to check if students have this misconception")
    severity: str = Field(description="'Low', 'Medium', or 'High'")
    remedial_action: str = Field(description="How the teacher should correct it")
    source_refs: List[Citation] = Field(default_factory=list, description="Citations related to this gap")
