from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata about the document context."""
    subject: str = Field(description="The primary subject area (e.g., Biology, History)")
    grade_level: str = Field(description="Target grade level (e.g., 10th Grade, University)")
    difficulty: str = Field(description="Overall difficulty (e.g., Beginner, Advanced)")
    topic: str = Field(description="The specific topic covered")
    chapter: Optional[str] = Field(default=None, description="Chapter name or number")
    category: str = Field(description="Category (e.g., STEM, Humanities)")
    language: str = Field(default="English", description="Primary language")


class Definition(BaseModel):
    term: str
    definition: str

class EducationalKnowledge(BaseModel):
    """The foundational extracted knowledge blueprint."""
    metadata: DocumentMetadata
    learning_objectives: List[str] = Field(default_factory=list, description="High-level learning goals")
    prerequisites: List[str] = Field(default_factory=list, description="What students should already know")
    key_concepts: List[str] = Field(default_factory=list, description="Core concepts taught in the text")
    definitions: List[Definition] = Field(default_factory=list, description="Important definitions")
    formulae: List[str] = Field(default_factory=list, description="Important mathematical or chemical formulae")
    keywords: List[str] = Field(default_factory=list, description="Vocabulary words")
    examples: List[str] = Field(default_factory=list, description="Real-world examples mentioned")
    applications: List[str] = Field(default_factory=list, description="Practical applications of the concepts")
    misconceptions: List[str] = Field(default_factory=list, alias="common_misconceptions", description="Common errors or misunderstandings to watch out for")
    
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
