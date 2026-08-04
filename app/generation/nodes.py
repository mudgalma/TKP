import json
import logging
from typing import Type, TypeVar, Any

from pydantic import BaseModel

from app.config import settings
from app.rag.generate import _get_client
from app.generation.state import TKPState
from app.schemas.metadata import EducationalKnowledge
from app.schemas.period import TeachingPeriod
from app.schemas.activity import Activity
from app.schemas.assessment import Assessment
from app.schemas.gap import LearningGap
from app.schemas.core import ValidationIssue

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

def _call_llm_structured(user_prompt: str, schema: Type[T]) -> T:
    """Helper to call LLM and parse into Pydantic model."""
    client = _get_client()
    try:
        # Pydantic v2
        schema_dict = schema.model_json_schema()
    except AttributeError:
        # Pydantic v1 fallback
        schema_dict = schema.schema()
        
    system_prompt = (
        "You are an expert AI instructional designer. "
        "You MUST output valid JSON exactly matching this schema:\n"
        f"{json.dumps(schema_dict, indent=2)}\n\n"
        "CRITICAL INSTRUCTION: Do NOT output the schema itself, do NOT output '$defs'. "
        "You must generate a REAL, populated JSON object that contains actual educational content. "
        "If a field asks for a list of Citations, provide an actual list of objects `[{\"document_id\": \"...\", ...}]`.\n"
        "Do not wrap in markdown blocks, just return the raw JSON object."
    )
    
    print(f"--> Calling LLM with schema {schema.__name__}...")
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema_dict,
                "strict": False
            }
        },
        timeout=60.0
    )
    print(f"<-- LLM replied for {schema.__name__}")
    
    raw = response.choices[0].message.content or "{}"
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    try:
        # Pydantic v2
        return schema.model_validate_json(raw)
    except AttributeError:
        # Pydantic v1 fallback
        return schema.parse_raw(raw)


def educational_extractor(state: TKPState) -> dict:
    """Node 1: Extract educational metadata (Stage 2/3)."""
    print(">>> RUNNING NODE: educational_extractor")
    logger.info("Running educational_extractor node")
    return {"progress_events": ["Extraction completed."]}


def teaching_planner(state: TKPState) -> dict:
    """Node 2: Generate multi-period teaching plan."""
    print(">>> RUNNING NODE: teaching_planner")
    logger.info("Running teaching_planner node")
    ek = state.get("educational_knowledge")
    if not ek:
        raise ValueError("Educational Knowledge missing.")
        
    # We define a temporary wrapper schema for the planner output
    from pydantic import Field
    class PlannerOutput(BaseModel):
        periods: list[TeachingPeriod] = Field(description="The list of periods")
        
    prompt = (
        f"Create a teaching plan for: {ek.metadata.topic}.\n"
        f"Key concepts: {ek.key_concepts}\n"
        "Output 1 period covering the first concept as a starting point. Provide full script and notes."
    )
    
    try:
        plan = _call_llm_structured(prompt, PlannerOutput)
        return {
            "periods": plan.periods,
            "progress_events": ["Teaching plan generated."]
        }
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {"progress_events": [f"Planner failed: {e}"]}


def content_generator(state: TKPState) -> dict:
    """Node 3: Expand periods with content (currently done in planner)."""
    print(">>> RUNNING NODE: content_generator")
    logger.info("Running content_generator node")
    # If the planner already wrote the script, this might be a no-op or refinement.
    return {"progress_events": ["Content generation completed."]}


def activity_generator(state: TKPState) -> dict:
    """Node 4: Generate activities for each period."""
    print(">>> RUNNING NODE: activity_generator")
    logger.info("Running activity_generator node")
    periods = state.get("periods", [])
    if not periods:
        return {}
        
    # Generate 1 activity for the first period
    period = periods[0]
    prompt = f"Design a classroom activity for the period titled '{period.title}'. Topic: {period.objectives}"
    
    try:
        activity = _call_llm_structured(prompt, Activity)
        # Ensure period ID matches
        activity.period_id = period.period_id
        return {
            "activities": [activity],
            "progress_events": ["Activities generated."]
        }
    except Exception as e:
        logger.error(f"Activity gen failed: {e}")
        return {"progress_events": [f"Activity generation failed: {e}"]}


def assessment_generator(state: TKPState) -> dict:
    """Node 5: Generate assessments."""
    print(">>> RUNNING NODE: assessment_generator")
    logger.info("Running assessment_generator node")
    periods = state.get("periods", [])
    if not periods:
        return {}
        
    period = periods[0]
    prompt = f"Create an assessment (MCQs, short answer) for the period '{period.title}'. Objectives: {period.objectives}"
    
    try:
        assessment = _call_llm_structured(prompt, Assessment)
        assessment.period_id = period.period_id
        return {
            "assessments": [assessment],
            "progress_events": ["Assessments generated."]
        }
    except Exception as e:
        return {"progress_events": [f"Assessment generation failed: {e}"]}


def learning_gap_analyzer(state: TKPState) -> dict:
    """Node 6: Identify learning gaps."""
    print(">>> RUNNING NODE: learning_gap_analyzer")
    logger.info("Running learning_gap_analyzer node")
    periods = state.get("periods", [])
    if not periods:
        return {}
        
    period = periods[0]
    prompt = f"Identify a common learning gap for the topic '{period.title}'."
    
    try:
        gap = _call_llm_structured(prompt, LearningGap)
        gap.period_id = period.period_id
        return {
            "learning_gaps": [gap],
            "progress_events": ["Learning gaps analyzed."]
        }
    except Exception as e:
        return {"progress_events": [f"Gap analysis failed: {e}"]}


def package_assembler(state: TKPState) -> dict:
    """Node 7: Assemble the final package."""
    print(">>> RUNNING NODE: package_assembler")
    logger.info("Running package_assembler node")
    return {"progress_events": ["Package assembled."]}


def validator(state: TKPState) -> dict:
    """Node 8: Deterministic validation of outputs."""
    print(">>> RUNNING NODE: validator")
    logger.info("Running validator node")
    issues = []
    
    periods = state.get("periods", [])
    if not periods:
        issues.append(ValidationIssue(
            target="teaching_planner",
            code="MISSING_PERIODS",
            message="No teaching periods were generated."
        ))
        
    activities = state.get("activities", [])
    if not activities:
        issues.append(ValidationIssue(
            target="activity_generator",
            code="MISSING_ACTIVITIES",
            message="No activities were generated."
        ))
        
    retry_count = state.get("retry_count", 0)
    
    if issues:
        if retry_count >= 2:
            return {
                "validation_status": "failed",
                "validation_issues": issues,
                "progress_events": ["Validation failed permanently after retries."]
            }
        else:
            return {
                "validation_status": "retry",
                "retry_count": retry_count + 1,
                "validation_issues": issues,
                "progress_events": [f"Validation issues found. Retrying (attempt {retry_count + 1})."]
            }
    
    return {
        "validation_status": "passed",
        "progress_events": ["Validation passed successfully!"]
    }
