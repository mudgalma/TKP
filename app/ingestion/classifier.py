"""Document Classifier (Stage 2/3 Extraction).

Uses an LLM to extract educational metadata and key concepts
from the parsed document text during ingestion.
"""

from __future__ import annotations

import json
import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.exceptions import GenerationError
from app.rag.generate import _get_client

logger = logging.getLogger(__name__)

_CLASSIFIER_PROMPT = """\
You are an expert educational taxonomist and curriculum designer.
Analyze the following document excerpt and extract structured metadata.

Output ONLY valid JSON matching the following structure:
{
  "subject": "e.g., Biology",
  "grade_level": "e.g., Class 10",
  "difficulty": "e.g., Intermediate",
  "topic": "e.g., Photosynthesis",
  "chapter": "e.g., Chapter 6",
  "category": "e.g., NCERT Textbook",
  "language": "e.g., English",
  "learning_objectives": ["obj1", "obj2"],
  "prerequisites": ["prereq1"],
  "key_concepts": ["concept1", "concept2"],
  "definitions": [{"term": "...", "definition": "..."}],
  "formulae": ["formula1"],
  "keywords": ["kw1", "kw2"],
  "examples": ["ex1"],
  "common_misconceptions": ["misc1"]
}

If any field cannot be determined, use null or an empty list.
Do not wrap the JSON in markdown code blocks. Just output raw JSON.
"""

@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def classify_document(text_excerpt: str) -> dict:
    """Extract Stage 2/3 metadata from a document excerpt."""
    client = _get_client()

    user_prompt = f"<document>\n{text_excerpt}\n</document>"
    
    logger.info("Classifying document using model=%s", settings.llm_model)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            # OpenRouter / Llama 3.1 json mode
            response_format={"type": "json_object"}
        )
    except APIStatusError as exc:
        if exc.status_code == 429:
            raise APIConnectionError(request=exc.request) from exc
        if 400 <= exc.status_code < 500:
            raise GenerationError(f"Classifier API error {exc.status_code}: {exc.message}") from exc
        raise

    raw = response.choices[0].message.content or "{}"
    
    # Strip markdown block if model ignored the instruction
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse classifier JSON: %s\nRaw output: %s", exc, raw)
        return {}
