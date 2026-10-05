"""
jd_analyzer.py
──────────────
Parse a Job Description (JD) into structured requirements using an LLM.
Returns a validated JDData dict.
"""

from __future__ import annotations

from typing import Any

from utils.llm_client import call_llm_json, LLMError


# ──────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = """
You are a precise job description parser. Extract structured requirements
from the job description text provided.

STRICT RULES:
1. Only extract information explicitly stated in the JD.
2. Distinguish between "required" and "preferred/nice-to-have" items.
3. Return ONLY valid JSON, no markdown fences, no explanations.

Return exactly this JSON structure:
{
  "job_title": "",
  "company": "",
  "location": "",
  "employment_type": "",
  "required_skills": [],
  "preferred_skills": [],
  "education_requirements": [],
  "experience_requirements": {
    "years": "",
    "level": "",
    "details": []
  },
  "tools_technologies": [],
  "certifications": [],
  "language_requirements": [],
  "responsibilities": [],
  "important_keywords": [],
  "other_requirements": [],
  "summary": ""
}
""".strip()


def analyze_jd(jd_text: str) -> dict[str, Any]:
    """
    Parse JD text into structured requirements using the LLM.

    Args:
        jd_text: Raw text of the job description.

    Returns:
        A dict matching the JD_SYSTEM_PROMPT JSON schema.

    Raises:
        LLMError: If the LLM call fails or returns invalid JSON.
        ValueError: If jd_text is empty.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text is empty.")

    user_prompt = f"""
Parse the following job description and return the structured JSON.
Only extract what is explicitly stated.

JOB DESCRIPTION:
{jd_text[:8000]}
""".strip()

    data = call_llm_json(JD_SYSTEM_PROMPT, user_prompt)
    return _validate_jd_data(data)


def _validate_jd_data(data: Any) -> dict[str, Any]:
    """Ensure all expected keys exist with safe defaults."""
    if not isinstance(data, dict):
        raise LLMError("JD analysis returned unexpected format (not a dict).")

    defaults: dict[str, Any] = {
        "job_title": "",
        "company": "",
        "location": "",
        "employment_type": "",
        "required_skills": [],
        "preferred_skills": [],
        "education_requirements": [],
        "experience_requirements": {"years": "", "level": "", "details": []},
        "tools_technologies": [],
        "certifications": [],
        "language_requirements": [],
        "responsibilities": [],
        "important_keywords": [],
        "other_requirements": [],
        "summary": "",
    }

    for key, default in defaults.items():
        if key not in data:
            data[key] = default
        elif isinstance(default, dict) and isinstance(data[key], dict):
            for sub_key, sub_default in default.items():
                data[key].setdefault(sub_key, sub_default)

    list_fields = [
        "required_skills", "preferred_skills", "education_requirements",
        "tools_technologies", "certifications", "language_requirements",
        "responsibilities", "important_keywords", "other_requirements",
    ]
    for field in list_fields:
        if not isinstance(data[field], list):
            data[field] = []

    return data


def get_all_jd_requirements(jd_data: dict) -> list[str]:
    """
    Return a flat deduplicated list of all skill/tool requirements.
    Combines required, preferred and tools/technologies.
    """
    items: list[str] = []
    items.extend(jd_data.get("required_skills", []))
    items.extend(jd_data.get("preferred_skills", []))
    items.extend(jd_data.get("tools_technologies", []))
    items.extend(jd_data.get("certifications", []))
    items.extend(jd_data.get("language_requirements", []))

    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        lower = str(item).lower().strip()
        if lower and lower not in seen:
            seen.add(lower)
            unique.append(str(item).strip())

    return unique
