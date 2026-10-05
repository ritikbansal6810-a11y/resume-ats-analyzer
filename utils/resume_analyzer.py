"""
resume_analyzer.py
──────────────────
Parse and structure resume text using an LLM.
Returns a validated ResumeData dict.

Evidence rule: Only report what is explicitly present in the resume text.
"""

from __future__ import annotations

from typing import Any

from utils.llm_client import call_llm_json, LLMError


# ──────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────

RESUME_SYSTEM_PROMPT = """
You are a precise resume parser. Your ONLY job is to extract structured
information that is EXPLICITLY present in the resume text provided.

STRICT RULES:
1. Never invent, infer, or assume any information not directly stated.
2. If a section is absent from the resume, return an empty list [] or empty string "".
3. Do not claim the candidate has a skill unless it is written in the resume.
4. Do not add skills based on job titles or project names alone — only add
   them if the skill is explicitly named or clearly described.
5. Return ONLY valid JSON, no markdown fences, no explanations.

Return exactly this JSON structure:
{
  "candidate_name": "",
  "contact": {
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "website": ""
  },
  "summary": "",
  "education": [
    {
      "degree": "",
      "field": "",
      "institution": "",
      "year": "",
      "gpa": "",
      "details": ""
    }
  ],
  "experience": [
    {
      "title": "",
      "company": "",
      "duration": "",
      "start_date": "",
      "end_date": "",
      "responsibilities": [],
      "technologies_used": []
    }
  ],
  "internships": [
    {
      "title": "",
      "company": "",
      "duration": "",
      "responsibilities": [],
      "technologies_used": []
    }
  ],
  "projects": [
    {
      "name": "",
      "description": "",
      "technologies": [],
      "link": ""
    }
  ],
  "technical_skills": [],
  "soft_skills": [],
  "certifications": [
    {
      "name": "",
      "issuer": "",
      "year": ""
    }
  ],
  "languages": [],
  "tools_technologies": [],
  "achievements": [],
  "keywords": []
}
""".strip()


def analyze_resume(resume_text: str) -> dict[str, Any]:
    """
    Parse resume text into structured data using the LLM.

    Args:
        resume_text: Raw extracted text from the resume.

    Returns:
        A dict matching the RESUME_SYSTEM_PROMPT JSON schema.

    Raises:
        LLMError: If the LLM call fails or returns invalid JSON.
        ValueError: If resume_text is empty.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Please upload a valid resume.")

    user_prompt = f"""
Parse the following resume text and return the structured JSON.
Do NOT invent information. Only extract what is explicitly present.

RESUME TEXT:
{resume_text[:12000]}
""".strip()

    data = call_llm_json(RESUME_SYSTEM_PROMPT, user_prompt)

    # ── Validate and fill missing keys with safe defaults ──────────────────
    return _validate_resume_data(data)


def _validate_resume_data(data: Any) -> dict[str, Any]:
    """Ensure all expected keys exist with safe defaults."""
    if not isinstance(data, dict):
        raise LLMError("Resume analysis returned unexpected format (not a dict).")

    defaults: dict[str, Any] = {
        "candidate_name": "",
        "contact": {
            "email": "", "phone": "", "location": "",
            "linkedin": "", "github": "", "website": ""
        },
        "summary": "",
        "education": [],
        "experience": [],
        "internships": [],
        "projects": [],
        "technical_skills": [],
        "soft_skills": [],
        "certifications": [],
        "languages": [],
        "tools_technologies": [],
        "achievements": [],
        "keywords": [],
    }

    for key, default in defaults.items():
        if key not in data:
            data[key] = default
        elif isinstance(default, dict) and isinstance(data[key], dict):
            for sub_key, sub_default in default.items():
                data[key].setdefault(sub_key, sub_default)

    # Ensure list fields are actually lists
    list_fields = [
        "education", "experience", "internships", "projects",
        "technical_skills", "soft_skills", "certifications",
        "languages", "tools_technologies", "achievements", "keywords",
    ]
    for field in list_fields:
        if not isinstance(data[field], list):
            data[field] = []

    return data


def get_all_resume_text_skills(resume_data: dict) -> list[str]:
    """
    Return a flat list of all skills/technologies mentioned in the resume data.
    Used for evidence-based matching.
    """
    skills: list[str] = []
    skills.extend(resume_data.get("technical_skills", []))
    skills.extend(resume_data.get("soft_skills", []))
    skills.extend(resume_data.get("tools_technologies", []))
    skills.extend(resume_data.get("languages", []))

    # Also gather from projects and experience
    for project in resume_data.get("projects", []):
        skills.extend(project.get("technologies", []))

    for exp in resume_data.get("experience", []) + resume_data.get("internships", []):
        skills.extend(exp.get("technologies_used", []))

    # Deduplicate, preserve case
    seen: set[str] = set()
    unique: list[str] = []
    for s in skills:
        lower = s.lower().strip()
        if lower and lower not in seen:
            seen.add(lower)
            unique.append(s.strip())

    return unique
