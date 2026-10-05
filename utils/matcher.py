"""
matcher.py
──────────
Evidence-based skill matching between JD requirements and resume content.

Statuses:
  FOUND        – explicit evidence in resume for this requirement
  PARTIAL      – related evidence exists but incomplete
  NOT_FOUND    – no evidence in resume
  NOT_VERIFIABLE – cannot be confirmed from resume text alone

Important: never fabricate resume evidence.
"""

from __future__ import annotations

from typing import Any

from utils.llm_client import call_llm_json, LLMError


# ──────────────────────────────────────────────────────────────────────────
# Status constants
# ──────────────────────────────────────────────────────────────────────────

STATUS_FOUND = "FOUND"
STATUS_PARTIAL = "PARTIAL"
STATUS_NOT_FOUND = "NOT FOUND"
STATUS_NOT_VERIFIABLE = "NOT VERIFIABLE"

VALID_STATUSES = {STATUS_FOUND, STATUS_PARTIAL, STATUS_NOT_FOUND, STATUS_NOT_VERIFIABLE}

STATUS_COLORS = {
    STATUS_FOUND: "success",
    STATUS_PARTIAL: "warning",
    STATUS_NOT_FOUND: "error",
    STATUS_NOT_VERIFIABLE: "info",
}

STATUS_EMOJI = {
    STATUS_FOUND: "✅",
    STATUS_PARTIAL: "🔶",
    STATUS_NOT_FOUND: "❌",
    STATUS_NOT_VERIFIABLE: "🔍",
}


# ──────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────

MATCHER_SYSTEM_PROMPT = """
You are an evidence-based resume-to-job-description matcher.
Your ONLY job is to check whether each JD requirement is explicitly
supported by the resume text.

CRITICAL RULES:
1. Use ONLY information present in the resume text.
2. NEVER invent, assume, or infer evidence.
3. Do NOT say a candidate "lacks" a skill — only say the evidence was not found.
4. Use exactly these statuses:
   - FOUND: resume explicitly mentions this requirement.
   - PARTIAL: resume has related content but does not fully satisfy the requirement.
   - NOT FOUND: no evidence in the resume for this requirement.
   - NOT VERIFIABLE: the requirement cannot be confirmed from resume text alone
     (e.g., personality traits, background checks).
5. The "evidence" field must be a direct quote or specific reference from the resume.
   If no evidence, set evidence to "No relevant evidence found in resume."
6. Return ONLY valid JSON array, no markdown, no extra text.

Return this JSON array:
[
  {
    "requirement": "exact JD requirement text",
    "requirement_type": "required|preferred|tool|certification|language|other",
    "evidence": "exact quote or reference from resume, or 'No relevant evidence found in resume.'",
    "status": "FOUND|PARTIAL|NOT FOUND|NOT VERIFIABLE",
    "explanation": "one sentence explaining the status"
  }
]
""".strip()


def match_requirements(
    resume_text: str,
    resume_data: dict,
    jd_data: dict,
) -> list[dict[str, Any]]:
    """
    Evidence-based match of all JD requirements against the resume.

    Args:
        resume_text: Raw extracted resume text.
        resume_data: Structured resume data dict.
        jd_data: Structured JD data dict.

    Returns:
        List of match result dicts, one per JD requirement.
    """
    # Build requirements list
    requirements: list[dict] = []

    for skill in jd_data.get("required_skills", []):
        requirements.append({"req": skill, "type": "required"})
    for skill in jd_data.get("preferred_skills", []):
        requirements.append({"req": skill, "type": "preferred"})
    for tool in jd_data.get("tools_technologies", []):
        requirements.append({"req": tool, "type": "tool"})
    for cert in jd_data.get("certifications", []):
        requirements.append({"req": cert, "type": "certification"})
    for lang in jd_data.get("language_requirements", []):
        requirements.append({"req": lang, "type": "language"})
    for other in jd_data.get("other_requirements", []):
        requirements.append({"req": other, "type": "other"})

    if not requirements:
        return []

    # Deduplicate
    seen: set[str] = set()
    unique_reqs: list[dict] = []
    for r in requirements:
        key = r["req"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique_reqs.append(r)

    # Build requirements list for the prompt
    req_lines = "\n".join(
        f"- [{r['type'].upper()}] {r['req']}" for r in unique_reqs
    )

    user_prompt = f"""
RESUME TEXT (use ONLY this for evidence):
{resume_text[:8000]}

JD REQUIREMENTS TO MATCH (check each one):
{req_lines}

For each requirement above, return a match result JSON object.
""".strip()

    raw_results = call_llm_json(MATCHER_SYSTEM_PROMPT, user_prompt)

    if not isinstance(raw_results, list):
        raise LLMError("Matcher returned unexpected format (not a list).")

    return [_validate_match_item(item) for item in raw_results]


def _validate_match_item(item: Any) -> dict[str, Any]:
    """Ensure a match result has all required keys with safe values."""
    if not isinstance(item, dict):
        item = {}

    status = item.get("status", STATUS_NOT_FOUND).upper().strip()
    if status not in VALID_STATUSES:
        status = STATUS_NOT_FOUND

    return {
        "requirement": str(item.get("requirement", "Unknown requirement")),
        "requirement_type": str(item.get("requirement_type", "other")),
        "evidence": str(item.get("evidence", "No relevant evidence found in resume.")),
        "status": status,
        "explanation": str(item.get("explanation", "")),
    }


def compute_match_summary(matches: list[dict]) -> dict[str, Any]:
    """
    Compute aggregate statistics from match results.

    Returns dict with:
        total, found, partial, not_found, not_verifiable
        found_pct, partial_pct, not_found_pct
        required_found, required_total
    """
    total = len(matches)
    if total == 0:
        return {
            "total": 0, "found": 0, "partial": 0,
            "not_found": 0, "not_verifiable": 0,
            "found_pct": 0, "partial_pct": 0, "not_found_pct": 0,
            "required_found": 0, "required_total": 0,
        }

    found = sum(1 for m in matches if m["status"] == STATUS_FOUND)
    partial = sum(1 for m in matches if m["status"] == STATUS_PARTIAL)
    not_found = sum(1 for m in matches if m["status"] == STATUS_NOT_FOUND)
    not_verifiable = sum(1 for m in matches if m["status"] == STATUS_NOT_VERIFIABLE)

    required = [m for m in matches if m.get("requirement_type") == "required"]
    required_found = sum(
        1 for m in required
        if m["status"] in (STATUS_FOUND, STATUS_PARTIAL)
    )

    return {
        "total": total,
        "found": found,
        "partial": partial,
        "not_found": not_found,
        "not_verifiable": not_verifiable,
        "found_pct": round(found / total * 100),
        "partial_pct": round(partial / total * 100),
        "not_found_pct": round(not_found / total * 100),
        "required_found": required_found,
        "required_total": len(required),
    }
