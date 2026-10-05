"""
ats_analyzer.py
───────────────
ATS (Applicant Tracking System) compatibility analysis.

Analyses:
  A. Resume structure
  B. ATS readability signals
  C. JD keyword coverage
  D. Job title alignment
  E. Skills coverage
  F. Missing keywords
  G. Content quality

Computes a transparent ATS score with methodology shown.
"""

from __future__ import annotations

import re
from typing import Any

from utils.llm_client import call_llm_json, LLMError


# ──────────────────────────────────────────────────────────────────────────
# Scoring weights (must sum to 100)
# ──────────────────────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "jd_keyword_coverage":    25,  # % of JD keywords found in resume
    "required_skill_coverage": 25, # % of required skills FOUND/PARTIAL
    "resume_structure":        20, # clear sections, contact, etc.
    "ats_readability":         15, # no tables, images, complex formatting
    "job_title_alignment":     10, # relevant titles in resume
    "content_quality":          5, # measurable achievements, action verbs
}

assert sum(SCORE_WEIGHTS.values()) == 100, "Score weights must sum to 100"


# ──────────────────────────────────────────────────────────────────────────
# Structure & readability checks (heuristic, no LLM needed)
# ──────────────────────────────────────────────────────────────────────────

EXPECTED_SECTIONS = [
    "education", "experience", "skills", "projects",
    "certifications", "summary", "objective", "contact",
    "work experience", "technical skills", "internship",
]

STRUCTURE_KEYWORDS = {
    "contact_info": [
        r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b",        # email regex
        r"\b\+?[\d\s\-().]{7,}\b",                     # phone
    ],
    "education": ["education", "academic", "degree", "university", "college", "school"],
    "experience": ["experience", "employment", "work history", "career"],
    "skills": ["skills", "technologies", "technical", "competencies", "expertise"],
    "projects": ["projects", "project", "portfolio", "works"],
    "certifications": ["certification", "certificate", "credentials", "courses"],
}

ATS_READABILITY_FLAGS = {
    "tables": [r"\|.+\|", r"─{3,}", r"={3,}"],          # markdown table-like
    "special_chars": [r"[^\x00-\x7F]{5,}"],              # dense non-ASCII
    "repeated_newlines": [r"\n{5,}"],                     # excessive whitespace
}


def check_resume_structure(resume_text: str) -> dict[str, Any]:
    """
    Heuristically check which standard resume sections are present.
    Returns a dict of section_name -> found (bool).
    """
    text_lower = resume_text.lower()
    found_sections: dict[str, bool] = {}

    for section, keywords in STRUCTURE_KEYWORDS.items():
        if section == "contact_info":
            found_sections[section] = any(
                re.search(pat, resume_text) for pat in keywords
            )
        else:
            found_sections[section] = any(kw in text_lower for kw in keywords)

    return found_sections


def check_ats_readability(resume_text: str) -> dict[str, Any]:
    """
    Detect formatting elements that typically hurt ATS readability.
    Returns issues list and a readability score (0–100).
    """
    issues: list[str] = []

    for flag_name, patterns in ATS_READABILITY_FLAGS.items():
        for pat in patterns:
            if re.search(pat, resume_text):
                if flag_name == "tables":
                    issues.append(
                        "Possible table or complex layout detected. "
                        "ATS systems may misread tabular content."
                    )
                elif flag_name == "special_chars":
                    issues.append(
                        "High concentration of special/non-ASCII characters detected. "
                        "These may not render correctly in all ATS systems."
                    )
                elif flag_name == "repeated_newlines":
                    issues.append(
                        "Excessive blank lines detected. "
                        "This may indicate complex formatting or text-box artefacts."
                    )
                break  # one report per flag_name

    # Readability score: start at 100, deduct per issue
    readability_score = max(0, 100 - len(issues) * 20)

    return {
        "issues": issues,
        "readability_score": readability_score,
        "warnings": issues,  # alias for UI
    }


def check_keyword_coverage(
    resume_text: str,
    jd_keywords: list[str],
) -> dict[str, Any]:
    """
    Check how many JD keywords appear verbatim in the resume (case-insensitive).
    Returns found, missing, coverage_pct.
    """
    text_lower = resume_text.lower()
    found: list[str] = []
    missing: list[str] = []

    for kw in jd_keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue
        # Check for whole-word or substring match
        if re.search(re.escape(kw_lower), text_lower):
            found.append(kw)
        else:
            missing.append(kw)

    total = len(found) + len(missing)
    coverage_pct = round(len(found) / total * 100) if total else 0

    return {
        "found": found,
        "missing": missing,
        "total": total,
        "coverage_pct": coverage_pct,
    }


def check_job_title_alignment(
    resume_text: str,
    jd_data: dict,
) -> dict[str, Any]:
    """
    Check if the JD job title or related titles appear in the resume.
    """
    job_title = jd_data.get("job_title", "").lower().strip()
    text_lower = resume_text.lower()

    if not job_title:
        return {"aligned": False, "found_titles": [], "score": 50, "note": "Job title not specified in JD."}

    # Check for the exact job title
    exact_match = job_title in text_lower
    # Check for key words of the title (e.g. "software" and "engineer")
    title_words = [w for w in job_title.split() if len(w) > 3]
    partial_match = sum(1 for w in title_words if w in text_lower)
    partial_ratio = partial_match / len(title_words) if title_words else 0

    if exact_match:
        score = 100
        note = f"Job title '{jd_data.get('job_title')}' found in resume."
    elif partial_ratio >= 0.5:
        score = 60
        note = f"Partial match: {partial_match}/{len(title_words)} title keywords found."
    else:
        score = 20
        note = f"Job title '{jd_data.get('job_title')}' not found in resume."

    return {
        "aligned": exact_match,
        "score": score,
        "note": note,
    }


# ──────────────────────────────────────────────────────────────────────────
# LLM-assisted content quality check
# ──────────────────────────────────────────────────────────────────────────

CONTENT_QUALITY_PROMPT = """
You are an ATS content quality reviewer. Analyze the resume for content quality issues.

RULES:
1. Only report issues that are clearly present in the text.
2. Do not invent problems.
3. Return ONLY valid JSON, no markdown, no extra text.

Return this JSON:
{
  "vague_statements": [],
  "missing_metrics": [],
  "weak_action_verbs": [],
  "unnecessary_info": [],
  "repeated_info": [],
  "positive_aspects": [],
  "content_quality_score": 0
}

content_quality_score: integer 0-100 reflecting how well-written the resume is.
Deduct points for vague statements, missing metrics, weak verbs.
Add points for strong action verbs, quantified achievements, clear descriptions.
""".strip()


def check_content_quality(resume_text: str) -> dict[str, Any]:
    """Use LLM to evaluate resume content quality."""
    user_prompt = f"""
Analyze this resume for content quality issues.

RESUME:
{resume_text[:6000]}
""".strip()

    try:
        data = call_llm_json(CONTENT_QUALITY_PROMPT, user_prompt)
        if not isinstance(data, dict):
            return _default_content_quality()

        # Ensure all keys exist
        for key in ["vague_statements", "missing_metrics", "weak_action_verbs",
                    "unnecessary_info", "repeated_info", "positive_aspects"]:
            if not isinstance(data.get(key), list):
                data[key] = []

        score = data.get("content_quality_score", 50)
        if not isinstance(score, (int, float)):
            score = 50
        data["content_quality_score"] = max(0, min(100, int(score)))

        return data
    except LLMError:
        return _default_content_quality()


def _default_content_quality() -> dict[str, Any]:
    return {
        "vague_statements": [],
        "missing_metrics": [],
        "weak_action_verbs": [],
        "unnecessary_info": [],
        "repeated_info": [],
        "positive_aspects": [],
        "content_quality_score": 50,
    }


# ──────────────────────────────────────────────────────────────────────────
# Full ATS analysis & scoring
# ──────────────────────────────────────────────────────────────────────────

def run_ats_analysis(
    resume_text: str,
    resume_data: dict,
    jd_data: dict,
    match_results: list[dict],
) -> dict[str, Any]:
    """
    Run full ATS analysis and return a scored report.

    Args:
        resume_text:   Raw extracted resume text.
        resume_data:   Structured resume data dict.
        jd_data:       Structured JD data dict.
        match_results: List of match results from matcher.py.

    Returns:
        Full ATS analysis dict including score breakdown.
    """
    # ── A. Resume structure ────────────────────────────────────────────────
    structure = check_resume_structure(resume_text)
    structure_score = round(sum(structure.values()) / len(structure) * 100)

    # ── B. ATS readability ─────────────────────────────────────────────────
    readability = check_ats_readability(resume_text)
    readability_score = readability["readability_score"]

    # ── C. JD keyword coverage ─────────────────────────────────────────────
    jd_keywords = jd_data.get("important_keywords", [])
    # Add required skills as keywords too
    all_keywords = list({
        *jd_keywords,
        *jd_data.get("required_skills", []),
        *jd_data.get("tools_technologies", []),
    })
    keyword_coverage = check_keyword_coverage(resume_text, all_keywords)
    keyword_score = keyword_coverage["coverage_pct"]

    # ── D. Job title alignment ─────────────────────────────────────────────
    title_alignment = check_job_title_alignment(resume_text, jd_data)
    title_score = title_alignment["score"]

    # ── E. Required skill coverage (from match results) ────────────────────
    required_matches = [
        m for m in match_results
        if m.get("requirement_type") == "required"
    ]
    if required_matches:
        found_required = sum(
            1 for m in required_matches
            if m["status"] in ("FOUND", "PARTIAL")
        )
        skill_coverage_score = round(found_required / len(required_matches) * 100)
    else:
        skill_coverage_score = keyword_score  # fallback

    # ── G. Content quality ─────────────────────────────────────────────────
    content_quality = check_content_quality(resume_text)
    content_score = content_quality["content_quality_score"]

    # ── Compute weighted ATS score ─────────────────────────────────────────
    component_scores = {
        "jd_keyword_coverage":    keyword_score,
        "required_skill_coverage": skill_coverage_score,
        "resume_structure":        structure_score,
        "ats_readability":         readability_score,
        "job_title_alignment":     title_score,
        "content_quality":         content_score,
    }

    weighted_score = sum(
        component_scores[k] * SCORE_WEIGHTS[k] / 100
        for k in SCORE_WEIGHTS
    )
    ats_score = round(weighted_score)

    # ── Classify score ─────────────────────────────────────────────────────
    if ats_score >= 80:
        score_label = "Excellent"
        score_color = "green"
    elif ats_score >= 60:
        score_label = "Good"
        score_color = "blue"
    elif ats_score >= 40:
        score_label = "Fair"
        score_color = "orange"
    else:
        score_label = "Needs Improvement"
        score_color = "red"

    return {
        "ats_score": ats_score,
        "score_label": score_label,
        "score_color": score_color,
        "score_weights": SCORE_WEIGHTS,
        "component_scores": component_scores,
        "structure": structure,
        "structure_score": structure_score,
        "readability": readability,
        "readability_score": readability_score,
        "keyword_coverage": keyword_coverage,
        "keyword_score": keyword_score,
        "title_alignment": title_alignment,
        "title_score": title_score,
        "skill_coverage_score": skill_coverage_score,
        "content_quality": content_quality,
        "content_score": content_score,
    }
