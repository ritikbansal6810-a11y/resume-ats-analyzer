"""
app.py
──────
Resume ATS & Job Description Analyzer
Streamlit multi-page dashboard.

Pages:
  1. Resume Upload
  2. Job Description
  3. Resume Analysis
  4. ATS Analysis
  5. JD Match
  6. Skill Gap
  7. Resume Improvement
  8. Interview Preparation
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
# Add the project root to sys.path so `utils.*` imports work regardless of
# the directory from which `streamlit run app.py` is launched.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── IMPORTANT: import config FIRST — it loads .env via an absolute path ───
# All other utils that need env vars import config.py themselves, but doing
# it here guarantees the environment is populated before Streamlit renders
# any widget that calls is_llm_configured() or get_provider_info().
from utils.config import ensure_env_loaded, validate_config  # noqa: E402
ensure_env_loaded()

# ── Application imports (after env is loaded) ──────────────────────────────
from utils.pdf_parser import extract_text_from_pdf
from utils.docx_parser import extract_text_from_docx
from utils.resume_analyzer import analyze_resume
from utils.jd_analyzer import analyze_jd
from utils.matcher import (
    match_requirements, compute_match_summary,
    STATUS_FOUND, STATUS_PARTIAL, STATUS_NOT_FOUND, STATUS_NOT_VERIFIABLE,
    STATUS_EMOJI,
)
from utils.ats_analyzer import run_ats_analysis, SCORE_WEIGHTS
from utils.llm_client import is_llm_configured, get_provider_info, LLMError

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Card-style containers */
.ats-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.ats-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    color: #1f2328;
}
/* Status badges */
.badge-found      { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.badge-partial    { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.badge-notfound   { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.badge-noverify   { background:#d1ecf1; color:#0c5460; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
/* Score circle */
.score-circle {
    display:inline-block; border-radius:50%; width:90px; height:90px;
    line-height:90px; text-align:center; font-size:1.6rem; font-weight:700;
    color:white;
}
.score-green  { background:#28a745; }
.score-blue   { background:#007bff; }
.score-orange { background:#fd7e14; }
.score-red    { background:#dc3545; }
/* Section header */
.section-header {
    border-left: 4px solid #007bff;
    padding-left: 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1rem 0 0.5rem 0;
    color: #1f2328;
}
/* Info note */
.info-note {
    background:#e8f4fd; border:1px solid #bee5fd; border-radius:6px;
    padding:0.6rem 1rem; font-size:0.85rem; color:#084298;
}
/* Warning note */
.warn-note {
    background:#fff8e6; border:1px solid #ffc107; border-radius:6px;
    padding:0.6rem 1rem; font-size:0.85rem; color:#664d03;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "resume_text": "",
        "resume_data": None,
        "jd_text": "",
        "jd_data": None,
        "match_results": None,
        "match_summary": None,
        "ats_report": None,
        "recommendations": None,
        "interview_questions": None,
        "page": "📄 Resume Upload",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ──────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    mapping = {
        STATUS_FOUND: '<span class="badge-found">✅ FOUND</span>',
        STATUS_PARTIAL: '<span class="badge-partial">🔶 PARTIAL</span>',
        STATUS_NOT_FOUND: '<span class="badge-notfound">❌ NOT FOUND IN RESUME</span>',
        STATUS_NOT_VERIFIABLE: '<span class="badge-noverify">🔍 NOT VERIFIABLE</span>',
    }
    return mapping.get(status, f"<span>{status}</span>")


def _score_color_class(score: int) -> str:
    if score >= 80:
        return "score-green"
    elif score >= 60:
        return "score-blue"
    elif score >= 40:
        return "score-orange"
    return "score-red"


def _extract_file_text(uploaded_file) -> tuple[str, list[str]]:
    """
    Extract text from an uploaded Streamlit file object.
    Returns (text, warnings).
    Raises ValueError on fatal errors.
    """
    warnings: list[str] = []
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        result = extract_text_from_pdf(file_bytes)
        if result.error:
            raise ValueError(result.error)
        warnings.extend(result.warnings)
        return result.text, warnings

    elif name.endswith(".docx"):
        result = extract_text_from_docx(file_bytes)
        if result.error:
            raise ValueError(result.error)
        warnings.extend(result.warnings)
        return result.text, warnings

    elif name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8", errors="replace"), warnings
        except Exception as exc:
            raise ValueError(f"Could not read text file: {exc}") from exc

    else:
        raise ValueError(
            f"Unsupported file type: '{uploaded_file.name}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )


def _llm_warning() -> bool:
    """
    Show a clear, actionable warning if the LLM is not configured.
    Returns True if not configured (caller should abort rendering).
    Never reveals the API key value.
    """
    if not is_llm_configured():
        cfg = validate_config()
        # cfg["error"] contains a human-readable message with the exact env var
        # name and .env file path — but never the actual key value.
        error_msg = cfg.get("error") or (
            f"LLM provider **{cfg.get('provider', 'unknown')}** is not configured. "
            f"No API key found."
        )
        dotenv_exists = cfg.get("dotenv_exists", False)

        st.error(f"⚠️ **LLM Configuration Error**\n\n{error_msg}", icon="🔑")

        if not dotenv_exists:
            dotenv_path = cfg.get("dotenv_path", "resume-ats-analyzer/.env")
            st.info(
                f"**`.env` file not found.** "
                f"Create it by copying `.env.example`:\n\n"
                f"```\n"
                f"copy .env.example .env\n"
                f"```\n\n"
                f"Then add your API key to: `{dotenv_path}`\n\n"
                f"Restart Streamlit after saving the file.",
                icon="📄",
            )
        else:
            st.info(
                "The `.env` file exists but the API key variable is missing or empty.\n\n"
                "Open your `.env` file and make sure the correct key variable is set, "
                "then restart Streamlit.",
                icon="📝",
            )
        return True
    return False


def _needs_resume_and_jd() -> bool:
    """Return True (and show error) if resume or JD is missing."""
    if not st.session_state.resume_text:
        st.error("Please upload and process your resume first (Page 1).")
        return True
    if not st.session_state.jd_text:
        st.error("Please enter the Job Description first (Page 2).")
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────

PAGES = [
    "📄 Resume Upload",
    "📋 Job Description",
    "🔍 Resume Analysis",
    "🤖 ATS Analysis",
    "🎯 JD Match",
    "📊 Skill Gap",
    "✏️ Resume Improvement",
    "🎤 Interview Prep",
]

with st.sidebar:
    st.markdown("## 📄 Resume ATS Analyzer")
    st.caption("AI-powered resume & JD analysis")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = selected_page

    st.divider()

    # Status indicators
    st.markdown("**Status**")
    st.markdown(
        f"{'✅' if st.session_state.resume_text else '⬜'} Resume uploaded"
    )
    st.markdown(
        f"{'✅' if st.session_state.jd_text else '⬜'} JD entered"
    )
    st.markdown(
        f"{'✅' if st.session_state.resume_data else '⬜'} Resume analyzed"
    )
    st.markdown(
        f"{'✅' if st.session_state.match_results else '⬜'} JD matched"
    )
    st.markdown(
        f"{'✅' if st.session_state.ats_report else '⬜'} ATS scored"
    )

    st.divider()

    # LLM info — never show API key values
    info = get_provider_info()
    status_icon = "🟢" if info["configured"] else "🔴"
    env_icon = "📄" if info.get("dotenv_exists") else "⚠️"
    st.markdown(f"**LLM** {status_icon}")
    st.caption(f"Provider: {info['provider']}")
    st.caption(f"Model: {info['model']}")
    st.caption(f"{env_icon} .env {'found' if info.get('dotenv_exists') else 'NOT found'}")

    if st.button("🗑️ Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# PAGE 1: Resume Upload
# ──────────────────────────────────────────────────────────────────────────

def page_resume_upload():
    st.title("📄 Resume Upload")
    st.markdown(
        '<div class="info-note">Upload your resume as PDF or DOCX. '
        'The text will be extracted and used for all subsequent analysis. '
        '<strong>Tip:</strong> Use a text-based PDF (not a scanned image) for best results.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Resume File")
        uploaded = st.file_uploader(
            "Choose PDF or DOCX",
            type=["pdf", "docx"],
            help="Maximum file size: 10 MB",
        )

        if uploaded:
            with st.spinner("Extracting text from resume…"):
                try:
                    text, warnings = _extract_file_text(uploaded)
                    st.session_state.resume_text = text
                    # Clear downstream results when resume changes
                    st.session_state.resume_data = None
                    st.session_state.match_results = None
                    st.session_state.ats_report = None
                    st.session_state.recommendations = None
                    st.session_state.interview_questions = None

                    st.success(f"✅ Text extracted successfully ({len(text):,} characters)")

                    for w in warnings:
                        st.warning(w)

                except ValueError as exc:
                    st.error(f"❌ {exc}")
                    st.session_state.resume_text = ""

    with col2:
        st.subheader("Or Paste Resume Text")
        pasted = st.text_area(
            "Paste resume text here",
            height=250,
            placeholder="Paste the full text of your resume here…",
        )
        if st.button("Use Pasted Text", use_container_width=True):
            if pasted.strip():
                st.session_state.resume_text = pasted.strip()
                st.session_state.resume_data = None
                st.session_state.match_results = None
                st.session_state.ats_report = None
                st.success("✅ Resume text saved.")
            else:
                st.warning("Please paste some text first.")

    # Preview
    if st.session_state.resume_text:
        st.divider()
        st.subheader("Extracted Resume Text Preview")
        with st.expander("Show extracted text", expanded=False):
            st.text_area(
                "Extracted text",
                value=st.session_state.resume_text,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )
        st.caption(f"Total characters: {len(st.session_state.resume_text):,}")

        st.divider()
        st.subheader("Quick Analyze Resume")
        st.markdown("Parse the resume into structured sections using AI.")

        if _llm_warning():
            return

        if st.button("🔍 Analyze Resume with AI", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume with AI…"):
                try:
                    st.session_state.resume_data = analyze_resume(
                        st.session_state.resume_text
                    )
                    st.success("✅ Resume analyzed! Go to **🔍 Resume Analysis** page.")
                except (LLMError, ValueError) as exc:
                    st.error(f"Analysis failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2: Job Description
# ──────────────────────────────────────────────────────────────────────────

def page_job_description():
    st.title("📋 Job Description")
    st.markdown(
        '<div class="info-note">Paste or upload the job description. '
        'This will be parsed and compared against your resume.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    tab1, tab2 = st.tabs(["Paste JD Text", "Upload JD File"])

    with tab1:
        jd_input = st.text_area(
            "Paste the full Job Description here",
            value=st.session_state.jd_text,
            height=350,
            placeholder="Paste the complete job description including responsibilities, requirements, and qualifications…",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Save JD", type="primary", use_container_width=True):
                if jd_input.strip():
                    st.session_state.jd_text = jd_input.strip()
                    # Clear downstream JD results
                    st.session_state.jd_data = None
                    st.session_state.match_results = None
                    st.session_state.ats_report = None
                    st.success("✅ JD saved.")
                else:
                    st.warning("Please paste a job description first.")

    with tab2:
        jd_file = st.file_uploader(
            "Upload JD (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_file",
        )
        if jd_file:
            with st.spinner("Extracting JD text…"):
                try:
                    jd_text, warnings = _extract_file_text(jd_file)
                    st.session_state.jd_text = jd_text
                    st.session_state.jd_data = None
                    st.session_state.match_results = None
                    for w in warnings:
                        st.warning(w)
                    st.success(f"✅ JD text extracted ({len(jd_text):,} chars).")
                except ValueError as exc:
                    st.error(f"❌ {exc}")

    # Show current JD
    if st.session_state.jd_text:
        st.divider()
        with st.expander("📄 Current JD Text Preview", expanded=False):
            st.text_area(
                "",
                value=st.session_state.jd_text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

        st.subheader("Parse Job Description with AI")
        if _llm_warning():
            return

        if st.button("🔍 Parse JD with AI", type="primary", use_container_width=True):
            with st.spinner("Parsing job description…"):
                try:
                    st.session_state.jd_data = analyze_jd(st.session_state.jd_text)
                    st.success("✅ JD parsed! Continue to **🎯 JD Match** page.")
                except (LLMError, ValueError) as exc:
                    st.error(f"JD analysis failed: {exc}")

        # Show parsed JD
        if st.session_state.jd_data:
            jd = st.session_state.jd_data
            st.divider()
            st.subheader("Parsed JD Structure")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">Job Info</div>', unsafe_allow_html=True)
                st.markdown(f"**Title:** {jd.get('job_title', 'N/A')}")
                st.markdown(f"**Company:** {jd.get('company', 'N/A')}")
                st.markdown(f"**Location:** {jd.get('location', 'N/A')}")
                st.markdown(f"**Type:** {jd.get('employment_type', 'N/A')}")

                st.markdown('<div class="section-header">Required Skills</div>', unsafe_allow_html=True)
                for s in jd.get("required_skills", []):
                    st.markdown(f"• {s}")

                st.markdown('<div class="section-header">Preferred Skills</div>', unsafe_allow_html=True)
                for s in jd.get("preferred_skills", []):
                    st.markdown(f"• {s}")

            with col2:
                st.markdown('<div class="section-header">Tools & Technologies</div>', unsafe_allow_html=True)
                for t in jd.get("tools_technologies", []):
                    st.markdown(f"• {t}")

                st.markdown('<div class="section-header">Experience Requirements</div>', unsafe_allow_html=True)
                exp_req = jd.get("experience_requirements", {})
                if exp_req.get("years"):
                    st.markdown(f"**Years:** {exp_req['years']}")
                if exp_req.get("level"):
                    st.markdown(f"**Level:** {exp_req['level']}")
                for d in exp_req.get("details", []):
                    st.markdown(f"• {d}")

                st.markdown('<div class="section-header">Education Requirements</div>', unsafe_allow_html=True)
                for e in jd.get("education_requirements", []):
                    st.markdown(f"• {e}")

            with st.expander("Responsibilities"):
                for r in jd.get("responsibilities", []):
                    st.markdown(f"• {r}")

            with st.expander("Important Keywords"):
                kws = jd.get("important_keywords", [])
                if kws:
                    st.write(", ".join(kws))


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3: Resume Analysis
# ──────────────────────────────────────────────────────────────────────────

def page_resume_analysis():
    st.title("🔍 Resume Analysis")

    if not st.session_state.resume_text:
        st.error("Please upload your resume first (Page 1).")
        return

    if _llm_warning():
        return

    # Run analysis if not done
    if not st.session_state.resume_data:
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume with AI…"):
                try:
                    st.session_state.resume_data = analyze_resume(
                        st.session_state.resume_text
                    )
                    st.rerun()
                except (LLMError, ValueError) as exc:
                    st.error(f"Analysis failed: {exc}")
        return

    data = st.session_state.resume_data

    # ── Header ─────────────────────────────────────────────────────────────
    name = data.get("candidate_name") or "Candidate"
    st.markdown(f"## {name}")

    contact = data.get("contact", {})
    contact_parts = [
        v for v in [
            contact.get("email"), contact.get("phone"),
            contact.get("location"), contact.get("linkedin"),
        ] if v
    ]
    if contact_parts:
        st.caption(" · ".join(contact_parts))

    if data.get("summary"):
        st.markdown(f"*{data['summary']}*")

    st.divider()

    # ── Main sections ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        # Education
        if data.get("education"):
            st.markdown('<div class="section-header">🎓 Education</div>', unsafe_allow_html=True)
            for edu in data["education"]:
                degree = " ".join(filter(None, [edu.get("degree"), edu.get("field")]))
                institution = edu.get("institution", "")
                year = edu.get("year", "")
                gpa = edu.get("gpa", "")
                st.markdown(
                    f'<div class="ats-card">'
                    f'<div class="ats-card-title">{degree}</div>'
                    f'{"<div>" + institution + "</div>" if institution else ""}'
                    f'{"<small>" + year + ("  |  GPA: " + gpa if gpa else "") + "</small>" if year else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Experience
        if data.get("experience"):
            st.markdown('<div class="section-header">💼 Work Experience</div>', unsafe_allow_html=True)
            for exp in data["experience"]:
                with st.expander(
                    f"{exp.get('title', 'N/A')} @ {exp.get('company', 'N/A')} "
                    f"({exp.get('duration') or exp.get('start_date', '')})"
                ):
                    for r in exp.get("responsibilities", []):
                        st.markdown(f"• {r}")
                    if exp.get("technologies_used"):
                        st.caption("Tech: " + ", ".join(exp["technologies_used"]))

        # Internships
        if data.get("internships"):
            st.markdown('<div class="section-header">📌 Internships</div>', unsafe_allow_html=True)
            for intern in data["internships"]:
                with st.expander(
                    f"{intern.get('title', 'N/A')} @ {intern.get('company', 'N/A')}"
                ):
                    for r in intern.get("responsibilities", []):
                        st.markdown(f"• {r}")
                    if intern.get("technologies_used"):
                        st.caption("Tech: " + ", ".join(intern["technologies_used"]))

    with col2:
        # Projects
        if data.get("projects"):
            st.markdown('<div class="section-header">🛠️ Projects</div>', unsafe_allow_html=True)
            for proj in data["projects"]:
                with st.expander(proj.get("name", "Unnamed Project")):
                    if proj.get("description"):
                        st.write(proj["description"])
                    if proj.get("technologies"):
                        st.caption("Tech: " + ", ".join(proj["technologies"]))
                    if proj.get("link"):
                        st.markdown(f"[Project Link]({proj['link']})")

        # Technical Skills
        if data.get("technical_skills"):
            st.markdown('<div class="section-header">⚙️ Technical Skills</div>', unsafe_allow_html=True)
            skills_text = " • ".join(data["technical_skills"])
            st.markdown(f'<div class="ats-card">{skills_text}</div>', unsafe_allow_html=True)

        # Soft Skills
        if data.get("soft_skills"):
            st.markdown('<div class="section-header">🤝 Soft Skills</div>', unsafe_allow_html=True)
            st.write(", ".join(data["soft_skills"]))

        # Tools & Technologies
        if data.get("tools_technologies"):
            st.markdown('<div class="section-header">🔧 Tools & Technologies</div>', unsafe_allow_html=True)
            st.write(", ".join(data["tools_technologies"]))

        # Certifications
        if data.get("certifications"):
            st.markdown('<div class="section-header">🏆 Certifications</div>', unsafe_allow_html=True)
            for cert in data["certifications"]:
                parts = [cert.get("name", "")]
                if cert.get("issuer"):
                    parts.append(cert["issuer"])
                if cert.get("year"):
                    parts.append(cert["year"])
                st.markdown(f"• {' · '.join(filter(None, parts))}")

        # Languages
        if data.get("languages"):
            st.markdown('<div class="section-header">🌐 Languages</div>', unsafe_allow_html=True)
            st.write(", ".join(data["languages"]))

        # Achievements
        if data.get("achievements"):
            st.markdown('<div class="section-header">🏅 Achievements</div>', unsafe_allow_html=True)
            for a in data["achievements"]:
                st.markdown(f"• {a}")

    # Keywords
    if data.get("keywords"):
        st.divider()
        st.markdown('<div class="section-header">🔑 Extracted Keywords</div>', unsafe_allow_html=True)
        st.write(", ".join(data["keywords"]))

    # Raw JSON
    with st.expander("🔧 Raw Parsed Data (JSON)"):
        st.json(data)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 4: ATS Analysis
# ──────────────────────────────────────────────────────────────────────────

def page_ats_analysis():
    st.title("🤖 ATS Analysis")

    if _needs_resume_and_jd():
        return
    if _llm_warning():
        return

    # Ensure we have parsed data
    if not st.session_state.resume_data:
        with st.spinner("Analyzing resume…"):
            try:
                st.session_state.resume_data = analyze_resume(st.session_state.resume_text)
            except (LLMError, ValueError) as exc:
                st.error(f"Resume analysis failed: {exc}")
                return

    if not st.session_state.jd_data:
        with st.spinner("Parsing JD…"):
            try:
                st.session_state.jd_data = analyze_jd(st.session_state.jd_text)
            except (LLMError, ValueError) as exc:
                st.error(f"JD analysis failed: {exc}")
                return

    if not st.session_state.match_results:
        with st.spinner("Running evidence-based matching…"):
            try:
                st.session_state.match_results = match_requirements(
                    st.session_state.resume_text,
                    st.session_state.resume_data,
                    st.session_state.jd_data,
                )
                st.session_state.match_summary = compute_match_summary(
                    st.session_state.match_results
                )
            except (LLMError, ValueError) as exc:
                st.error(f"Matching failed: {exc}")
                return

    # Run ATS analysis
    if not st.session_state.ats_report:
        if st.button("🤖 Run Full ATS Analysis", type="primary", use_container_width=True):
            with st.spinner("Running ATS analysis (this may take a moment)…"):
                try:
                    st.session_state.ats_report = run_ats_analysis(
                        st.session_state.resume_text,
                        st.session_state.resume_data,
                        st.session_state.jd_data,
                        st.session_state.match_results,
                    )
                    st.rerun()
                except (LLMError, ValueError) as exc:
                    st.error(f"ATS analysis failed: {exc}")
            return

    ats = st.session_state.ats_report
    score = ats["ats_score"]
    color = ats["score_color"]

    # ── Score header ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center; padding: 1.5rem 0;">'
        f'<div class="score-circle score-{color}">{score}</div>'
        f'<div style="font-size:1.2rem; margin-top:0.5rem; font-weight:600;">'
        f'ATS Compatibility: {score}/100</div>'
        f'<div style="color:#57606a; font-size:0.9rem;">{ats["score_label"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="warn-note">⚠️ This is an <strong>estimated</strong> ATS compatibility score. '
        'It is NOT a guarantee that a real company\'s ATS system will accept or reject your resume. '
        'Different ATS platforms have different algorithms and thresholds.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Score breakdown ────────────────────────────────────────────────────
    st.subheader("Score Breakdown & Methodology")
    st.caption("Weights used in scoring calculation:")

    comp = ats["component_scores"]
    weights = ats["score_weights"]

    import pandas as pd
    breakdown_rows = []
    for key, weight in weights.items():
        raw_score = comp.get(key, 0)
        contribution = round(raw_score * weight / 100, 1)
        label = key.replace("_", " ").title()
        breakdown_rows.append({
            "Category": label,
            "Weight": f"{weight}%",
            "Your Score": f"{raw_score}/100",
            "Contribution": f"{contribution} pts",
        })

    df_breakdown = pd.DataFrame(breakdown_rows)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    for key, weight in weights.items():
        raw_score = comp.get(key, 0)
        label = key.replace("_", " ").title()
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.progress(raw_score / 100, text=label)
        with col2:
            st.caption(f"Score: {raw_score}/100")
        with col3:
            st.caption(f"Weight: {weight}%")

    st.divider()

    # ── Section details ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # A. Structure
        st.markdown('<div class="section-header">A. Resume Structure</div>', unsafe_allow_html=True)
        structure = ats["structure"]
        for section, found in structure.items():
            icon = "✅" if found else "❌"
            label = section.replace("_", " ").title()
            st.markdown(f"{icon} {label}")

        # B. ATS Readability
        st.markdown('<div class="section-header">B. ATS Readability</div>', unsafe_allow_html=True)
        readability = ats["readability"]
        if readability["issues"]:
            for issue in readability["issues"]:
                st.warning(issue)
        else:
            st.success("✅ No major readability issues detected.")

    with col2:
        # C. Keyword Coverage
        st.markdown('<div class="section-header">C. JD Keyword Coverage</div>', unsafe_allow_html=True)
        kw = ats["keyword_coverage"]
        st.metric("Keywords Found", f"{len(kw['found'])}/{kw['total']}", f"{kw['coverage_pct']}%")

        if kw["found"]:
            with st.expander(f"✅ Found Keywords ({len(kw['found'])})"):
                st.write(", ".join(kw["found"]))

        if kw["missing"]:
            with st.expander(f"❌ Missing Keywords ({len(kw['missing'])})"):
                st.write(", ".join(kw["missing"]))

        # D. Job Title Alignment
        st.markdown('<div class="section-header">D. Job Title Alignment</div>', unsafe_allow_html=True)
        title_info = ats["title_alignment"]
        icon = "✅" if title_info["aligned"] else "⚠️"
        st.markdown(f"{icon} {title_info['note']}")

    # G. Content Quality
    st.markdown('<div class="section-header">G. Content Quality</div>', unsafe_allow_html=True)
    cq = ats["content_quality"]
    cq_score = cq.get("content_quality_score", 50)
    st.progress(cq_score / 100, text=f"Content Quality: {cq_score}/100")

    cq_col1, cq_col2 = st.columns(2)
    with cq_col1:
        if cq.get("vague_statements"):
            with st.expander(f"⚠️ Vague Statements ({len(cq['vague_statements'])})"):
                for v in cq["vague_statements"]:
                    st.markdown(f"• {v}")
        if cq.get("weak_action_verbs"):
            with st.expander(f"⚠️ Weak Action Verbs ({len(cq['weak_action_verbs'])})"):
                for v in cq["weak_action_verbs"]:
                    st.markdown(f"• {v}")
        if cq.get("missing_metrics"):
            with st.expander(f"📊 Missing Metrics ({len(cq['missing_metrics'])})"):
                for v in cq["missing_metrics"]:
                    st.markdown(f"• {v}")

    with cq_col2:
        if cq.get("positive_aspects"):
            with st.expander(f"✅ Positive Aspects ({len(cq['positive_aspects'])})"):
                for v in cq["positive_aspects"]:
                    st.markdown(f"• {v}")
        if cq.get("unnecessary_info"):
            with st.expander(f"ℹ️ Unnecessary Info ({len(cq['unnecessary_info'])})"):
                for v in cq["unnecessary_info"]:
                    st.markdown(f"• {v}")

    if st.button("🔄 Re-run ATS Analysis", use_container_width=True):
        st.session_state.ats_report = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# PAGE 5: JD Match
# ──────────────────────────────────────────────────────────────────────────

def page_jd_match():
    st.title("🎯 Resume vs Job Description Match")

    if _needs_resume_and_jd():
        return
    if _llm_warning():
        return

    # Ensure prerequisite data
    if not st.session_state.resume_data:
        with st.spinner("Analyzing resume…"):
            try:
                st.session_state.resume_data = analyze_resume(st.session_state.resume_text)
            except (LLMError, ValueError) as exc:
                st.error(f"Resume analysis failed: {exc}")
                return

    if not st.session_state.jd_data:
        with st.spinner("Parsing JD…"):
            try:
                st.session_state.jd_data = analyze_jd(st.session_state.jd_text)
            except (LLMError, ValueError) as exc:
                st.error(f"JD analysis failed: {exc}")
                return

    if not st.session_state.match_results:
        if st.button("🎯 Run JD Match Analysis", type="primary", use_container_width=True):
            with st.spinner("Running evidence-based matching…"):
                try:
                    st.session_state.match_results = match_requirements(
                        st.session_state.resume_text,
                        st.session_state.resume_data,
                        st.session_state.jd_data,
                    )
                    st.session_state.match_summary = compute_match_summary(
                        st.session_state.match_results
                    )
                    st.rerun()
                except (LLMError, ValueError) as exc:
                    st.error(f"Matching failed: {exc}")
        return

    matches = st.session_state.match_results
    summary = st.session_state.match_summary or compute_match_summary(matches)

    # ── Summary metrics ────────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Requirements", summary["total"])
    with cols[1]:
        st.metric("✅ Found", summary["found"], f"{summary['found_pct']}%")
    with cols[2]:
        st.metric("🔶 Partial", summary["partial"], f"{summary['partial_pct']}%")
    with cols[3]:
        st.metric("❌ Not Found", summary["not_found"], f"{summary['not_found_pct']}%")

    if summary["required_total"] > 0:
        required_pct = round(summary["required_found"] / summary["required_total"] * 100)
        st.progress(
            summary["required_found"] / summary["required_total"],
            text=f"Required Skills Coverage: {summary['required_found']}/{summary['required_total']} ({required_pct}%)"
        )

    st.divider()

    st.markdown(
        '<div class="info-note">'
        '<strong>Evidence rule:</strong> A skill is only marked as FOUND when explicit evidence exists in your resume. '
        '"NOT FOUND IN RESUME" means no evidence was found — it does NOT mean you lack the skill.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Filter controls ────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        filter_status = st.multiselect(
            "Filter by status",
            options=[STATUS_FOUND, STATUS_PARTIAL, STATUS_NOT_FOUND, STATUS_NOT_VERIFIABLE],
            default=[STATUS_FOUND, STATUS_PARTIAL, STATUS_NOT_FOUND, STATUS_NOT_VERIFIABLE],
        )
    with col_f2:
        filter_type = st.multiselect(
            "Filter by requirement type",
            options=sorted({m.get("requirement_type", "other") for m in matches}),
            default=sorted({m.get("requirement_type", "other") for m in matches}),
        )

    filtered = [
        m for m in matches
        if m["status"] in filter_status
        and m.get("requirement_type", "other") in filter_type
    ]

    st.caption(f"Showing {len(filtered)} of {len(matches)} requirements")

    # ── Match table ────────────────────────────────────────────────────────
    import pandas as pd

    table_rows = []
    for m in filtered:
        table_rows.append({
            "Status": f"{STATUS_EMOJI.get(m['status'], '')} {m['status']}",
            "Type": m.get("requirement_type", "").upper(),
            "JD Requirement": m["requirement"],
            "Resume Evidence": m["evidence"],
            "Explanation": m["explanation"],
        })

    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No results match the selected filters.")

    # ── Detailed cards ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Detailed Match Cards")

    for m in filtered:
        status = m["status"]
        badge = _status_badge(status)

        with st.expander(f"{STATUS_EMOJI.get(status,'')} {m['requirement']} [{m.get('requirement_type','').upper()}]"):
            st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
            st.markdown(f"**JD Requirement:** {m['requirement']}")
            st.markdown(f"**Resume Evidence:** `{m['evidence']}`")
            st.markdown(f"**Explanation:** {m['explanation']}")

    if st.button("🔄 Re-run Match", use_container_width=True):
        st.session_state.match_results = None
        st.session_state.match_summary = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# PAGE 6: Skill Gap
# ──────────────────────────────────────────────────────────────────────────

def page_skill_gap():
    st.title("📊 Skill Gap Analysis")

    if _needs_resume_and_jd():
        return

    if not st.session_state.match_results:
        st.warning("Please run the JD Match analysis first (Page 5).")
        return

    matches = st.session_state.match_results
    summary = st.session_state.match_summary or compute_match_summary(matches)

    # ── Categorize matches ─────────────────────────────────────────────────
    found_skills = [m for m in matches if m["status"] == STATUS_FOUND]
    partial_skills = [m for m in matches if m["status"] == STATUS_PARTIAL]
    not_found_skills = [m for m in matches if m["status"] == STATUS_NOT_FOUND]
    not_verifiable = [m for m in matches if m["status"] == STATUS_NOT_VERIFIABLE]

    required_gap = [
        m for m in not_found_skills
        if m.get("requirement_type") == "required"
    ]
    preferred_gap = [
        m for m in not_found_skills
        if m.get("requirement_type") == "preferred"
    ]

    # ── Visual summary ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Matching Skills", len(found_skills))
    col2.metric("🔶 Partial Matches", len(partial_skills))
    col3.metric("❌ Missing Skills", len(not_found_skills))
    col4.metric("🔍 Not Verifiable", len(not_verifiable))

    # Pie-like progress bars
    total = summary["total"] or 1
    st.progress(len(found_skills) / total, text=f"Found: {len(found_skills)}/{total} ({round(len(found_skills)/total*100)}%)")
    st.progress((len(found_skills) + len(partial_skills)) / total,
                text=f"Found + Partial: {len(found_skills)+len(partial_skills)}/{total}")

    st.divider()

    tab_match, tab_gap, tab_pref = st.tabs(["✅ Matching", "❌ Gaps (Required)", "🔶 Preferred Gaps"])

    with tab_match:
        if found_skills:
            st.markdown("These requirements have **explicit evidence** in your resume:")
            for m in found_skills:
                st.markdown(
                    f'<div class="ats-card">'
                    f'<div class="ats-card-title">✅ {m["requirement"]}</div>'
                    f'<div style="font-size:0.85rem; color:#155724;">Evidence: {m["evidence"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No fully matched requirements found.")

        if partial_skills:
            st.markdown("**Partial Matches** — related but incomplete evidence:")
            for m in partial_skills:
                st.markdown(
                    f'<div class="ats-card">'
                    f'<div class="ats-card-title">🔶 {m["requirement"]}</div>'
                    f'<div style="font-size:0.85rem; color:#856404;">Evidence: {m["evidence"]}</div>'
                    f'<div style="font-size:0.8rem; color:#57606a;">{m["explanation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab_gap:
        if required_gap:
            st.markdown(
                '<div class="warn-note">'
                '⚠️ These <strong>required</strong> JD skills have no evidence in your resume. '
                'This does NOT mean you lack these skills — only that they are not mentioned in your resume.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            for m in required_gap:
                st.markdown(
                    f'<div class="ats-card">'
                    f'<div class="ats-card-title">❌ {m["requirement"]} '
                    f'<span style="font-size:0.75rem; color:#721c24;">[NOT FOUND IN RESUME]</span></div>'
                    f'<div style="font-size:0.85rem; color:#57606a;">{m["explanation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ All required skills have evidence in your resume!")

    with tab_pref:
        if preferred_gap:
            for m in preferred_gap:
                st.markdown(
                    f'<div class="ats-card">'
                    f'<div class="ats-card-title">🔶 {m["requirement"]} '
                    f'<span style="font-size:0.75rem; color:#856404;">[PREFERRED – NOT FOUND IN RESUME]</span></div>'
                    f'<div style="font-size:0.85rem; color:#57606a;">{m["explanation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ All preferred skills are covered!")


# ──────────────────────────────────────────────────────────────────────────
# PAGE 7: Resume Improvement
# ──────────────────────────────────────────────────────────────────────────

IMPROVEMENT_SYSTEM_PROMPT = """
You are an expert resume consultant and ATS specialist.
Based ONLY on the provided resume text and job description, generate
specific, actionable improvement recommendations.

CRITICAL RULES:
1. NEVER tell the candidate to add skills they don't have.
2. Only suggest adding JD keywords if there is evidence in the resume
   that the candidate genuinely has the skill.
3. Classify each missing keyword as one of:
   A. "Add if you genuinely have this skill" (when resume has partial evidence)
   B. "Learn this skill" (clearly a capability gap)
   C. "Not necessary to add" (irrelevant or low-priority)
4. Do not invent resume content.
5. Return ONLY valid JSON, no markdown, no extra text.

Return this JSON:
{
  "priority_improvements": [
    {
      "category": "keyword|bullet_point|structure|ats_readability|skill_section|other",
      "issue": "specific issue",
      "recommendation": "specific actionable fix",
      "example_before": "",
      "example_after": "",
      "impact": "high|medium|low"
    }
  ],
  "missing_keyword_actions": [
    {
      "keyword": "",
      "action": "A|B|C",
      "rationale": ""
    }
  ],
  "summary": "overall summary of recommended changes"
}
""".strip()


def page_resume_improvement():
    st.title("✏️ Resume Improvement Recommendations")

    if _needs_resume_and_jd():
        return
    if _llm_warning():
        return

    if not st.session_state.match_results:
        st.warning("Please run the JD Match analysis first (Page 5).")
        return

    # Generate recommendations
    if not st.session_state.recommendations:
        if st.button("✏️ Generate Improvement Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating personalized recommendations…"):
                try:
                    from utils.llm_client import call_llm_json

                    # Build missing keywords list from match results
                    not_found = [
                        m["requirement"] for m in st.session_state.match_results
                        if m["status"] == STATUS_NOT_FOUND
                    ]
                    partial = [
                        m["requirement"] for m in st.session_state.match_results
                        if m["status"] == STATUS_PARTIAL
                    ]

                    ats_issues = []
                    if st.session_state.ats_report:
                        ats = st.session_state.ats_report
                        ats_issues = ats.get("readability", {}).get("issues", [])

                    user_prompt = f"""
RESUME TEXT:
{st.session_state.resume_text[:6000]}

JOB DESCRIPTION:
{st.session_state.jd_text[:3000]}

MISSING REQUIRED KEYWORDS (not found in resume):
{json.dumps(not_found)}

PARTIAL MATCH KEYWORDS:
{json.dumps(partial)}

ATS READABILITY ISSUES:
{json.dumps(ats_issues)}

Generate improvement recommendations.
""".strip()

                    st.session_state.recommendations = call_llm_json(
                        IMPROVEMENT_SYSTEM_PROMPT, user_prompt
                    )
                    st.rerun()
                except (LLMError, ValueError) as exc:
                    st.error(f"Failed to generate recommendations: {exc}")
        return

    recs = st.session_state.recommendations
    if not isinstance(recs, dict):
        st.error("Invalid recommendations format. Please re-run.")
        return

    # ── Summary ────────────────────────────────────────────────────────────
    if recs.get("summary"):
        st.info(recs["summary"])

    st.divider()

    # ── Priority improvements ──────────────────────────────────────────────
    improvements = recs.get("priority_improvements", [])
    if improvements:
        st.subheader(f"Priority Improvements ({len(improvements)})")

        # Sort by impact
        impact_order = {"high": 0, "medium": 1, "low": 2}
        improvements_sorted = sorted(
            improvements,
            key=lambda x: impact_order.get(x.get("impact", "low"), 2)
        )

        for imp in improvements_sorted:
            impact = imp.get("impact", "medium").lower()
            impact_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}.get(impact, impact)
            category = imp.get("category", "other").replace("_", " ").title()

            with st.expander(f"{impact_badge} | {category} — {imp.get('issue', 'Issue')}"):
                st.markdown(f"**Issue:** {imp.get('issue', '')}")
                st.markdown(f"**Recommendation:** {imp.get('recommendation', '')}")

                if imp.get("example_before") or imp.get("example_after"):
                    col_b, col_a = st.columns(2)
                    with col_b:
                        if imp.get("example_before"):
                            st.markdown("**Before:**")
                            st.code(imp["example_before"], language=None)
                    with col_a:
                        if imp.get("example_after"):
                            st.markdown("**After:**")
                            st.code(imp["example_after"], language=None)

    # ── Missing keyword actions ────────────────────────────────────────────
    kw_actions = recs.get("missing_keyword_actions", [])
    if kw_actions:
        st.divider()
        st.subheader("Missing Keyword Action Plan")

        st.markdown(
            '<div class="info-note">'
            '<strong>Action Guide:</strong> '
            '<strong>A</strong> = Add if you genuinely have this skill | '
            '<strong>B</strong> = Consider learning this skill | '
            '<strong>C</strong> = Not necessary to add'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        import pandas as pd
        kw_rows = []
        for kw in kw_actions:
            action = kw.get("action", "C")
            action_label = {
                "A": "✏️ A — Add if genuine",
                "B": "📚 B — Learn this skill",
                "C": "⏭️ C — Skip",
            }.get(action, action)
            kw_rows.append({
                "Keyword": kw.get("keyword", ""),
                "Action": action_label,
                "Rationale": kw.get("rationale", ""),
            })
        df_kw = pd.DataFrame(kw_rows)
        st.dataframe(df_kw, use_container_width=True, hide_index=True)

    if st.button("🔄 Re-generate Recommendations", use_container_width=True):
        st.session_state.recommendations = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# PAGE 8: Interview Preparation
# ──────────────────────────────────────────────────────────────────────────

INTERVIEW_SYSTEM_PROMPT = """
You are an expert interview coach. Based on the resume, job description,
and skill gap analysis, generate targeted interview questions.

CRITICAL RULES:
1. Only use information present in the resume and JD.
2. For skills/experience found in the resume, generate depth questions.
3. For missing skills, generate gap questions the interviewer is likely to ask.
4. Do not fabricate resume content.
5. Return ONLY valid JSON, no markdown, no extra text.

Return this JSON:
{
  "technical_questions": [
    {"question": "", "why_asked": "", "based_on": "resume|jd|gap"}
  ],
  "hr_questions": [
    {"question": "", "why_asked": "", "based_on": "resume|jd|gap"}
  ],
  "project_questions": [
    {"question": "", "why_asked": "", "based_on": "resume|jd|gap"}
  ],
  "resume_based_questions": [
    {"question": "", "why_asked": "", "based_on": "resume|jd|gap"}
  ],
  "gap_questions": [
    {"question": "", "why_asked": "", "based_on": "gap", "missing_skill": ""}
  ]
}
""".strip()


def page_interview_prep():
    st.title("🎤 Interview Preparation")

    if _needs_resume_and_jd():
        return
    if _llm_warning():
        return

    if not st.session_state.match_results:
        st.warning("Please run the JD Match first (Page 5).")
        return

    if not st.session_state.interview_questions:
        if st.button("🎤 Generate Interview Questions", type="primary", use_container_width=True):
            with st.spinner("Generating interview questions…"):
                try:
                    from utils.llm_client import call_llm_json

                    not_found = [
                        m["requirement"] for m in st.session_state.match_results
                        if m["status"] == STATUS_NOT_FOUND
                    ]
                    found = [
                        m["requirement"] for m in st.session_state.match_results
                        if m["status"] == STATUS_FOUND
                    ]

                    user_prompt = f"""
RESUME TEXT:
{st.session_state.resume_text[:5000]}

JOB DESCRIPTION:
{st.session_state.jd_text[:2000]}

CONFIRMED SKILLS (found in resume):
{json.dumps(found[:20])}

MISSING SKILLS (not found in resume):
{json.dumps(not_found[:20])}

Generate interview questions across all categories.
""".strip()

                    st.session_state.interview_questions = call_llm_json(
                        INTERVIEW_SYSTEM_PROMPT, user_prompt
                    )
                    st.rerun()
                except (LLMError, ValueError) as exc:
                    st.error(f"Failed to generate questions: {exc}")
        return

    questions = st.session_state.interview_questions
    if not isinstance(questions, dict):
        st.error("Invalid questions format. Please re-run.")
        return

    def _render_question_list(q_list: list, section_label: str):
        if not q_list:
            st.info(f"No {section_label} questions generated.")
            return
        for i, q in enumerate(q_list, 1):
            based_on = q.get("based_on", "").lower()
            icon = {"resume": "📄", "jd": "📋", "gap": "❓"}.get(based_on, "❓")
            with st.expander(f"{icon} Q{i}: {q.get('question', 'Question')}"):
                st.markdown(f"**Why this may be asked:** {q.get('why_asked', '')}")
                if q.get("missing_skill"):
                    st.markdown(
                        f'<div class="warn-note">⚠️ This question is about a skill '
                        f'<strong>not found in your resume</strong>: '
                        f'{q["missing_skill"]}</div>',
                        unsafe_allow_html=True,
                    )
                source_badge = {"resume": "📄 Resume-based", "jd": "📋 JD-based", "gap": "❓ Skill Gap"}.get(based_on, based_on)
                st.caption(f"Source: {source_badge}")

    tabs = st.tabs([
        "⚙️ Technical", "🤝 HR / Behavioral", "🛠️ Projects",
        "📄 Resume-based", "❓ Gap Questions"
    ])

    with tabs[0]:
        st.markdown("Questions about your technical skills and the JD's technical requirements.")
        _render_question_list(questions.get("technical_questions", []), "technical")

    with tabs[1]:
        st.markdown("Behavioral and culture-fit questions.")
        _render_question_list(questions.get("hr_questions", []), "HR")

    with tabs[2]:
        st.markdown("Questions about projects listed in your resume.")
        _render_question_list(questions.get("project_questions", []), "project")

    with tabs[3]:
        st.markdown("Questions directly based on items in your resume.")
        _render_question_list(questions.get("resume_based_questions", []), "resume-based")

    with tabs[4]:
        st.markdown(
            '<div class="warn-note">These are questions the interviewer may ask '
            'about skills <strong>not found in your resume</strong>. '
            'Prepare honest answers about your exposure to these areas.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        _render_question_list(questions.get("gap_questions", []), "gap")

    if st.button("🔄 Re-generate Questions", use_container_width=True):
        st.session_state.interview_questions = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────

page_map = {
    "📄 Resume Upload":      page_resume_upload,
    "📋 Job Description":    page_job_description,
    "🔍 Resume Analysis":    page_resume_analysis,
    "🤖 ATS Analysis":       page_ats_analysis,
    "🎯 JD Match":           page_jd_match,
    "📊 Skill Gap":          page_skill_gap,
    "✏️ Resume Improvement": page_resume_improvement,
    "🎤 Interview Prep":     page_interview_prep,
}

current_page = st.session_state.page
page_fn = page_map.get(current_page)
if page_fn:
    page_fn()
else:
    st.error(f"Unknown page: {current_page}")
