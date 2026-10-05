# Resume ATS & Job Description Analyzer

An AI-powered web application that analyzes a candidate's resume against a specific Job Description (JD), checks ATS compatibility, identifies skill gaps, and provides evidence-based recommendations.

---

## ⚠️ Important Disclaimer

> **The ATS score produced by this tool is an *estimated* compatibility score.**
> It is NOT a guarantee that any real company's ATS system will accept or reject your resume.
> Different ATS platforms (Greenhouse, Workday, Lever, Taleo, etc.) use different algorithms.
> Use this score as a directional guide, not an absolute metric.

---

## What the Application Does

```
RESUME
   ↓
TEXT EXTRACTION (PDF / DOCX)
   ↓
RESUME STRUCTURED DATA (name, education, experience, skills, projects, …)
   ↓
JOB DESCRIPTION ANALYSIS (required skills, preferred, keywords, …)
   ↓
REQUIREMENTS EXTRACTION
   ↓
EVIDENCE-BASED MATCHING (FOUND / PARTIAL / NOT FOUND / NOT VERIFIABLE)
   ↓
ATS ANALYSIS (structure, readability, keyword coverage, title alignment)
   ↓
SKILL GAP REPORT
   ↓
RESUME IMPROVEMENTS (evidence-based, never fabricates skills)
   ↓
INTERVIEW PREPARATION (technical, HR, project, gap questions)
```

---

## Architecture

```
resume-ats-analyzer/
│
├── app.py                  ← Streamlit 8-page dashboard (entry point)
├── requirements.txt
├── README.md
├── .env.example            ← Template — copy to .env and add your API key
├── .env                    ← YOUR secrets (git-ignored, never commit this)
├── .gitignore              ← Excludes .env from version control
│
├── utils/
│   ├── config.py           ← Central env loader (loads .env via absolute path)
│   ├── pdf_parser.py       ← PyMuPDF text extraction (handles image-only PDFs)
│   ├── docx_parser.py      ← python-docx text extraction
│   ├── resume_analyzer.py  ← LLM-based resume parsing → structured JSON
│   ├── jd_analyzer.py      ← LLM-based JD parsing → structured JSON
│   ├── ats_analyzer.py     ← ATS scoring (structure, readability, keywords, …)
│   ├── matcher.py          ← Evidence-based skill matching
│   └── llm_client.py       ← Multi-provider LLM wrapper (OpenAI/Gemini/Groq)
│
└── data/
    └── .gitkeep
```

---

## Installation & Setup

### 1. Enter the project directory

```bash
cd resume-ats-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file and add your API key

> ⚠️ This step is **required** before the LLM features will work.

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open the newly created `.env` file in any text editor and replace the placeholder:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your-real-key-here...
OPENAI_MODEL=gpt-4o-mini
```

**Where to get an API key:**
- OpenAI: https://platform.openai.com/api-keys
- Google Gemini: https://aistudio.google.com/app/apikey
- Groq (free): https://console.groq.com/keys

> **Security:** `.env` is listed in `.gitignore` and will never be committed.
> Never paste your API key into any source file or share it.

Edit `.env` and set your provider and API key:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

**Supported providers:**

| Provider | Env var | Model env var | Free tier? |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` | No |
| Google Gemini | `GOOGLE_API_KEY` | `GEMINI_MODEL` | Yes (limited) |
| Groq | `GROQ_API_KEY` | `GROQ_MODEL` | Yes (generous) |

### 5. Run the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to Use

| Page | What to do |
|---|---|
| **📄 Resume Upload** | Upload PDF/DOCX or paste resume text → click "Analyze Resume with AI" |
| **📋 Job Description** | Paste the JD or upload it → click "Parse JD with AI" |
| **🔍 Resume Analysis** | View structured sections extracted from your resume |
| **🤖 ATS Analysis** | Run full ATS scoring with methodology explained |
| **🎯 JD Match** | Evidence-based requirement matching table |
| **📊 Skill Gap** | See matching, partial, and missing skills |
| **✏️ Resume Improvement** | Get specific recommendations (never adds fake skills) |
| **🎤 Interview Prep** | Technical, HR, project, and gap questions |

---

## How Resume Analysis Works

1. Text is extracted from the uploaded file using PyMuPDF (PDF) or python-docx (DOCX).
2. The extracted text is sent to the configured LLM with a strict prompt that forbids inventing information.
3. The LLM returns a structured JSON with: name, contact, education, experience, internships, projects, technical skills, soft skills, certifications, languages, tools, achievements, and keywords.
4. The JSON is validated and defaults are filled for any missing keys.

**Evidence rule:** A skill is only listed as present if it is explicitly written in the resume. No inference is made from job titles, company names, or project descriptions alone.

---

## How ATS Scoring Works

The ATS score is a weighted average of 6 components:

| Component | Weight | How measured |
|---|---|---|
| JD Keyword Coverage | 25% | % of JD keywords found verbatim in resume |
| Required Skill Coverage | 25% | % of required skills with FOUND or PARTIAL match |
| Resume Structure | 20% | Presence of standard sections (contact, education, experience, skills, etc.) |
| ATS Readability | 15% | Absence of tables, image artefacts, excessive special characters |
| Job Title Alignment | 10% | JD job title keywords found in resume |
| Content Quality | 5% | LLM assessment of action verbs, metrics, vague statements |

**Score ranges:**
- 80–100: Excellent
- 60–79: Good
- 40–59: Fair
- 0–39: Needs Improvement

---

## Limitations

- **Image-only PDFs (scanned documents):** Text cannot be extracted. Use OCR first.
- **Complex layouts:** Multi-column PDFs, tables, and text boxes may not extract cleanly.
- **LLM hallucination:** The system uses strict prompts to prevent fabrication, but LLMs are not perfect. Always verify the output.
- **ATS score is an estimate:** Real ATS systems vary significantly. This tool gives a directional score, not a precise prediction.
- **Language support:** Best results with English resumes and JDs.
- **File size limit:** 10 MB per file.

---

## License

MIT
