"""
config.py
─────────
Central environment configuration for the Resume ATS Analyzer.

ALWAYS import and call ensure_env_loaded() before reading any os.getenv() value.

The .env file is resolved relative to THIS file's parent's parent directory
(i.e., resume-ats-analyzer/.env), so it is found correctly regardless of
the working directory from which Streamlit or Python is launched.

Security rules enforced here:
  - API key values are NEVER logged, printed, or returned to callers.
  - validate_config() returns human-readable status only (key present/absent).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# ── Resolve the project root (.env lives here) ─────────────────────────────
# utils/config.py  →  utils/  →  resume-ats-analyzer/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"

# ── Load once at import time using the resolved absolute path ──────────────
# override=False: never overwrite variables already set in the real environment
# (e.g. CI secrets, Docker env, system env vars always win)
_env_loaded: bool = False


def ensure_env_loaded() -> None:
    """
    Load the .env file from the project root.
    Safe to call multiple times — only loads once.
    """
    global _env_loaded
    if _env_loaded:
        return

    if _DOTENV_PATH.exists():
        load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
        _env_loaded = True
    else:
        # .env not found — warn once but do not crash.
        # Variables may still be set via real environment (Docker, CI, etc.)
        _env_loaded = True  # mark as attempted so we don't loop


# Load immediately when this module is first imported.
ensure_env_loaded()


# ── Provider constants ─────────────────────────────────────────────────────
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_GROQ = "groq"
VALID_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_GROQ}

# Maps provider → environment variable name for its API key
PROVIDER_KEY_ENVVAR: dict[str, str] = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_GEMINI: "GOOGLE_API_KEY",
    PROVIDER_GROQ:   "GROQ_API_KEY",
}

# Maps provider → environment variable name for its model
PROVIDER_MODEL_ENVVAR: dict[str, str] = {
    PROVIDER_OPENAI: "OPENAI_MODEL",
    PROVIDER_GEMINI: "GEMINI_MODEL",
    PROVIDER_GROQ:   "GROQ_MODEL",
}

# Default model for each provider (used when the env var is absent)
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_GEMINI: "gemini-1.5-flash",
    PROVIDER_GROQ:   "llama-3.1-8b-instant",
}


# ── Public accessors (never expose raw key values) ─────────────────────────

def get_provider() -> str:
    """Return the active LLM provider name (lower-case)."""
    ensure_env_loaded()
    return os.getenv("LLM_PROVIDER", PROVIDER_OPENAI).lower().strip()


def get_model() -> str:
    """Return the model name for the active provider."""
    ensure_env_loaded()
    provider = get_provider()
    env_var = PROVIDER_MODEL_ENVVAR.get(provider, "")
    default = PROVIDER_DEFAULT_MODEL.get(provider, "unknown")
    return os.getenv(env_var, default) if env_var else default


def _get_api_key(provider: str) -> str:
    """
    Return the raw API key string for the given provider.
    INTERNAL USE ONLY — never pass the result to logs, UI, or external callers.
    """
    ensure_env_loaded()
    env_var = PROVIDER_KEY_ENVVAR.get(provider, "")
    return os.getenv(env_var, "") if env_var else ""


def is_configured() -> bool:
    """Return True if the active provider has a non-empty API key."""
    ensure_env_loaded()
    provider = get_provider()
    return bool(_get_api_key(provider))


def validate_config() -> dict[str, object]:
    """
    Return a safe configuration status dict.
    Values are booleans / strings — NEVER the actual API key.

    Example return:
        {
            "provider": "openai",
            "provider_valid": True,
            "api_key_present": True,
            "model": "gpt-4o-mini",
            "dotenv_path": "/abs/path/to/.env",
            "dotenv_exists": True,
            "error": None,
        }
    """
    ensure_env_loaded()
    provider = get_provider()
    api_key_present = bool(_get_api_key(provider))
    model = get_model()
    provider_valid = provider in VALID_PROVIDERS

    error: str | None = None
    if not provider_valid:
        error = (
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: {', '.join(sorted(VALID_PROVIDERS))}."
        )
    elif not api_key_present:
        key_var = PROVIDER_KEY_ENVVAR.get(provider, "API_KEY")
        error = (
            f"{provider.capitalize()} API key is not configured. "
            f"Add {key_var}=<your-key> to the project's .env file "
            f"(located at: {_DOTENV_PATH}) and restart Streamlit."
        )

    return {
        "provider": provider,
        "provider_valid": provider_valid,
        "api_key_present": api_key_present,
        "model": model,
        "dotenv_path": str(_DOTENV_PATH),
        "dotenv_exists": _DOTENV_PATH.exists(),
        "error": error,
    }


def get_safe_provider_info() -> dict[str, object]:
    """
    Alias used by the Streamlit UI — returns safe display information only.
    No API key values are included.
    """
    cfg = validate_config()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "configured": cfg["api_key_present"] and cfg["provider_valid"],
        "dotenv_exists": cfg["dotenv_exists"],
        "dotenv_path": cfg["dotenv_path"],
        "error": cfg["error"],
    }
