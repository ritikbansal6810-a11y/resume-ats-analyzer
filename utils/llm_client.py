"""
llm_client.py
─────────────
Thin wrapper around multiple LLM providers (OpenAI, Gemini, Groq).

Configuration is managed entirely by utils/config.py, which loads the
project-root .env file using an absolute path before any key is accessed.

Usage:
    from utils.llm_client import call_llm, call_llm_json, LLMError
    response_text = call_llm(system_prompt, user_prompt)

Security:
    - API keys are read once inside each _call_* function via config._get_api_key().
    - Keys are NEVER logged, stored in variables visible outside the function,
      or included in error messages / exceptions.
"""

from __future__ import annotations

import json
import re

# config.py loads .env at import time using an absolute path — must be first.
from utils.config import (
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_GROQ,
    _get_api_key,        # internal — key value stays inside this module
    get_provider,
    get_model,
    is_configured,
    get_safe_provider_info,
    validate_config,
)


class LLMError(Exception):
    """Raised when an LLM call fails or returns unusable output."""


# ──────────────────────────────────────────────────────────────────────────
# Provider implementations
# ──────────────────────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError(
            "openai package is not installed. Run: pip install openai"
        )

    api_key = _get_api_key(PROVIDER_OPENAI)
    if not api_key:
        cfg = validate_config()
        raise LLMError(cfg["error"] or "OpenAI API key is not configured.")

    model = get_model()
    # The OpenAI client is constructed per-call so it always picks up the
    # latest key (handles hot-reload in development).
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        # Sanitise the error: OpenAI sometimes echoes the key in auth errors.
        # Replace any sk-... pattern to be safe.
        safe_msg = re.sub(r"sk-[A-Za-z0-9\-_]{10,}", "sk-***REDACTED***", str(exc))
        raise LLMError(f"OpenAI API error: {safe_msg}") from exc


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise LLMError(
            "google-generativeai package is not installed. "
            "Run: pip install google-generativeai"
        )

    api_key = _get_api_key(PROVIDER_GEMINI)
    if not api_key:
        cfg = validate_config()
        raise LLMError(cfg["error"] or "Google API key is not configured.")

    model_name = get_model()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )

    try:
        response = model.generate_content(user_prompt)
        return response.text or ""
    except Exception as exc:
        raise LLMError(f"Gemini API error: {exc}") from exc


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise LLMError(
            "groq package is not installed. Run: pip install groq"
        )

    api_key = _get_api_key(PROVIDER_GROQ)
    if not api_key:
        cfg = validate_config()
        raise LLMError(cfg["error"] or "Groq API key is not configured.")

    model = get_model()
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise LLMError(f"Groq API error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM provider and return the raw response string.
    Raises LLMError on failure.
    """
    provider = get_provider()

    if provider == PROVIDER_OPENAI:
        return _call_openai(system_prompt, user_prompt)
    elif provider == PROVIDER_GEMINI:
        return _call_gemini(system_prompt, user_prompt)
    elif provider == PROVIDER_GROQ:
        return _call_groq(system_prompt, user_prompt)
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: openai, gemini, groq"
        )


def call_llm_json(system_prompt: str, user_prompt: str) -> dict | list:
    """
    Call the LLM and parse the response as JSON.
    Strips markdown code fences if present.
    Raises LLMError if the response is not valid JSON.
    """
    raw = call_llm(system_prompt, user_prompt)

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Fallback: try to extract a JSON object or array with a regex
        json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        raise LLMError(
            f"LLM response was not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"First 500 chars of response:\n{raw[:500]}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────────
# Convenience re-exports so existing import sites don't need to change
# ──────────────────────────────────────────────────────────────────────────

def is_llm_configured() -> bool:
    """Return True if the active provider has a valid API key."""
    return is_configured()


def get_provider_info() -> dict:
    """Return safe provider info (no API key values)."""
    return get_safe_provider_info()
