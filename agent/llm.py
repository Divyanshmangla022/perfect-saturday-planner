"""Thin wrapper around the Gemini API (google-genai SDK).

Exposes two calls the rest of the agent needs:
  * complete_json  — structured output validated against a Pydantic schema
  * complete_text  — free-form text

Both retry transient failures and raise a single LLMError on give-up so callers
have one exception type to handle.
"""

from __future__ import annotations

import os
import time
from typing import Type, TypeVar

from pydantic import BaseModel

try:  # the SDK is optional at import time so unit tests can run without it
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai = None
    types = None
    genai_errors = None

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_MAX_RETRIES = 3


class LLMError(Exception):
    """Any failure talking to the language model."""


def _get_api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError(
            "No Gemini API key found. Set GEMINI_API_KEY in your environment "
            "or Streamlit secrets. Get a free key at https://aistudio.google.com/apikey"
        )
    return key


class LLMClient:
    """Stateless-ish Gemini client. One instance per session is plenty."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        if genai is None:
            raise LLMError(
                "The 'google-genai' package is not installed. Run: pip install -r requirements.txt"
            )
        self.model = model
        self._client = genai.Client(api_key=_get_api_key(api_key))

    # -- internal -----------------------------------------------------------
    def _generate(self, contents: str, config) -> "types.GenerateContentResponse":
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except Exception as exc:  # noqa: BLE001 - normalise everything to LLMError
                last_exc = exc
                # Retry only on transient / rate-limit style errors.
                transient = genai_errors is not None and isinstance(
                    exc, (genai_errors.ServerError, genai_errors.APIError)
                )
                if attempt < _MAX_RETRIES - 1 and transient:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
        raise LLMError(f"Gemini request failed: {last_exc}") from last_exc

    # -- public -------------------------------------------------------------
    def complete_json(self, prompt: str, schema: Type[T], *,
                      system: str = "", temperature: float = 0.4) -> T:
        """Return a validated instance of `schema` from the model."""
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self._generate(prompt, config)
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        # Fall back to manual validation if the SDK didn't auto-parse.
        try:
            return schema.model_validate_json(resp.text)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Model did not return valid {schema.__name__}: {exc}") from exc
