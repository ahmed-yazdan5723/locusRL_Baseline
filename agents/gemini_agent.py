"""Gemini backend for the LLM-policy baseline.

Uses the Gemini generateContent REST endpoint. Set GEMINI_API_KEY or
GOOGLE_API_KEY in the environment before running this agent.
"""
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agents.llm_agent import LLMPolicyAgent
from agents.registry import register_agent
from utils.logging_utils import get_logger

logger = get_logger(__name__)

GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


@register_agent("gemini")
class GeminiAgent(LLMPolicyAgent):
    name = "gemini"
    model_name = "gemini-2.0-flash"

    def __init__(self, seed=None, checkpoint_path=None, **kwargs):
        super().__init__(seed=seed, checkpoint_path=checkpoint_path, **kwargs)
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._warned_no_key = False

    def _call_backend(self, prompt: str) -> str:
        if not self.api_key:
            if not self._warned_no_key:
                logger.warning(
                    "GEMINI_API_KEY or GOOGLE_API_KEY is not set; GeminiAgent will "
                    "fall back through the shared illegal-action path. Set one of "
                    "those env vars to get real model actions."
                )
                self._warned_no_key = True
            raise RuntimeError("Gemini API key not set")

        request_body = json.dumps(
            {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "topP": self.top_p,
                    "maxOutputTokens": 80,
                },
            }
        ).encode("utf-8")
        request = Request(
            GEMINI_API_URL_TEMPLATE.format(model=self.model_name),
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

        return "".join(part.get("text", "") for part in parts)
