"""DeepSeek backend for the LLM-policy baseline.

Uses DeepSeek's OpenAI-compatible chat completions endpoint. Needs
DEEPSEEK_API_KEY in the environment; if it's missing (e.g. running this
demo without credentials) the agent logs a warning once and the shared
LLMPolicyAgent.act() fallback logic takes over so the eval run still
completes instead of crashing.
"""
import os

from agents.llm_agent import LLMPolicyAgent
from agents.registry import register_agent
from utils.logging_utils import get_logger

logger = get_logger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


@register_agent("deepseek")
class DeepSeekAgent(LLMPolicyAgent):
    name = "deepseek"
    model_name = "deepseek-chat"

    def __init__(self, seed=None, checkpoint_path=None, **kwargs):
        super().__init__(seed=seed, checkpoint_path=checkpoint_path, **kwargs)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self._warned_no_key = False

    def _call_backend(self, prompt: str) -> str:
        if not self.api_key:
            if not self._warned_no_key:
                logger.warning(
                    "DEEPSEEK_API_KEY is not set — DeepSeekAgent will fall back to "
                    "random legal actions for this whole run. Set the env var to "
                    "get real model actions."
                )
                self._warned_no_key = True
            raise RuntimeError("DEEPSEEK_API_KEY not set")

        import requests  # imported lazily so the package still imports without it

        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 50,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
