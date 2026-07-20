"""Generic action-as-text LLM policy agent (the "Prompt-Only LLM" baseline
in WP1). Concrete backends (DeepSeek, Qwen, ...) only need to implement
`_call_backend(prompt) -> str`; parsing/fallback/retry logic lives here
once so every model benefits from the same robustness and every model
plugs into the same evaluation path.

This is deliberately the seam described in WP2: action representation
(prompt schema + parsing) is centralized here so later Masked Action
Augmentation / Ensemble Prediction work only needs to hook into
`act()`, not be re-implemented per model.
"""
import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from adapters.base import Observation
from agents.base import BaseAgent
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMPolicyAgent(BaseAgent):
    """Abstract base for LLM-as-policy baselines. Subclass and implement
    `_call_backend`. Do not register this class itself.
    """
    model_name: str = "unset"
    system_prompt: str = (
        "You are playing a two-player game. You will be given the board "
        "state as text and a list of legal actions. Reply with ONLY a "
        "single JSON object describing your chosen action, no extra text."
    )

    def __init__(self, seed: Optional[int] = None, checkpoint_path: Optional[str] = None,
                 max_retries: int = 2, temperature: float = 0.7,
                 top_p: float = 0.95, ensemble: int = 1,
                 mask_actions: bool = True, **kwargs):
        super().__init__(seed=seed, checkpoint_path=checkpoint_path, **kwargs)
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.ensemble = ensemble
        self.mask_actions = mask_actions
        self._last_action_diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def _call_backend(self, prompt: str) -> str:
        """Send `prompt` to the model, return the raw text response.
        Should raise on hard failure (e.g. network/auth error) rather
        than returning garbage — act() below decides what to do with
        that failure (fallback to random legal action)."""

    def _build_prompt(self, obs: Observation) -> str:
        return obs.text_state

    def _parse_action(self, response_text: str, legal_actions: List[int]) -> Tuple[Optional[int], Dict[str, int]]:
        """Best-effort parse of the model's JSON action into a column/id.
        Accepts either {"action": "DROP", "column": N} or a bare integer,
        since different games will have different schemas — subclasses
        can override for game-specific schemas.
        """
        diagnostics = {
            "parser_failures": 0,
            "invalid_json": 0,
        }
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                for key in ("column", "value", "action_id", "id"):
                    if key in obj and isinstance(obj[key], int):
                        return obj[key], diagnostics
            except (json.JSONDecodeError, TypeError):
                diagnostics["invalid_json"] += 1
        # fallback: first standalone integer in the response
        num_match = re.search(r"-?\d+", response_text)
        if num_match:
            return int(num_match.group(0)), diagnostics
        diagnostics["parser_failures"] += 1
        return None, diagnostics

    def act(self, obs: Observation) -> int:
        """Tries up to max_retries+1 times to get a legal action from the
        model. If it never manages to, this returns the last best-effort
        parse (or a sentinel if nothing parsed at all) WITHOUT silently
        substituting a legal action itself — eval/runner.py is the single
        source of truth for illegal-action counting and fallback, so that
        metric actually reflects model failures instead of being masked
        by per-agent self-correction.
        """
        prompt = f"{self.system_prompt}\n\n{self._build_prompt(obs)}"

        parsed_action = None
        diagnostics = {
            "retry_count": 0,
            "parser_failures": 0,
            "invalid_json": 0,
            "backend_failures": 0,
            "api_calls": 0,
        }
        for attempt in range(self.max_retries + 1):
            try:
                diagnostics["api_calls"] += 1
                response = self._call_backend(prompt)
            except Exception as exc:  # network/auth/etc.
                diagnostics["backend_failures"] += 1
                diagnostics["retry_count"] = min(attempt + 1, self.max_retries)
                logger.warning("%s backend call failed (attempt %d/%d): %s",
                                self.model_name, attempt + 1, self.max_retries + 1, exc)
                continue
            parsed_action, parse_diagnostics = self._parse_action(response, obs.legal_actions)
            diagnostics["parser_failures"] += parse_diagnostics["parser_failures"]
            diagnostics["invalid_json"] += parse_diagnostics["invalid_json"]
            if parsed_action in obs.legal_actions:
                diagnostics["retry_count"] = attempt
                self._last_action_diagnostics = diagnostics
                return parsed_action
            diagnostics["retry_count"] = min(attempt + 1, self.max_retries)
            logger.debug("%s produced non-legal/unparseable action %r, retrying",
                          self.model_name, parsed_action)

        logger.info("%s exhausted retries without a legal action (last parse: %r); "
                     "runner will record this as illegal and substitute a fallback.",
                     self.model_name, parsed_action)
        self._last_action_diagnostics = diagnostics
        return parsed_action if parsed_action is not None else -1

    def consume_action_diagnostics(self) -> Dict[str, Any]:
        diagnostics = self._last_action_diagnostics
        self._last_action_diagnostics = {}
        return diagnostics
