"""Abstract agent interface. Every baseline (Legal-Random, rule agents,
Prompt-Only LLM, SFT-Only LLM, ...) implements this same small surface,
so eval/runner.py never needs to know which kind of agent it's driving.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from adapters.base import Action, Observation


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, seed: Optional[int] = None, checkpoint_path: Optional[str] = None,
                 **kwargs):
        self.seed = seed
        self.checkpoint_path = checkpoint_path

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Return a legal action id given the current observation.
        Implementations should try to only return values in
        obs.legal_actions; the eval runner treats anything else as an
        illegal-action event and substitutes a random legal fallback.
        """

    def reset(self) -> None:
        """Optional per-episode reset hook (e.g. clear conversation history)."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Serializable provenance for result files."""
        return {
            "name": self.name,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "model": getattr(self, "model_name", None),
            "temperature": getattr(self, "temperature", None),
            "top_p": getattr(self, "top_p", None),
            "ensemble": getattr(self, "ensemble", None),
            "mask_actions": getattr(self, "mask_actions", None),
            "max_retries": getattr(self, "max_retries", None),
        }

    def consume_action_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics for the most recent action, if any."""
        return {}
