"""Legal-Random baseline: the floor every other agent must clear."""
import random

from adapters.base import Observation
from agents.base import BaseAgent
from agents.registry import register_agent


@register_agent("random")
class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)

    def act(self, obs: Observation) -> int:
        return self._rng.choice(obs.legal_actions)
