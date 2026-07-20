"""Minimal rule-based baseline. Stands in for the "rule/search agents"
bullet in WP1 until a real solver/search reference is wired in. Prefers
the most central legal action, which is a stronger-than-random opening
heuristic for Connect Four and a harmless generic tie-break elsewhere.
"""
from adapters.base import Observation
from agents.base import BaseAgent
from agents.registry import register_agent


@register_agent("heuristic")
class HeuristicAgent(BaseAgent):
    name = "heuristic"

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        center = sum(legal) / len(legal)
        return min(legal, key=lambda a: abs(a - center))
