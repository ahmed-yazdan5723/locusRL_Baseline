"""Minimal rule-based baseline. Stands in for the "rule/search agents"
bullet in WP1 until a real solver/search reference is wired in. Prefers
the most central legal action, which is a stronger-than-random opening
heuristic for Connect Four and a harmless generic tie-break elsewhere.
"""
from adapters.base import Action, Observation
from agents.base import BaseAgent
from agents.registry import register_agent


@register_agent("heuristic")
class HeuristicAgent(BaseAgent):
    name = "heuristic"

    def act(self, obs: Observation) -> Action:
        legal = obs.legal_actions
        if all(isinstance(action, (int, float)) for action in legal):
            center = sum(legal) / len(legal)
            return min(legal, key=lambda action: abs(action - center))

        preferred = ("check_call", "bet_raise", "fold")
        for action in preferred:
            if action in legal:
                return action
        return legal[0]
