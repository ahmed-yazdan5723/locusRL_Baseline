"""Abstract GameAdapter interface.

This mirrors what the management doc specifies for GameAdapter v0.1:
each adapter must expose numeric state, text state, legal actions, the
true terminal outcome, and current-player info, and the same test/eval
code must be able to switch between environments.

When Yuxiang's real GameAdapter lands, it should satisfy this same
interface (or we adjust this file once, not every agent/eval file).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    numeric_state: Any                 # e.g. board as array/grid
    text_state: str                    # canonical text rendering for LLM agents
    legal_actions: List[int]           # list of legal action ids for current_player
    current_player: int                # whose turn it is, 0-indexed
    done: bool                         # episode terminated
    outcome: Optional[Dict[int, float]] = None   # {player_id: return}, set when done
    info: Dict[str, Any] = field(default_factory=dict)  # adapter-specific extras


class GameAdapter(ABC):
    """One instance = one game episode. Call reset() to (re)start."""

    name: str = "base"
    num_players: int = 2

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    @abstractmethod
    def reset(self) -> Observation:
        """Start a new episode, return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> Observation:
        """Apply `action` for the current player, return the next observation.
        Implementations should raise ValueError on an illegal action rather
        than silently ignoring it — the eval runner is responsible for
        catching that and logging an illegal-action event.
        """

    def action_schema_example(self) -> str:
        """Human-readable example of the canonical action-as-text schema
        for this game (used to build LLM-agent prompts). Override per game.
        """
        return '{"action": "<TYPE>", "value": <int>}'

    def close(self) -> None:
        """Optional cleanup hook."""
        pass
