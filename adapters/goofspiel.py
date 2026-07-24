"""Sequential-interface Goofspiel adapter from the LocusRL scaffold."""
import random
from typing import Optional

from adapters.base import Action, GameAdapter, Observation
from adapters.registry import register_env


@register_env("goofspiel")
class GoofspielAdapter(GameAdapter):
    name = "goofspiel"
    num_players = 2

    def __init__(self, seed: Optional[int] = None, deck_size: int = 5):
        super().__init__(seed)
        self.deck_size = deck_size
        self._episode_index = 0
        self._rng = random.Random(seed)
        self.prize_deck = []
        self.hands = []
        self.scores = []
        self.current_player_idx = 0
        self.round_index = 0
        self.pending_bid_p0 = None

    def reset(self) -> Observation:
        if self.seed is not None:
            self._rng = random.Random(self.seed + self._episode_index)
        self._episode_index += 1
        self.prize_deck = list(range(1, self.deck_size + 1))
        self._rng.shuffle(self.prize_deck)
        self.hands = [set(range(1, self.deck_size + 1)), set(range(1, self.deck_size + 1))]
        self.scores = [0.0, 0.0]
        self.current_player_idx = 0
        self.round_index = 0
        self.pending_bid_p0 = None
        return self._make_observation()

    def legal_actions(self):
        if self._is_terminal():
            return []
        return sorted(self.hands[self.current_player_idx])

    def action_space(self):
        return list(range(1, self.deck_size + 1))

    def step(self, action: Action) -> Observation:
        if self._is_terminal():
            raise ValueError("Cannot step a terminal Goofspiel game.")
        if not isinstance(action, int):
            raise ValueError(f"Goofspiel action must be an int bid, got {action!r}.")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal Goofspiel bid {action} for player {self.current_player_idx}.")

        player = self.current_player_idx
        self.hands[player].remove(action)
        info = {"player": player, "bid": action}
        if player == 0:
            self.pending_bid_p0 = action
            self.current_player_idx = 1
        else:
            if self.pending_bid_p0 is None:
                raise RuntimeError("Goofspiel player 1 acted before player 0 bid was stored.")
            prize = self.prize_deck[self.round_index]
            bid0, bid1 = self.pending_bid_p0, action
            if bid0 > bid1:
                self.scores[0] += prize
            elif bid1 > bid0:
                self.scores[1] += prize
            else:
                self.scores[0] += prize / 2.0
                self.scores[1] += prize / 2.0
            info.update({"prize": prize, "bid0": bid0, "bid1": bid1})
            self.round_index += 1
            self.pending_bid_p0 = None
            self.current_player_idx = 0

        return self._make_observation(info=info)

    def _is_terminal(self) -> bool:
        return self.round_index >= self.deck_size

    def _returns(self):
        diff = self.scores[0] - self.scores[1]
        if diff > 0:
            return (1.0, -1.0)
        if diff < 0:
            return (-1.0, 1.0)
        return (0.0, 0.0)

    def _make_observation(self, info=None) -> Observation:
        done = self._is_terminal()
        legal = self.legal_actions()
        current_prize = None if done else self.prize_deck[self.round_index]
        action_space = self.action_space()
        action_mask = [1 if action in legal else 0 for action in action_space]
        numeric_state = {
            "deck_size": self.deck_size,
            "round_index": self.round_index,
            "current_prize": current_prize,
            "revealed_completed_prizes": self.prize_deck[: self.round_index],
            "scores": self.scores[:],
            "hands": [sorted(hand) for hand in self.hands],
            "pending_bid_p0": self.pending_bid_p0,
            "action_space": action_space,
            "action_mask": action_mask,
        }
        text_state = (
            "Game: Goofspiel\n"
            f"Current player: {self.current_player_idx}\n"
            f"Round: {self.round_index + 1}/{self.deck_size}\n"
            f"Current prize: {current_prize}\n"
            f"Scores: P0={self.scores[0]}, P1={self.scores[1]}\n"
            f"Your legal bids: {legal}\n"
            f"Pending player-0 bid: {self.pending_bid_p0}\n"
            f"Respond with an action in the schema: {self.action_schema_example()}"
        )
        outcome = dict(enumerate(self._returns())) if done else None
        return Observation(
            numeric_state=numeric_state,
            text_state=text_state,
            legal_actions=legal,
            current_player=self.current_player_idx,
            done=done,
            outcome=outcome,
            info=info or {},
        )

    def action_schema_example(self) -> str:
        return '{"action": "BID", "bid": <card-in-hand>}'
