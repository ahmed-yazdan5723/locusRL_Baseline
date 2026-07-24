"""Small two-player Leduc Poker adapter from the LocusRL environment scaffold."""
import random
from typing import Optional

from adapters.base import Action, GameAdapter, Observation
from adapters.registry import register_env

RANKS = ("J", "Q", "K")
ACTION_CHECK_CALL = "check_call"
ACTION_BET_RAISE = "bet_raise"
ACTION_FOLD = "fold"


@register_env("leduc_poker")
class LeducPokerAdapter(GameAdapter):
    name = "leduc_poker"
    num_players = 2

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._episode_index = 0
        self._rng = random.Random(seed)
        self.deck = []
        self.private_cards = []
        self.public_card = None
        self.current_player_idx = 0
        self.round_index = 0
        self.terminal = False
        self.folded_player = None
        self.contributions = []
        self.round_bets = []
        self.checks_in_row = 0
        self.raises_this_round = 0
        self.pending_bet = False

    def reset(self) -> Observation:
        if self.seed is not None:
            self._rng = random.Random(self.seed + self._episode_index)
        self._episode_index += 1
        self.deck = [rank for rank in RANKS for _ in range(2)]
        self._rng.shuffle(self.deck)
        self.private_cards = [self.deck.pop(), self.deck.pop()]
        self.public_card = None
        self.current_player_idx = 0
        self.round_index = 0
        self.terminal = False
        self.folded_player = None
        self.contributions = [1, 1]
        self.round_bets = [0, 0]
        self.checks_in_row = 0
        self.raises_this_round = 0
        self.pending_bet = False
        return self._make_observation()

    def legal_actions(self):
        if self.terminal:
            return []
        if self.pending_bet:
            return [ACTION_CHECK_CALL, ACTION_FOLD]
        actions = [ACTION_CHECK_CALL]
        if self.raises_this_round < 1:
            actions.append(ACTION_BET_RAISE)
        return actions

    def action_space(self):
        return [ACTION_CHECK_CALL, ACTION_BET_RAISE, ACTION_FOLD]

    def step(self, action: Action) -> Observation:
        legal = self.legal_actions()
        if self.terminal:
            raise ValueError("Cannot step a terminal Leduc Poker game.")
        if action not in legal:
            raise ValueError(f"Illegal Leduc action {action!r}; legal={legal}.")

        player = self.current_player_idx
        info = {"player": player, "round": self.round_index, "action": action}
        if action == ACTION_FOLD:
            self.folded_player = player
            self.terminal = True
        elif action == ACTION_BET_RAISE:
            amount = self._bet_size()
            self.contributions[player] += amount
            self.round_bets[player] += amount
            self.raises_this_round += 1
            self.pending_bet = True
            self.checks_in_row = 0
            self.current_player_idx = 1 - player
        else:
            if self.pending_bet:
                owed = max(self.round_bets) - self.round_bets[player]
                self.contributions[player] += owed
                self.round_bets[player] += owed
                self.pending_bet = False
                self._advance_round_or_showdown()
            else:
                self.checks_in_row += 1
                if self.checks_in_row >= 2:
                    self._advance_round_or_showdown()
                else:
                    self.current_player_idx = 1 - player

        return self._make_observation(info=info)

    def _bet_size(self) -> int:
        return 2 if self.round_index == 0 else 4

    def _advance_round_or_showdown(self) -> None:
        if self.round_index == 0:
            self.round_index = 1
            self.public_card = self.deck.pop()
            self.round_bets = [0, 0]
            self.checks_in_row = 0
            self.raises_this_round = 0
            self.pending_bet = False
            self.current_player_idx = 0
        else:
            self.terminal = True

    def _showdown_winner(self) -> int:
        p0_pair = self.private_cards[0] == self.public_card
        p1_pair = self.private_cards[1] == self.public_card
        if p0_pair and not p1_pair:
            return 0
        if p1_pair and not p0_pair:
            return 1
        rank_value = {rank: index for index, rank in enumerate(RANKS)}
        return 0 if rank_value[self.private_cards[0]] >= rank_value[self.private_cards[1]] else 1

    def _returns(self):
        if not self.terminal:
            return (0.0, 0.0)
        pot = float(sum(self.contributions))
        winner = 1 - self.folded_player if self.folded_player is not None else self._showdown_winner()
        raw = [-float(self.contributions[0]), -float(self.contributions[1])]
        raw[winner] += pot
        return (raw[0], raw[1])

    def _make_observation(self, info=None) -> Observation:
        legal = self.legal_actions()
        action_space = self.action_space()
        action_mask = [1 if action in legal else 0 for action in action_space]
        numeric_state = {
            "private_card": None if self.terminal else self.private_cards[self.current_player_idx],
            "public_card": self.public_card,
            "round_index": self.round_index,
            "pot": sum(self.contributions),
            "contributions": self.contributions[:],
            "round_bets": self.round_bets[:],
            "pending_bet": self.pending_bet,
            "raises_this_round": self.raises_this_round,
            "folded_player": self.folded_player,
            "action_space": action_space,
            "action_mask": action_mask,
        }
        text_state = (
            "Game: Leduc Poker\n"
            f"Current player: {self.current_player_idx}\n"
            f"Private card: {numeric_state['private_card']}\n"
            f"Public card: {self.public_card}\n"
            f"Round: {'preflop' if self.round_index == 0 else 'postflop'}\n"
            f"Pot: {numeric_state['pot']}\n"
            f"Legal actions: {legal}\n"
            f"Pending bet to call: {self.pending_bet}\n"
            f"Respond with an action in the schema: {self.action_schema_example()}"
        )
        outcome = dict(enumerate(self._returns())) if self.terminal else None
        return Observation(
            numeric_state=numeric_state,
            text_state=text_state,
            legal_actions=legal,
            current_player=self.current_player_idx,
            done=self.terminal,
            outcome=outcome,
            info=info or {},
        )

    def action_schema_example(self) -> str:
        return '{"action": "check_call|bet_raise|fold"}'
