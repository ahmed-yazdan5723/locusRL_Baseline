"""Minimal, dependency-free Connect Four adapter.

This is a STAND-IN for Yuxiang's GameAdapter v0.1 (which will also cover
Leduc Poker and Goofspiel). It's here so the eval pipeline is runnable
today. It implements the same interface (adapters/base.py) so replacing
it later shouldn't require touching baseline.py, eval/, or agents/.
"""
import random
from typing import List, Optional

from adapters.base import GameAdapter, Observation
from adapters.registry import register_env

ROWS = 6
COLS = 7
EMPTY, P0, P1 = 0, 1, -1
SYMBOLS = {EMPTY: ".", P0: "X", P1: "O"}
PLAYER_TOKEN = {0: P0, 1: P1}


@register_env("connectfour")
class ConnectFourDemo(GameAdapter):
    name = "connectfour"
    num_players = 2

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._rng = random.Random(seed)
        self.board = None
        self.current_player_idx = 0
        self._done = False

    def reset(self) -> Observation:
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player_idx = 0
        self._done = False
        return self._make_observation()

    def _legal_columns(self) -> List[int]:
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def _drop(self, col: int, token: int) -> int:
        """Drop token into column, return the row it lands in."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = token
                return row
        raise ValueError(f"Column {col} is full")

    def _check_win(self, row: int, col: int, token: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == token:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def step(self, action: int) -> Observation:
        if self._done:
            raise ValueError("Episode already finished; call reset().")
        legal = self._legal_columns()
        if action not in legal:
            raise ValueError(f"Illegal action {action}; legal columns: {legal}")

        token = PLAYER_TOKEN[self.current_player_idx]
        row = self._drop(action, token)

        outcome = None
        if self._check_win(row, action, token):
            self._done = True
            outcome = {self.current_player_idx: 1.0, 1 - self.current_player_idx: -1.0}
        elif not self._legal_columns():
            self._done = True
            outcome = {0: 0.0, 1: 0.0}
        else:
            self.current_player_idx = 1 - self.current_player_idx

        return self._make_observation(outcome=outcome)

    def _render_text(self) -> str:
        lines = ["".join(SYMBOLS[cell] for cell in row) for row in self.board]
        lines.append("0123456")
        return "\n".join(lines)

    def _make_observation(self, outcome=None) -> Observation:
        legal = [] if self._done else self._legal_columns()
        text = (
            f"Connect Four board (X=player0, O=player1), you are player "
            f"{self.current_player_idx}.\n{self._render_text()}\n"
            f"Legal columns: {legal}\n"
            f'Respond with an action in the schema: {self.action_schema_example()}'
        )
        return Observation(
            numeric_state=[row[:] for row in self.board],
            text_state=text,
            legal_actions=legal,
            current_player=self.current_player_idx,
            done=self._done,
            outcome=outcome,
            info={},
        )

    def action_schema_example(self) -> str:
        return '{"action": "DROP", "column": <0-6>}'
