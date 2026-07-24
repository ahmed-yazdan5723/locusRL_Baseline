"""Connect Four adapter aligned with the LocusRL scaffold API."""
import random
from typing import List, Optional

from adapters.base import Action, GameAdapter, Observation
from adapters.registry import register_env

ROWS = 6
COLUMNS = 7
EMPTY = -1
SYMBOLS = {EMPTY: ".", 0: "X", 1: "O"}


@register_env("connect_four")
class ConnectFourAdapter(GameAdapter):
    name = "connect_four"
    num_players = 2

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._rng = random.Random(seed)
        self._episode_index = 0
        self.board: List[List[int]] = []
        self.current_player_idx = 0
        self.winner = None
        self.move_count = 0

    def reset(self) -> Observation:
        if self.seed is not None:
            self._rng = random.Random(self.seed + self._episode_index)
        self._episode_index += 1
        self.board = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.current_player_idx = 0
        self.winner = None
        self.move_count = 0
        return self._make_observation()

    def legal_actions(self) -> List[int]:
        if self._is_terminal():
            return []
        return [col for col in range(COLUMNS) if self.board[0][col] == EMPTY]

    def action_space(self) -> List[int]:
        return list(range(COLUMNS))

    def step(self, action: Action) -> Observation:
        if self._is_terminal():
            raise ValueError("Cannot step a terminal Connect Four game.")
        if not isinstance(action, int):
            raise ValueError(f"Connect Four action must be an int column, got {action!r}.")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal Connect Four column: {action}.")

        player = self.current_player_idx
        placed_row = None
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][action] == EMPTY:
                self.board[row][action] = player
                placed_row = row
                break
        if placed_row is None:
            raise ValueError(f"Column {action} is full.")

        self.move_count += 1
        if self._has_four(placed_row, action, player):
            self.winner = player
        elif not self._is_terminal():
            self.current_player_idx = 1 - self.current_player_idx

        info = {"placed_row": placed_row, "player": player}
        return self._make_observation(info=info)

    def _is_terminal(self) -> bool:
        return self.winner is not None or self.move_count == ROWS * COLUMNS

    def _returns(self):
        if self.winner is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if self.winner == 0 else (-1.0, 1.0)

    def _make_observation(self, info=None) -> Observation:
        done = self._is_terminal()
        legal = self.legal_actions()
        outcome = dict(enumerate(self._returns())) if done else None
        action_space = self.action_space()
        action_mask = [1 if action in legal else 0 for action in action_space]
        numeric_state = {
            "board": [row[:] for row in self.board],
            "board_legend": {"empty": EMPTY, "player_0": 0, "player_1": 1},
            "move_count": self.move_count,
            "winner": self.winner,
            "action_space": action_space,
            "action_mask": action_mask,
        }
        text_state = (
            "Game: Connect Four\n"
            f"Current player: {self.current_player_idx}\n"
            f"Legal columns: {legal}\n"
            "Board top-to-bottom:\n"
            f"{self._render_text()}\n"
            f"Respond with an action in the schema: {self.action_schema_example()}"
        )
        return Observation(
            numeric_state=numeric_state,
            text_state=text_state,
            legal_actions=legal,
            current_player=self.current_player_idx,
            done=done,
            outcome=outcome,
            info=info or {},
        )

    def _render_text(self) -> str:
        return "\n".join(" ".join(SYMBOLS[cell] for cell in row) for row in self.board)

    def _has_four(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= 4:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLUMNS and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def action_schema_example(self) -> str:
        return '{"action": "DROP", "column": <0-6>}'
