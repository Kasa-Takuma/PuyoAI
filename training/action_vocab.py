"""Shared action vocabulary for learned policies."""

from __future__ import annotations

BOARD_WIDTH = 6

ACTION_KEYS = []

for column in range(BOARD_WIDTH):
    ACTION_KEYS.append(f"UP:{column}")
    ACTION_KEYS.append(f"DOWN:{column}")

for column in range(BOARD_WIDTH - 1):
    ACTION_KEYS.append(f"RIGHT:{column}")

for column in range(1, BOARD_WIDTH):
    ACTION_KEYS.append(f"LEFT:{column}")

ACTION_TO_INDEX = {action_key: index for index, action_key in enumerate(ACTION_KEYS)}


def action_index(action_key: str) -> int:
    """Return the fixed action index used by the learned policy."""

    return ACTION_TO_INDEX[action_key]
