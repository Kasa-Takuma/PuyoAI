"""Small fixed-seed regression check recorded from the JavaScript v12AC search."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .evolve_profiles import BASE_PROFILE, pair_stream
from .jax_board import resolve_turn
from .jax_profile_search import search_profile_batch


def main() -> None:
    expected_actions = [13, 6, 17, 21, 5]
    expected_scores = [2572.5354666666667, 2923.8688, 11859.8192, 11978.395466666667, 12381.2012]
    pairs = pair_stream("parity-5", len(expected_actions))
    board = jnp.zeros((1, 6, 14), dtype=jnp.uint8)
    actual = []
    for turn, expected_action in enumerate(expected_actions):
        actions, scores, _ = search_profile_batch(
            board,
            jnp.asarray(pairs[turn : turn + 3][None]),
            jnp.asarray(BASE_PROFILE[None]),
            beam_width=24,
        )
        action = int(np.asarray(actions)[0])
        score = float(np.asarray(scores)[0, action]) * 1000
        if action != expected_action:
            raise AssertionError(
                f"turn {turn + 1}: action {action} != JavaScript {expected_action}"
            )
        if abs(score - expected_scores[turn]) > 1e-5:
            raise AssertionError(
                f"turn {turn + 1}: score {score} != JavaScript {expected_scores[turn]}"
            )
        resolved = resolve_turn(
            board[0],
            jnp.uint8(pairs[turn, 0]),
            jnp.uint8(pairs[turn, 1]),
            jnp.int32(action),
        )
        board = resolved[0][None]
        actual.append({"turn": turn + 1, "action": action, "score": score})
    print({"matched": True, "turns": actual})


if __name__ == "__main__":
    main()
