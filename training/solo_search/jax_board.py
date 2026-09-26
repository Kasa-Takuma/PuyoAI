"""JAX implementation of the current JavaScript board rules.

Boards use uint8 values and shape ``(6, 14)``.  The first dimension is the
column and y=0 is the bottom, matching ``src/core/fast-board.js``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BOARD_WIDTH = 6
BOARD_HEIGHT = 14
VISIBLE_HEIGHT = 12
STORAGE_HEIGHT = 13
ACTION_COUNT = 22
MAX_CHAIN_STEPS = 20

EMPTY = 0
RED = 1
GREEN = 2
BLUE = 3
YELLOW = 4

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

# Existing action order: UP/DOWN for each column, then RIGHT, then LEFT.
ACTION_COLUMNS = jnp.asarray(
    [value for column in range(6) for value in (column, column)]
    + list(range(5))
    + list(range(1, 6)),
    dtype=jnp.int32,
)
ACTION_ORIENTATIONS = jnp.asarray(
    [value for _ in range(6) for value in (UP, DOWN)]
    + [RIGHT] * 5
    + [LEFT] * 5,
    dtype=jnp.int32,
)
SAME_COLOR_LEGAL = jnp.asarray(
    [index % 2 == 0 if index < 12 else 12 <= index < 17 for index in range(22)],
    dtype=jnp.bool_,
)

CHAIN_BONUS = jnp.asarray(
    [0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
    dtype=jnp.int32,
)
COLOR_BONUS = jnp.asarray([0, 0, 3, 6, 12], dtype=jnp.int32)
GROUP_BONUS = jnp.asarray([0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)


def legal_action_mask(axis: jax.Array, child: jax.Array) -> jax.Array:
    """Return the 22-action mask, including same-color deduplication."""
    return jnp.where(axis == child, SAME_COLOR_LEGAL, jnp.ones(ACTION_COUNT, dtype=jnp.bool_))


def _column_heights(board: jax.Array) -> jax.Array:
    occupied = board[:, :STORAGE_HEIGHT] != EMPTY
    rows = jnp.arange(1, STORAGE_HEIGHT + 1, dtype=jnp.int32)
    return jnp.max(jnp.where(occupied, rows[None, :], 0), axis=1)


def _placement(board: jax.Array, axis: jax.Array, child: jax.Array, action_id: jax.Array):
    column = ACTION_COLUMNS[action_id]
    orientation = ACTION_ORIENTATIONS[action_id]
    heights = _column_heights(board)
    horizontal_right = orientation == RIGHT
    horizontal_left = orientation == LEFT
    vertical = ~(horizontal_right | horizontal_left)

    axis_x = column
    child_x = jnp.where(horizontal_right, column + 1, jnp.where(horizontal_left, column - 1, column))
    axis_y = heights[axis_x]
    child_y = heights[child_x]
    axis_y = jnp.where(vertical & (orientation == DOWN), axis_y + 1, axis_y)
    child_y = jnp.where(vertical & (orientation == UP), child_y + 1, child_y)

    def set_cell(current, x, y, color):
        safe_y = jnp.minimum(y, BOARD_HEIGHT - 1)
        old = current[x, safe_y]
        return current.at[x, safe_y].set(jnp.where(y < STORAGE_HEIGHT, color, old))

    placed = set_cell(board, axis_x, axis_y, axis)
    placed = set_cell(placed, child_x, child_y, child)
    topout = ((axis_x == 2) & (axis_y == 11)) | ((child_x == 2) & (child_y == 11))
    return placed, topout


def _component_labels(board: jax.Array) -> tuple[jax.Array, jax.Array]:
    colors = board[:, :VISIBLE_HEIGHT]
    occupied = (colors >= RED) & (colors <= YELLOW)
    initial = jnp.arange(1, BOARD_WIDTH * VISIBLE_HEIGHT + 1, dtype=jnp.int32).reshape(
        BOARD_WIDTH, VISIBLE_HEIGHT
    )
    labels = jnp.where(occupied, initial, 0)
    sentinel = BOARD_WIDTH * VISIBLE_HEIGHT + 1

    def propagate(_, current):
        left_labels = jnp.pad(current[:-1, :], ((1, 0), (0, 0)), constant_values=sentinel)
        right_labels = jnp.pad(current[1:, :], ((0, 1), (0, 0)), constant_values=sentinel)
        down_labels = jnp.pad(current[:, :-1], ((0, 0), (1, 0)), constant_values=sentinel)
        up_labels = jnp.pad(current[:, 1:], ((0, 0), (0, 1)), constant_values=sentinel)
        left_colors = jnp.pad(colors[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        right_colors = jnp.pad(colors[1:, :], ((0, 1), (0, 0)), constant_values=0)
        down_colors = jnp.pad(colors[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        up_colors = jnp.pad(colors[:, 1:], ((0, 0), (0, 1)), constant_values=0)
        candidates = jnp.stack(
            [
                jnp.where(occupied, current, sentinel),
                jnp.where(occupied & (left_colors == colors), left_labels, sentinel),
                jnp.where(occupied & (right_colors == colors), right_labels, sentinel),
                jnp.where(occupied & (down_colors == colors), down_labels, sentinel),
                jnp.where(occupied & (up_colors == colors), up_labels, sentinel),
            ],
            axis=0,
        )
        return jnp.where(occupied, jnp.min(candidates, axis=0), 0)

    # The longest shortest path in a 6x12 component is at most 71 edges.
    labels = jax.lax.fori_loop(0, BOARD_WIDTH * VISIBLE_HEIGHT, propagate, labels)
    counts = jnp.bincount(labels.reshape(-1), length=BOARD_WIDTH * VISIBLE_HEIGHT + 1)
    erase = occupied & (counts[labels] >= 4)
    return labels, erase


def _group_bonus_sum(labels: jax.Array) -> jax.Array:
    counts = jnp.bincount(labels.reshape(-1), length=BOARD_WIDTH * VISIBLE_HEIGHT + 1)
    sizes = counts[1:]
    capped = jnp.minimum(sizes, 10)
    bonuses = jnp.where(sizes >= 11, 10, GROUP_BONUS[capped])
    return jnp.sum(jnp.where(sizes >= 4, bonuses, 0), dtype=jnp.int32)


def _gravity(board: jax.Array) -> jax.Array:
    def compact(column):
        storage = column[:STORAGE_HEIGHT]
        occupied = storage != EMPTY
        sort_key = jnp.where(occupied, jnp.arange(STORAGE_HEIGHT), STORAGE_HEIGHT + jnp.arange(STORAGE_HEIGHT))
        order = jnp.argsort(sort_key)
        values = storage[order]
        count = jnp.sum(occupied, dtype=jnp.int32)
        compacted = jnp.where(jnp.arange(STORAGE_HEIGHT) < count, values, EMPTY)
        return jnp.concatenate([compacted, jnp.zeros((1,), dtype=board.dtype)])

    return jax.vmap(compact)(board)


def _clear_once(board: jax.Array, chain_before: jax.Array, labels: jax.Array, erase_visible: jax.Array):
    has_clear = jnp.any(erase_visible)
    erased_count = jnp.sum(erase_visible, dtype=jnp.int32)
    colors_cleared = jnp.sum(
        jnp.asarray([jnp.any(erase_visible & (board[:, :VISIBLE_HEIGHT] == color)) for color in range(1, 5)]),
        dtype=jnp.int32,
    )
    group_bonus = _group_bonus_sum(jnp.where(erase_visible, labels, 0))
    chain_number = chain_before + 1
    chain_bonus = CHAIN_BONUS[jnp.minimum(chain_number - 1, CHAIN_BONUS.shape[0] - 1)]
    multiplier = jnp.clip(chain_bonus + COLOR_BONUS[colors_cleared] + group_bonus, 1, 999)
    score = jnp.where(has_clear, 10 * erased_count * multiplier, 0)

    erase_full = jnp.pad(erase_visible, ((0, 0), (0, BOARD_HEIGHT - VISIBLE_HEIGHT)))
    adjacent = erase_full
    adjacent = adjacent | jnp.pad(erase_full[:-1, :], ((1, 0), (0, 0)))
    adjacent = adjacent | jnp.pad(erase_full[1:, :], ((0, 1), (0, 0)))
    adjacent = adjacent | jnp.pad(erase_full[:, :-1], ((0, 0), (1, 0)))
    adjacent = adjacent | jnp.pad(erase_full[:, 1:], ((0, 0), (0, 1)))
    clear_mask = erase_full | ((board == 5) & adjacent)
    cleared = jnp.where(clear_mask, EMPTY, board)
    resolved = _gravity(cleared)
    next_board = jnp.where(has_clear, resolved, board)
    return next_board, has_clear, score


def _resolve(board: jax.Array):
    initial_labels, initial_erase = _component_labels(board)

    def condition(carry):
        _, chains, _, _, erase = carry
        return (chains < MAX_CHAIN_STEPS) & jnp.any(erase)

    def step(carry):
        current, chains, total_score, labels, erase = carry
        next_board, _, step_score = _clear_once(current, chains, labels, erase)
        next_labels, next_erase = _component_labels(next_board)
        return next_board, chains + 1, total_score + step_score, next_labels, next_erase

    final_board, chains, total_score, _, remaining_erase = jax.lax.while_loop(
        condition,
        step,
        (board, jnp.int32(0), jnp.int32(0), initial_labels, initial_erase),
    )
    overflow = jnp.any(remaining_erase)
    all_clear = jnp.all(final_board[:, :STORAGE_HEIGHT] == EMPTY)
    return final_board, chains, total_score, all_clear, overflow


def resolve_turn(board: jax.Array, axis: jax.Array, child: jax.Array, action_id: jax.Array):
    """Apply one action and return board/topout/chains/score/all-clear/overflow."""
    placed, topout = _placement(board, axis, child, action_id)

    def topout_result(_):
        return placed, jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.bool_(False)

    def live_result(_):
        return _resolve(placed)

    final_board, chains, score, all_clear, overflow = jax.lax.cond(
        topout, topout_result, live_result, operand=None
    )
    return final_board, topout, chains, score, all_clear, overflow


resolve_turn_batch = jax.jit(jax.vmap(resolve_turn, in_axes=(0, 0, 0, 0)))
