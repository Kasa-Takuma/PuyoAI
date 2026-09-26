"""Fixed-shape batched beam search for solo-search data generation."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .jax_board import ACTION_COUNT, BOARD_HEIGHT, BOARD_WIDTH, legal_action_mask, resolve_turn
from .model import predict
from .jax_structure import (
    MODEL_DEATH_WEIGHT,
    MODEL_VALUE_WEIGHT,
    structure_value_batch,
    turn_prior,
)

DISCOUNT = 0.99


def chain_utility(chains: jax.Array) -> jax.Array:
    return jnp.where(
        chains < 10,
        0.0,
        jnp.where(chains == 10, 1.0, jnp.where(chains == 11, 1.25, jnp.where(chains == 12, 1.5, jnp.where(chains == 13, 1.75, 2.0)))),
    ).astype(jnp.float32)


def _pack_boards(boards: jax.Array) -> jax.Array:
    """Pack each board into seven exact base-5 uint32 words for deduplication."""
    flat = boards.reshape(boards.shape[0], BOARD_WIDTH * BOARD_HEIGHT).astype(jnp.uint32)
    groups = flat.reshape(boards.shape[0], 7, 12)
    powers = jnp.asarray([5**index for index in range(12)], dtype=jnp.uint32)
    return jnp.sum(groups * powers[None, None, :], axis=2, dtype=jnp.uint32)


def _dedupe_one_game(
    boards: jax.Array,
    roots: jax.Array,
    path_keys: jax.Array,
    scores: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    packed = _pack_boards(boards)
    safe_root = jnp.where(valid, roots, ACTION_COUNT + 1)
    safe_packed = jnp.where(valid[:, None], packed, jnp.uint32(0xFFFFFFFF))
    safe_score = jnp.where(valid, scores, -jnp.inf)
    # lexsort uses the final key as the primary key.
    keys = [path_keys, -safe_score]
    keys.extend([safe_packed[:, index] for index in range(6, -1, -1)])
    keys.append(safe_root)
    order = jnp.lexsort(tuple(keys))
    ordered_root = safe_root[order]
    ordered_packed = safe_packed[order]
    ordered_valid = valid[order]
    same_previous = (ordered_root[1:] == ordered_root[:-1]) & jnp.all(
        ordered_packed[1:] == ordered_packed[:-1], axis=1
    )
    keep_ordered = ordered_valid & jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.bool_), ~same_previous]
    )
    return jnp.zeros_like(valid).at[order].set(keep_ordered)


def _select_one_game(
    scores: jax.Array,
    roots: jax.Array,
    path_keys: jax.Array,
    valid: jax.Array,
    beam_width: int,
    preserve_roots: bool,
) -> jax.Array:
    safe_scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)
    if preserve_roots:
        root_values = jnp.arange(ACTION_COUNT, dtype=jnp.int32)[:, None]
        root_masks = valid[None, :] & (roots[None, :] == root_values)
        root_indices = jnp.argmax(
            jnp.where(root_masks, safe_scores[None, :], -jnp.inf), axis=1
        )
        root_exists = jnp.any(root_masks, axis=1)
        root_best = jnp.zeros_like(valid).at[root_indices].max(root_exists)
        priority = root_best.astype(jnp.int32)
    else:
        priority = jnp.zeros_like(roots)
    order = jnp.lexsort((path_keys, -safe_scores, -priority))
    selected = order[:beam_width]
    selected_scores = safe_scores[selected]
    selected_paths = path_keys[selected]
    score_order = jnp.lexsort((selected_paths, -selected_scores))
    return selected[score_order]


def _simple_value(boards: jax.Array) -> jax.Array:
    storage = boards[:, :, :13]
    rows = jnp.arange(1, 14, dtype=jnp.int32)[None, None, :]
    heights = jnp.max(jnp.where(storage != 0, rows, 0), axis=2)
    cells = jnp.sum(heights, axis=1)
    roughness = jnp.sum(jnp.abs(heights[:, 1:] - heights[:, :-1]), axis=1)
    danger = jnp.sum(jnp.maximum(heights - 9, 0), axis=1)
    return cells * 0.002 - roughness * 0.01 - danger * 0.08


def _remaining_pairs(pairs: jax.Array, depth: int, candidates: int):
    games = pairs.shape[0]
    if depth == 1:
        remaining = pairs[:, 1:3, :]
        known = jnp.full((games,), 2, dtype=jnp.int32)
    elif depth == 2:
        remaining = jnp.concatenate(
            [pairs[:, 2:3, :], jnp.zeros((games, 1, 2), dtype=pairs.dtype)], axis=1
        )
        known = jnp.ones((games,), dtype=jnp.int32)
    else:
        remaining = jnp.zeros((games, 2, 2), dtype=pairs.dtype)
        known = jnp.zeros((games,), dtype=jnp.int32)
    return (
        jnp.repeat(remaining, candidates, axis=0),
        jnp.repeat(known, candidates, axis=0),
    )


@partial(
    jax.jit,
    static_argnames=("beam_width", "preserve_roots", "value_at_leaf_only", "dedupe"),
)
def search_batch(
    boards: jax.Array,
    pairs: jax.Array,
    params: dict[str, jax.Array],
    *,
    beam_width: int = 32,
    preserve_roots: bool = True,
    value_at_leaf_only: bool = False,
    dedupe: bool = True,
):
    """Search three supplied pairs for a batch of games.

    Returns best action, scores for all first actions, chosen three-action path,
    and expanded node count per game. ``pairs`` must contain current + NEXT2.
    """
    game_count = boards.shape[0]
    frontier_boards = jnp.repeat(boards[:, None, :, :], beam_width, axis=1)
    frontier_valid = jnp.zeros((game_count, beam_width), dtype=jnp.bool_).at[:, 0].set(True)
    frontier_roots = jnp.full((game_count, beam_width), -1, dtype=jnp.int32)
    frontier_paths = jnp.full((game_count, beam_width, 3), -1, dtype=jnp.int32)
    frontier_path_keys = jnp.zeros((game_count, beam_width), dtype=jnp.int32)
    frontier_rewards = jnp.zeros((game_count, beam_width), dtype=jnp.float32)
    frontier_priors = jnp.zeros((game_count, beam_width), dtype=jnp.float32)
    frontier_scores = jnp.zeros((game_count, beam_width), dtype=jnp.float32)
    expanded_counts = jnp.zeros((game_count,), dtype=jnp.int32)

    for depth_index in range(3):
        candidate_count = beam_width * ACTION_COUNT
        parent_boards = jnp.repeat(frontier_boards, ACTION_COUNT, axis=1)
        parent_valid = jnp.repeat(frontier_valid, ACTION_COUNT, axis=1)
        parent_roots = jnp.repeat(frontier_roots, ACTION_COUNT, axis=1)
        parent_paths = jnp.repeat(frontier_paths, ACTION_COUNT, axis=1)
        parent_path_keys = jnp.repeat(frontier_path_keys, ACTION_COUNT, axis=1)
        parent_rewards = jnp.repeat(frontier_rewards, ACTION_COUNT, axis=1)
        parent_priors = jnp.repeat(frontier_priors, ACTION_COUNT, axis=1)
        action_ids = jnp.tile(jnp.arange(ACTION_COUNT, dtype=jnp.int32), (game_count, beam_width))
        pair = pairs[:, depth_index, :]
        axes = jnp.repeat(pair[:, 0], candidate_count)
        children = jnp.repeat(pair[:, 1], candidate_count)
        flat_actions = action_ids.reshape(-1)
        resolved = jax.vmap(resolve_turn)(
            parent_boards.reshape(-1, BOARD_WIDTH, BOARD_HEIGHT),
            axes,
            children,
            flat_actions,
        )
        candidate_boards = resolved[0].reshape(game_count, candidate_count, BOARD_WIDTH, BOARD_HEIGHT)
        topouts = resolved[1].reshape(game_count, candidate_count)
        chains = resolved[2].reshape(game_count, candidate_count)
        turn_scores = resolved[3].reshape(game_count, candidate_count)
        all_clears = resolved[4].reshape(game_count, candidate_count)
        overflow = resolved[5].reshape(game_count, candidate_count)
        legal = jax.vmap(legal_action_mask)(pair[:, 0], pair[:, 1])
        legal = jnp.tile(legal[:, None, :], (1, beam_width, 1)).reshape(game_count, candidate_count)
        valid = parent_valid & legal & ~overflow
        live_exists = jnp.any(valid & ~topouts, axis=1, keepdims=True)
        valid = valid & (~topouts | ~live_exists)

        roots = jnp.where(depth_index == 0, action_ids, parent_roots)
        paths = parent_paths.at[:, :, depth_index].set(action_ids)
        path_keys = parent_path_keys * ACTION_COUNT + action_ids
        rewards = jnp.where(topouts, -4.0, chain_utility(chains))
        cumulative = parent_rewards + (DISCOUNT**depth_index) * rewards
        placement_prior = turn_prior(chains, turn_scores, topouts, all_clears)
        cumulative_prior = parent_priors + (DISCOUNT**depth_index) * placement_prior

        flat_boards = candidate_boards.reshape(-1, BOARD_WIDTH, BOARD_HEIGHT)
        structural = structure_value_batch(
            flat_boards,
            include_virtual=depth_index == 2,
        ).reshape(game_count, candidate_count)
        if value_at_leaf_only and depth_index < 2:
            future_value = _simple_value(flat_boards)
        else:
            remaining, known = _remaining_pairs(pairs, depth_index + 1, candidate_count)
            prediction = predict(params, flat_boards, remaining, known)
            future_value = (
                prediction[:, 0] * MODEL_VALUE_WEIGHT
                - jax.nn.sigmoid(prediction[:, 1]) * MODEL_DEATH_WEIGHT
            )
        future_value = future_value.reshape(game_count, candidate_count)
        scores = cumulative + cumulative_prior + jnp.where(
            topouts,
            0.0,
            structural + (DISCOUNT ** (depth_index + 1)) * future_value,
        )
        scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)

        if dedupe:
            dedupe_mask = jax.vmap(_dedupe_one_game)(
                candidate_boards, roots, path_keys, scores, valid
            )
            valid = valid & dedupe_mask
        selected = jax.vmap(
            lambda game_scores, game_roots, game_paths, game_valid: _select_one_game(
                game_scores,
                game_roots,
                game_paths,
                game_valid,
                beam_width,
                preserve_roots,
            )
        )(scores, roots, path_keys, valid)
        gather_board = selected[:, :, None, None]
        frontier_boards = jnp.take_along_axis(candidate_boards, gather_board, axis=1)
        frontier_valid = jnp.take_along_axis(valid, selected, axis=1)
        frontier_roots = jnp.take_along_axis(roots, selected, axis=1)
        frontier_paths = jnp.take_along_axis(paths, selected[:, :, None], axis=1)
        frontier_path_keys = jnp.take_along_axis(path_keys, selected, axis=1)
        frontier_rewards = jnp.take_along_axis(cumulative, selected, axis=1)
        frontier_priors = jnp.take_along_axis(cumulative_prior, selected, axis=1)
        frontier_scores = jnp.take_along_axis(scores, selected, axis=1)
        expanded_counts += jnp.sum(parent_valid[:, ::ACTION_COUNT], axis=1) * jnp.sum(
            legal[:, :ACTION_COUNT], axis=1
        )

    root_values = jnp.arange(ACTION_COUNT, dtype=jnp.int32)[None, :, None]
    root_mask = frontier_valid[:, None, :] & (frontier_roots[:, None, :] == root_values)
    root_scores = jnp.max(
        jnp.where(root_mask, frontier_scores[:, None, :], -jnp.inf), axis=2
    )
    best_actions = jnp.argmax(root_scores, axis=1).astype(jnp.int32)
    best_indices = jnp.argmax(
        jnp.where(
            frontier_valid & (frontier_roots == best_actions[:, None]),
            frontier_scores,
            -jnp.inf,
        ),
        axis=1,
    )
    best_paths = jnp.take_along_axis(frontier_paths, best_indices[:, None, None], axis=1)[:, 0, :]
    return best_actions, root_scores, best_paths, expanded_counts
