"""Batched JAX search for evolving chain-builder profile weights."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .jax_board import ACTION_COUNT, BOARD_HEIGHT, BOARD_WIDTH, legal_action_mask, resolve_turn
from .jax_search import _pack_boards, _select_one_game
from .jax_structure import _base_features, _virtual_features, _zero_virtual_features

PROFILE_KEYS = (
    "chainValueBase",
    "chainExponent",
    "scoreScale",
    "smallChainPenaltyStep",
    "midChainPenalty",
    "sevenChainPenalty",
    "eightChainPenalty",
    "nineChainPenalty",
    "tenPlusBonus",
    "elevenPlusBonus",
    "twelvePlusBonus",
    "allClearBonus",
    "bestVirtualChain",
    "topVirtualChainSum",
    "bestVirtualScore",
    "topVirtualScoreSum",
    "virtualChainCount3Plus",
    "surfaceReadyGroup3Count",
    "surfaceExtendableGroup2Count",
    "dangerCells",
    "surfaceRoughness",
    "steepWalls",
    "largeChainScale",
    "v9bScale",
)
PROFILE_INDEX = {key: index for index, key in enumerate(PROFILE_KEYS)}
SCORE_SCALE = 1000.0


def _weight(weights: jax.Array, key: str) -> jax.Array:
    return weights[:, PROFILE_INDEX[key]]


def score_turn_batch(
    chains: jax.Array,
    scores: jax.Array,
    topouts: jax.Array,
    all_clears: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Match the configurable portion of JS scoreTurnResult for v12AC."""
    dtype = weights.dtype
    chains_f = chains.astype(dtype)
    score_f = scores.astype(dtype)
    value = (
        _weight(weights, "chainValueBase") * chains_f ** _weight(weights, "chainExponent")
        + score_f * _weight(weights, "scoreScale")
        + jnp.where(
            (chains >= 2) & (chains <= 6),
            _weight(weights, "smallChainPenaltyStep") * (7.0 - chains_f),
            0.0,
        )
        + jnp.where((chains >= 7) & (chains <= 9), _weight(weights, "midChainPenalty"), 0.0)
        + jnp.where(chains == 7, _weight(weights, "sevenChainPenalty"), 0.0)
        + jnp.where(chains == 8, _weight(weights, "eightChainPenalty"), 0.0)
        + jnp.where(chains == 9, _weight(weights, "nineChainPenalty"), 0.0)
        + jnp.where(chains >= 10, _weight(weights, "tenPlusBonus"), 0.0)
        + jnp.where(chains >= 11, _weight(weights, "elevenPlusBonus"), 0.0)
        + jnp.where(chains >= 12, _weight(weights, "twelvePlusBonus"), 0.0)
        + jnp.where(all_clears, _weight(weights, "allClearBonus"), 0.0)
    )
    value = jnp.where(chains == 1, -23_000.0 + score_f * 0.03 + jnp.where(all_clears, _weight(weights, "allClearBonus"), 0.0), value)
    value = jnp.where(chains == 0, jnp.where(all_clears, _weight(weights, "allClearBonus"), 0.0), value)
    return jnp.where(topouts, -5_000_000.0, value) / SCORE_SCALE


def _score_features(features: dict[str, jax.Array], weights: jax.Array) -> jax.Array:
    dtype = weights.dtype
    best = features["bestVirtualChain"].astype(dtype)
    top_sum = features["topVirtualChainSum"].astype(dtype)
    top_score = features["topVirtualScoreSum"].astype(dtype)
    count2 = jnp.minimum(features["virtualChainCount2Plus"], 6).astype(dtype)
    count3 = jnp.minimum(features["virtualChainCount3Plus"], 3).astype(dtype)
    base = (
        best**3 * _weight(weights, "bestVirtualChain")
        + top_sum * _weight(weights, "topVirtualChainSum")
        + count2 * 58.0
        + count3 * _weight(weights, "virtualChainCount3Plus")
        + features["bestVirtualScore"].astype(dtype) * _weight(weights, "bestVirtualScore")
        + top_score * _weight(weights, "topVirtualScoreSum")
        + features["surfaceReadyGroup3Count"] * _weight(weights, "surfaceReadyGroup3Count")
        + features["surfaceExtendableGroup2Count"] * _weight(weights, "surfaceExtendableGroup2Count")
        + features["group3Count"] * 62.0
        + features["group2Count"] * 18.0
        + features["adjacency"] * 12.0
        + features["staircaseLinks"] * 20.0
        + features["colorBalance"] * 140.0
        + features["stackCells"] * 12.0
        + features["columnsUsed"] * 14.0
        - features["hiddenCells"] * 5000.0
        + features["dangerCells"] * _weight(weights, "dangerCells")
        + features["surfaceRoughness"] * _weight(weights, "surfaceRoughness")
        + features["steepWalls"] * _weight(weights, "steepWalls")
        - features["valleyPenalty"] * 41.0
        - features["isolatedSingles"] * 39.0
    )
    large = _weight(weights, "largeChainScale") * (
        jnp.maximum(0.0, best - 5.0) ** 3 * 460.0
        + jnp.maximum(0.0, top_sum - 15.0) * 2400.0
        + jnp.where(best >= 10, 90_000.0, 0.0)
        - jnp.maximum(0.0, features["maxHeight"] - 9.0)
        * jnp.maximum(0.0, 6.0 - best)
        * 1400.0
    )
    mature = _weight(weights, "v9bScale") * (
        jnp.maximum(0.0, best - 8.0) ** 3 * 1450.0
        + jnp.maximum(0.0, best - 10.0) ** 3 * 5800.0
        + jnp.maximum(0.0, top_sum - 25.0) * 2500.0
        + jnp.maximum(0.0, top_sum - 29.0) * 8800.0
        + jnp.maximum(0.0, top_score - 115_000.0) * 0.2
        + jnp.maximum(0.0, top_score - 160_000.0) * 0.34
        + jnp.minimum(features["virtualChainCount3Plus"], 10) * jnp.where(best >= 10, 2200.0, 0.0)
        + jnp.where(best >= 11, 460_000.0, 0.0)
        + jnp.where(best >= 12, 400_000.0, 0.0)
        + jnp.where(
            (best >= 11) & (features["stackCells"] >= 52),
            jnp.minimum(features["stackCells"] - 51, 10) * 4200.0,
            0.0,
        )
        - jnp.where(
            (best >= 7) & (best <= 9),
            jnp.maximum(0.0, 10.0 - best) * jnp.maximum(0.0, features["stackCells"] - 36.0) * 4200.0,
            0.0,
        )
        - jnp.where(
            best == 10,
            jnp.maximum(0.0, 28.0 - top_sum) * 14_000.0
            + jnp.maximum(0.0, 135_000.0 - top_score) * 0.08,
            0.0,
        )
        - jnp.maximum(0.0, features["stackCells"] - 51.0) * jnp.maximum(0.0, 11.0 - best) * 13_500.0
        - jnp.maximum(0.0, features["maxHeight"] - 11.0) * jnp.maximum(0.0, 10.0 - best) * 6200.0
        - jnp.maximum(0.0, features["dangerCells"] - 3.0) * jnp.maximum(0.0, 11.0 - best) * 3700.0
        - jnp.maximum(0.0, features["surfaceRoughness"] - 16.0) * 1600.0
        - jnp.maximum(0.0, features["steepWalls"] - 9.0) * 2400.0
        - jnp.maximum(0.0, features["hiddenCells"]) * 14_000.0
    )
    return (base + large + mature) / SCORE_SCALE


def score_boards_batch(boards: jax.Array, weights: jax.Array, *, include_virtual: bool) -> jax.Array:
    def score_one(board, one_weights):
        features = _base_features(board)
        if include_virtual:
            count_dtype = jnp.sum(jnp.zeros((1,), dtype=jnp.bool_)).dtype
            should_probe = (
                (features["stackCells"] >= 6)
                & (
                    (features["group3Count"] > 0)
                    | (features["surfaceExtendableGroup2Count"] >= 2)
                    | (features["maxHeight"] >= 4)
                )
            )
            virtual = jax.lax.cond(
                should_probe,
                _virtual_features,
                lambda _: {
                    key: value.astype(
                        jnp.int32
                        if key in ("bestVirtualChain", "bestVirtualScore")
                        else count_dtype
                    )
                    for key, value in _zero_virtual_features().items()
                },
                board,
            )
        else:
            virtual = _zero_virtual_features()
        batched = {key: value[None] for key, value in {**features, **virtual}.items()}
        return _score_features(batched, one_weights[None])[0]

    return jax.vmap(score_one)(boards, weights)


def _dedupe_with_js_order(
    boards: jax.Array,
    roots: jax.Array,
    scores: jax.Array,
    valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Keep the best duplicate while preserving JS Map insertion order."""
    count = valid.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    packed = _pack_boards(boards)
    safe_roots = jnp.where(valid, roots, ACTION_COUNT + 1)
    safe_packed = jnp.where(valid[:, None], packed, jnp.uint32(0xFFFFFFFF))
    safe_scores = jnp.where(valid, scores, -jnp.inf)
    keys = [indices, -safe_scores]
    keys.extend([safe_packed[:, index] for index in range(6, -1, -1)])
    keys.append(safe_roots)
    order = jnp.lexsort(tuple(keys)).astype(jnp.int32)
    ordered_roots = safe_roots[order]
    ordered_packed = safe_packed[order]
    ordered_valid = valid[order]
    starts = jnp.concatenate(
        [
            jnp.ones((1,), dtype=jnp.bool_),
            (ordered_roots[1:] != ordered_roots[:-1])
            | jnp.any(ordered_packed[1:] != ordered_packed[:-1], axis=1),
        ]
    )
    groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
    first_indices = jax.ops.segment_min(
        order,
        groups,
        num_segments=count,
        indices_are_sorted=True,
    )
    group_first = first_indices[groups]
    keep_ordered = ordered_valid & starts
    keep = jnp.zeros_like(valid).at[order].set(keep_ordered)
    board_first = jnp.full((count,), count, dtype=jnp.int32).at[order].set(group_first)
    root_first = jnp.full((ACTION_COUNT + 2,), count, dtype=jnp.int32).at[
        safe_roots
    ].min(jnp.where(valid, indices, count))
    tie_order = root_first[jnp.minimum(safe_roots, ACTION_COUNT + 1)] * count + board_first
    return keep, tie_order


def _update_pool_one_root(
    old_boards: jax.Array,
    old_paths: jax.Array,
    old_turn: jax.Array,
    old_scores: jax.Array,
    old_valid: jax.Array,
    candidate_boards: jax.Array,
    candidate_paths: jax.Array,
    candidate_turn: jax.Array,
    candidate_scores: jax.Array,
    candidate_roots: jax.Array,
    candidate_ties: jax.Array,
    candidate_valid: jax.Array,
    root: jax.Array,
):
    matches = candidate_valid & (candidate_roots == root)
    combined_scores = jnp.concatenate(
        [old_scores, jnp.where(matches, candidate_scores, -jnp.inf)]
    )
    combined_valid = jnp.concatenate([old_valid, matches])
    combined_ties = jnp.concatenate(
        [jnp.arange(3, dtype=jnp.int32), candidate_ties + 3]
    )
    combined_boards = jnp.concatenate([old_boards, candidate_boards], axis=0)
    combined_paths = jnp.concatenate([old_paths, candidate_paths], axis=0)
    combined_turn = jnp.concatenate([old_turn, candidate_turn], axis=0)
    order = jnp.lexsort((combined_ties, -combined_scores))[:3]
    return (
        combined_boards[order],
        combined_paths[order],
        combined_turn[order],
        combined_scores[order],
        combined_valid[order],
    )


@partial(jax.jit, static_argnames=("beam_width", "dedupe"))
def search_profile_batch(
    boards: jax.Array,
    pairs: jax.Array,
    weights: jax.Array,
    *,
    beam_width: int = 24,
    dedupe: bool = True,
):
    """Search current + NEXT2 for games that may each use different weights."""
    games = boards.shape[0]
    frontier_boards = jnp.repeat(boards[:, None], beam_width, axis=1)
    frontier_valid = jnp.zeros((games, beam_width), dtype=jnp.bool_).at[:, 0].set(True)
    frontier_roots = jnp.full((games, beam_width), -1, dtype=jnp.int32)
    frontier_paths = jnp.full((games, beam_width, 3), -1, dtype=jnp.int32)
    frontier_keys = jnp.zeros((games, beam_width), dtype=jnp.int32)
    frontier_turn = jnp.zeros((games, beam_width), dtype=weights.dtype)
    frontier_scores = jnp.zeros((games, beam_width), dtype=weights.dtype)
    pool_boards = jnp.zeros(
        (games, ACTION_COUNT, 3, BOARD_WIDTH, BOARD_HEIGHT), dtype=boards.dtype
    )
    pool_paths = jnp.full((games, ACTION_COUNT, 3, 3), -1, dtype=jnp.int32)
    pool_turn = jnp.zeros((games, ACTION_COUNT, 3), dtype=weights.dtype)
    pool_scores = jnp.full((games, ACTION_COUNT, 3), -jnp.inf, dtype=weights.dtype)
    pool_valid = jnp.zeros((games, ACTION_COUNT, 3), dtype=jnp.bool_)

    for depth_index in range(3):
        candidate_count = beam_width * ACTION_COUNT
        parent_boards = jnp.repeat(frontier_boards, ACTION_COUNT, axis=1)
        parent_valid = jnp.repeat(frontier_valid, ACTION_COUNT, axis=1)
        parent_roots = jnp.repeat(frontier_roots, ACTION_COUNT, axis=1)
        parent_paths = jnp.repeat(frontier_paths, ACTION_COUNT, axis=1)
        parent_keys = jnp.repeat(frontier_keys, ACTION_COUNT, axis=1)
        parent_turn = jnp.repeat(frontier_turn, ACTION_COUNT, axis=1)
        action_ids = jnp.tile(jnp.arange(ACTION_COUNT, dtype=jnp.int32), (games, beam_width))
        pair = pairs[:, depth_index]
        resolved = jax.vmap(resolve_turn)(
            parent_boards.reshape(-1, BOARD_WIDTH, BOARD_HEIGHT),
            jnp.repeat(pair[:, 0], candidate_count),
            jnp.repeat(pair[:, 1], candidate_count),
            action_ids.reshape(-1),
        )
        candidate_boards = resolved[0].reshape(games, candidate_count, BOARD_WIDTH, BOARD_HEIGHT)
        topouts = resolved[1].reshape(games, candidate_count)
        chains = resolved[2].reshape(games, candidate_count)
        turn_scores = resolved[3].reshape(games, candidate_count)
        all_clears = resolved[4].reshape(games, candidate_count)
        overflow = resolved[5].reshape(games, candidate_count)
        legal = jax.vmap(legal_action_mask)(pair[:, 0], pair[:, 1])
        legal = jnp.tile(legal[:, None], (1, beam_width, 1)).reshape(games, candidate_count)
        valid = parent_valid & legal & ~overflow
        live_exists = jnp.any(valid & ~topouts, axis=1, keepdims=True)
        valid &= ~topouts | ~live_exists
        roots = jnp.where(depth_index == 0, action_ids, parent_roots)
        paths = parent_paths.at[:, :, depth_index].set(action_ids)
        path_keys = parent_keys * ACTION_COUNT + action_ids
        expanded_weights = jnp.repeat(weights, candidate_count, axis=0)
        cumulative_turn = parent_turn + score_turn_batch(
            chains.reshape(-1),
            turn_scores.reshape(-1),
            topouts.reshape(-1),
            all_clears.reshape(-1),
            expanded_weights,
        ).reshape(games, candidate_count)
        board_score = score_boards_batch(
            candidate_boards.reshape(-1, BOARD_WIDTH, BOARD_HEIGHT),
            expanded_weights,
            include_virtual=False,
        ).reshape(games, candidate_count)
        scores = jnp.where(valid, cumulative_turn + board_score, -jnp.inf)
        if dedupe:
            keep, tie_order = jax.vmap(_dedupe_with_js_order)(
                candidate_boards, roots, scores, valid
            )
            valid &= keep
        else:
            tie_order = jnp.tile(
                jnp.arange(candidate_count, dtype=jnp.int32), (games, 1)
            )
        def update_game(old_boards, old_paths, old_turn, old_scores, old_valid, new_boards, new_paths, new_turn, new_scores, new_roots, new_ties, new_valid):
            return jax.vmap(
                lambda root, root_boards, root_paths, root_turn, root_scores, root_valid: _update_pool_one_root(
                    root_boards,
                    root_paths,
                    root_turn,
                    root_scores,
                    root_valid,
                    new_boards,
                    new_paths,
                    new_turn,
                    new_scores,
                    new_roots,
                    new_ties,
                    new_valid,
                    root,
                )
            )(
                jnp.arange(ACTION_COUNT, dtype=jnp.int32),
                old_boards,
                old_paths,
                old_turn,
                old_scores,
                old_valid,
            )

        pool_boards, pool_paths, pool_turn, pool_scores, pool_valid = jax.vmap(
            update_game
        )(
            pool_boards,
            pool_paths,
            pool_turn,
            pool_scores,
            pool_valid,
            candidate_boards,
            paths,
            cumulative_turn,
            scores,
            roots,
            tie_order,
            valid,
        )
        selected = jax.vmap(
            lambda one_scores, one_roots, one_ties, one_valid: _select_one_game(
                one_scores, one_roots, one_ties, one_valid, beam_width, False
            )
        )(scores, roots, tie_order, valid)
        frontier_boards = jnp.take_along_axis(candidate_boards, selected[:, :, None, None], axis=1)
        frontier_valid = jnp.take_along_axis(valid, selected, axis=1)
        frontier_roots = jnp.take_along_axis(roots, selected, axis=1)
        frontier_paths = jnp.take_along_axis(paths, selected[:, :, None], axis=1)
        frontier_keys = jnp.take_along_axis(path_keys, selected, axis=1)
        frontier_turn = jnp.take_along_axis(cumulative_turn, selected, axis=1)
        frontier_scores = jnp.take_along_axis(scores, selected, axis=1)

    pool_size = ACTION_COUNT * 3
    pool_weights = jnp.repeat(weights, pool_size, axis=0)
    refined = score_boards_batch(
        pool_boards.reshape(-1, BOARD_WIDTH, BOARD_HEIGHT),
        pool_weights,
        include_virtual=True,
    ).reshape(games, ACTION_COUNT, 3)
    candidate_scores = jnp.where(pool_valid, pool_turn + refined, -jnp.inf)
    root_scores = jnp.max(candidate_scores, axis=2)
    root_best_indices = jnp.argmax(candidate_scores, axis=2)
    root_paths = jnp.take_along_axis(
        pool_paths, root_best_indices[:, :, None, None], axis=2
    )[:, :, 0, :]
    best_actions = jnp.argmax(root_scores, axis=1).astype(jnp.int32)
    best_paths = jnp.take_along_axis(
        root_paths, best_actions[:, None, None], axis=1
    )[:, 0, :]
    return best_actions, root_scores, best_paths


@partial(jax.jit, static_argnames=("beam_width",))
def play_profile_batch(
    boards: jax.Array,
    pair_stream: jax.Array,
    weights: jax.Array,
    *,
    beam_width: int = 24,
):
    """Play complete fixed-length games without returning to Python each turn."""
    active = jnp.ones((boards.shape[0],), dtype=jnp.bool_)

    def step(carry, turn_pairs):
        current_boards, current_active = carry
        actions, _, _ = search_profile_batch(
            current_boards,
            turn_pairs,
            weights,
            beam_width=beam_width,
        )
        resolved = jax.vmap(resolve_turn)(
            current_boards,
            turn_pairs[:, 0, 0],
            turn_pairs[:, 0, 1],
            actions,
        )
        next_boards = jnp.where(
            current_active[:, None, None], resolved[0], current_boards
        )
        topouts = current_active & resolved[1]
        chains = jnp.where(current_active, resolved[2], -1)
        scores = jnp.where(current_active, resolved[3], 0)
        all_clears = current_active & resolved[4]
        return (next_boards, current_active & ~resolved[1]), (
            chains,
            scores,
            topouts,
            all_clears,
        )

    windows = jax.vmap(
        lambda stream: jax.vmap(lambda index: jax.lax.dynamic_slice(stream, (index, 0), (3, 2)))(
            jnp.arange(stream.shape[0] - 2)
        )
    )(pair_stream)
    _, traces = jax.lax.scan(step, (boards, active), jnp.swapaxes(windows, 0, 1))
    return traces


@partial(jax.pmap, static_broadcasted_argnums=(3,))
def play_profile_batch_sharded(
    boards: jax.Array,
    pair_stream: jax.Array,
    weights: jax.Array,
    beam_width: int,
):
    """Run one independent profile batch on every visible accelerator."""
    return play_profile_batch(
        boards,
        pair_stream,
        weights,
        beam_width=beam_width,
    )
