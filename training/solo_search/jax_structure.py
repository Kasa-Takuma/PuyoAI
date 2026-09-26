"""JAX scoring shared with the browser solo-search candidate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_board import BOARD_HEIGHT, BOARD_WIDTH, STORAGE_HEIGHT, VISIBLE_HEIGHT, _component_labels, _column_heights, resolve_turn

STRUCTURE_SCALE = 1000.0
MODEL_VALUE_WEIGHT = 30.0
MODEL_DEATH_WEIGHT = 30.0

PROBE_PAIRS = jnp.asarray(
    [(axis, child) for axis in range(1, 5) for child in range(1, 5)],
    dtype=jnp.uint8,
)
PROBE_ACTIONS = jnp.asarray([0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16], dtype=jnp.int32)


def turn_prior(
    chains: jax.Array,
    scores: jax.Array,
    topouts: jax.Array,
    all_clears: jax.Array,
) -> jax.Array:
    """Score actual placement results, suppressing premature small fires."""
    chains_f = chains.astype(jnp.float32)
    chain_value = 825.0 * chains_f**3.10028
    small_penalty = jnp.where(
        (chains >= 2) & (chains <= 6), -71_545.0 * (7 - chains_f), 0.0
    )
    mid_penalty = jnp.where((chains >= 7) & (chains <= 9), -150_121.0, 0.0)
    value = (
        chain_value
        + scores.astype(jnp.float32) * 0.90442
        + small_penalty
        + mid_penalty
        + jnp.where(chains == 7, -46_388.0, 0.0)
        + jnp.where(chains == 8, -69_074.0, 0.0)
        + jnp.where(chains == 9, -64_412.0, 0.0)
        + jnp.where(chains >= 10, 95_631.0, 0.0)
        + jnp.where(chains >= 11, 203_425.0, 0.0)
        + jnp.where(chains >= 12, 478_153.0, 0.0)
    )
    value = jnp.where(
        chains == 1,
        -23_000.0 + scores.astype(jnp.float32) * 0.03,
        value,
    )
    value = jnp.where(chains == 0, jnp.where(all_clears, 180.0, 0.0), value)
    return jnp.where(topouts, -5_000_000.0, value) / STRUCTURE_SCALE


def _base_features(board: jax.Array) -> dict[str, jax.Array]:
    heights = _column_heights(board)
    labels, _ = _component_labels(board)
    counts = jnp.bincount(labels.reshape(-1), length=BOARD_WIDTH * VISIBLE_HEIGHT + 1)
    sizes = counts[1:]
    groups_present = sizes > 0
    stack_cells = jnp.sum(board[:, :STORAGE_HEIGHT] != 0, dtype=jnp.int32)
    hidden_cells = jnp.sum(jnp.maximum(heights - VISIBLE_HEIGHT, 0))
    danger_cells = jnp.sum(jnp.maximum(heights - (VISIBLE_HEIGHT - 2), 0))
    differences = jnp.abs(heights[1:] - heights[:-1])

    rows = jnp.arange(VISIBLE_HEIGHT, dtype=jnp.int32)[None, :]
    surface = (rows == heights[:, None]) & (board[:, :VISIBLE_HEIGHT] == 0)
    neighbor_labels = jnp.stack(
        [
            jnp.pad(labels[:-1], ((1, 0), (0, 0))),
            jnp.pad(labels[1:], ((0, 1), (0, 0))),
            jnp.pad(labels[:, :-1], ((0, 0), (1, 0))),
            jnp.pad(labels[:, 1:], ((0, 0), (0, 1))),
        ]
    )
    surface_groups = jnp.any(
        jax.nn.one_hot(
            neighbor_labels,
            BOARD_WIDTH * VISIBLE_HEIGHT + 1,
            dtype=jnp.bool_,
        )
        & surface[None, :, :, None],
        axis=(0, 1, 2),
    )[1:]

    color_counts = jnp.asarray(
        [jnp.sum(board[:, :VISIBLE_HEIGHT] == color) for color in range(1, 5)],
        dtype=jnp.int32,
    )
    colors_present = color_counts > 0
    minimum_color = jnp.min(jnp.where(colors_present, color_counts, stack_cells + 1))
    color_balance = jnp.where(
        (jnp.sum(colors_present) <= 1) | (stack_cells == 0),
        0.0,
        1.0
        - (jnp.max(color_counts) - minimum_color).astype(jnp.float32)
        / stack_cells.astype(jnp.float32),
    )
    left = heights[:-2]
    middle = heights[1:-1]
    right = heights[2:]
    valley_penalty = jnp.sum(jnp.maximum(jnp.minimum(left, right) - middle - 1, 0))

    return {
        "stackCells": stack_cells,
        "maxHeight": jnp.max(heights),
        "hiddenCells": hidden_cells,
        "dangerCells": danger_cells,
        "surfaceRoughness": jnp.sum(differences),
        "staircaseLinks": jnp.sum(jnp.where(differences == 1, 2, jnp.where(differences == 2, 1, 0))),
        "steepWalls": jnp.sum(jnp.maximum(differences - 2, 0)),
        "valleyPenalty": valley_penalty,
        "adjacency": jnp.sum(jnp.where(groups_present, sizes - 1, 0)),
        "group2Count": jnp.sum(sizes == 2),
        "group3Count": jnp.sum(sizes == 3),
        "surfaceExtendableGroup2Count": jnp.sum((sizes == 2) & surface_groups),
        "surfaceReadyGroup3Count": jnp.sum((sizes == 3) & surface_groups),
        "isolatedSingles": jnp.sum(sizes == 1),
        "colorBalance": color_balance,
        "columnsUsed": jnp.sum(heights > 0),
    }


def _virtual_features(board: jax.Array) -> dict[str, jax.Array]:
    pair_count = PROBE_PAIRS.shape[0]
    action_count = PROBE_ACTIONS.shape[0]
    axes = jnp.repeat(PROBE_PAIRS[:, 0], action_count)
    children = jnp.repeat(PROBE_PAIRS[:, 1], action_count)
    actions = jnp.tile(PROBE_ACTIONS, pair_count)
    boards = jnp.repeat(board[None, :, :], pair_count * action_count, axis=0)
    resolved = jax.vmap(resolve_turn)(boards, axes, children, actions)
    chains = resolved[2]
    scores = resolved[3]
    valid = ~resolved[1] & ~resolved[5] & (chains > 0)
    ranked_chains = jnp.where(valid, chains, -1)
    ranked_scores = jnp.where(valid, scores, -1)
    order = jnp.lexsort((-ranked_scores, -ranked_chains))
    top_chains = jnp.maximum(ranked_chains[order[:3]], 0)
    top_scores = jnp.maximum(ranked_scores[order[:3]], 0)
    return {
        "bestVirtualChain": top_chains[0],
        "bestVirtualScore": top_scores[0],
        "virtualChainCount2Plus": jnp.sum(valid & (chains >= 2)),
        "virtualChainCount3Plus": jnp.sum(valid & (chains >= 3)),
        "topVirtualChainSum": jnp.sum(top_chains),
        "topVirtualScoreSum": jnp.sum(top_scores),
    }


def _zero_virtual_features() -> dict[str, jax.Array]:
    zero = jnp.int32(0)
    return {
        "bestVirtualChain": zero,
        "bestVirtualScore": zero,
        "virtualChainCount2Plus": zero,
        "virtualChainCount3Plus": zero,
        "topVirtualChainSum": zero,
        "topVirtualScoreSum": zero,
    }


def structure_value_one(board: jax.Array, *, include_virtual: bool) -> jax.Array:
    features = _base_features(board)
    if include_virtual:
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
            lambda _: _zero_virtual_features(),
            board,
        )
    else:
        virtual = _zero_virtual_features()
    features = {**features, **virtual}
    best = features["bestVirtualChain"].astype(jnp.float32)
    top_sum = features["topVirtualChainSum"].astype(jnp.float32)
    top_score = features["topVirtualScoreSum"].astype(jnp.float32)
    count2 = jnp.minimum(features["virtualChainCount2Plus"], 6).astype(jnp.float32)
    count3 = jnp.minimum(features["virtualChainCount3Plus"], 3).astype(jnp.float32)
    base = (
        best**3 * 1051.0
        + top_sum * 376.0
        + count2 * 58.0
        + count3 * 215.0
        + features["bestVirtualScore"].astype(jnp.float32) * 0.48661
        + top_score * 0.13754
        + features["surfaceReadyGroup3Count"] * 209.0
        + features["surfaceExtendableGroup2Count"] * 64.0
        + features["group3Count"] * 62.0
        + features["group2Count"] * 18.0
        + features["adjacency"] * 12.0
        + features["staircaseLinks"] * 20.0
        + features["colorBalance"] * 140.0
        + features["stackCells"] * 12.0
        + features["columnsUsed"] * 14.0
        - features["hiddenCells"] * 5000.0
        - features["dangerCells"] * 241.0
        - features["surfaceRoughness"] * 15.0
        - features["steepWalls"] * 67.0
        - features["valleyPenalty"] * 41.0
        - features["isolatedSingles"] * 39.0
    )
    large = (
        jnp.maximum(0.0, best - 5.0) ** 3 * 460.0
        + jnp.maximum(0.0, top_sum - 15.0) * 2400.0
        + jnp.where(best >= 10, 90_000.0, 0.0)
        - jnp.maximum(0.0, features["maxHeight"] - 9.0)
        * jnp.maximum(0.0, 6.0 - best)
        * 1400.0
    )
    mature = (
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
            jnp.maximum(0.0, 10.0 - best)
            * jnp.maximum(0.0, features["stackCells"] - 36.0)
            * 4200.0,
            0.0,
        )
        - jnp.where(
            best == 10,
            jnp.maximum(0.0, 28.0 - top_sum) * 14_000.0
            + jnp.maximum(0.0, 135_000.0 - top_score) * 0.08,
            0.0,
        )
        - jnp.maximum(0.0, features["stackCells"] - 51.0)
        * jnp.maximum(0.0, 11.0 - best)
        * 13_500.0
        - jnp.maximum(0.0, features["maxHeight"] - 11.0)
        * jnp.maximum(0.0, 10.0 - best)
        * 6200.0
        - jnp.maximum(0.0, features["dangerCells"] - 3.0)
        * jnp.maximum(0.0, 11.0 - best)
        * 3700.0
        - jnp.maximum(0.0, features["surfaceRoughness"] - 16.0) * 1600.0
        - jnp.maximum(0.0, features["steepWalls"] - 9.0) * 2400.0
        - jnp.maximum(0.0, features["hiddenCells"]) * 14_000.0
    )
    return (base + large + mature) / STRUCTURE_SCALE


def structure_value_batch(boards: jax.Array, *, include_virtual: bool) -> jax.Array:
    return jax.vmap(lambda board: structure_value_one(board, include_virtual=include_virtual))(boards)
