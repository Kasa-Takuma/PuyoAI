"""Streaming value-function sample encoding for PuyoAI."""

from __future__ import annotations

from dataclasses import dataclass

from training.data import (
    BOARD_CELLS,
    BOARD_HEIGHT,
    BOARD_SYMBOL_TO_INDEX,
    BOARD_SYMBOLS,
    BOARD_WIDTH,
    MAX_NEXT_PAIRS,
    PLAYABLE_COLOR_TO_INDEX,
    PLAYABLE_COLORS,
)

VALUE_TARGET_NAMES = [
    "objective",
    "max_chain",
    "chains10_plus",
    "chains11_plus",
    "chains12_plus",
    "topout",
]

VALUE_FEATURE_KEYS = [
    "stackCells",
    "maxHeight",
    "hiddenCells",
    "dangerCells",
    "surfaceRoughness",
    "steepWalls",
    "valleyPenalty",
    "adjacency",
    "group2Count",
    "group3Count",
    "surfaceExtendableGroup2Count",
    "surfaceReadyGroup3Count",
    "isolatedSingles",
    "colorBalance",
    "columnsUsed",
    "bestVirtualChain",
    "bestVirtualScore",
    "virtualChainCount2Plus",
    "virtualChainCount3Plus",
    "topVirtualChainSum",
    "topVirtualScoreSum",
]

VALUE_FEATURE_SCALES = {
    "stackCells": 78.0,
    "maxHeight": 13.0,
    "hiddenCells": 6.0,
    "dangerCells": 18.0,
    "surfaceRoughness": 48.0,
    "steepWalls": 24.0,
    "valleyPenalty": 24.0,
    "adjacency": 78.0,
    "group2Count": 18.0,
    "group3Count": 18.0,
    "surfaceExtendableGroup2Count": 18.0,
    "surfaceReadyGroup3Count": 18.0,
    "isolatedSingles": 40.0,
    "colorBalance": 1.0,
    "columnsUsed": 6.0,
    "bestVirtualChain": 14.0,
    "bestVirtualScore": 300_000.0,
    "virtualChainCount2Plus": 48.0,
    "virtualChainCount3Plus": 24.0,
    "topVirtualChainSum": 36.0,
    "topVirtualScoreSum": 600_000.0,
}


@dataclass(slots=True)
class EncodedValueSample:
    """One encoded state and its multi-target future value labels."""

    inputs: list[float]
    targets: list[float]

    @property
    def input_dim(self) -> int:
        return len(self.inputs)


def _encode_board(board_rows: list[str]) -> list[float]:
    rows = list(board_rows)
    if len(rows) != BOARD_HEIGHT:
        raise ValueError(f"Expected {BOARD_HEIGHT} board rows, got {len(rows)}")

    encoded = [0.0] * (BOARD_CELLS * len(BOARD_SYMBOLS))
    offset = 0
    for row in rows:
        if len(row) != BOARD_WIDTH:
            raise ValueError(f"Expected row width {BOARD_WIDTH}, got {len(row)}")
        for symbol in row:
            encoded[offset + BOARD_SYMBOL_TO_INDEX[symbol]] = 1.0
            offset += len(BOARD_SYMBOLS)
    return encoded


def _encode_pair(pair: dict[str, str]) -> list[float]:
    vector = [0.0] * (len(PLAYABLE_COLORS) * 2)
    vector[PLAYABLE_COLOR_TO_INDEX[pair["axis"]]] = 1.0
    vector[len(PLAYABLE_COLORS) + PLAYABLE_COLOR_TO_INDEX[pair["child"]]] = 1.0
    return vector


def _encode_next_queue(next_queue: list[dict[str, str]], max_next_pairs: int) -> list[float]:
    vector = [0.0] * (max_next_pairs * len(PLAYABLE_COLORS) * 2)
    for index, pair in enumerate(next_queue[:max_next_pairs]):
        base = index * len(PLAYABLE_COLORS) * 2
        vector[base + PLAYABLE_COLOR_TO_INDEX[pair["axis"]]] = 1.0
        vector[base + len(PLAYABLE_COLORS) + PLAYABLE_COLOR_TO_INDEX[pair["child"]]] = 1.0
    return vector


def _scaled_feature(features: dict, key: str) -> float:
    value = float(features.get(key, 0.0) or 0.0)
    scale = VALUE_FEATURE_SCALES[key]
    return max(-4.0, min(4.0, value / scale))


def _encode_state(sample: dict, max_next_pairs: int) -> list[float]:
    state = sample["state"]
    features = sample.get("features", {})
    vector: list[float] = []
    vector.extend(_encode_board(list(state["boardRows"])))
    vector.extend(_encode_pair(dict(state["currentPair"])))
    vector.extend(_encode_next_queue(list(state["nextQueue"]), max_next_pairs))
    vector.append(min(float(state.get("turn", 0)) / 1000.0, 4.0))
    vector.append(min(float(state.get("totalScore", 0)) / 2_000_000.0, 4.0))
    vector.extend(_scaled_feature(features, key) for key in VALUE_FEATURE_KEYS)
    return vector


def _future_label(sample: dict, target_horizon: int) -> dict:
    future = sample.get("future", {})
    label = future.get(str(target_horizon)) or future.get(target_horizon)
    if not isinstance(label, dict):
        available = ", ".join(str(key) for key in future.keys()) or "none"
        raise ValueError(
            f"Value sample is missing horizon {target_horizon}; available={available}"
        )
    if not label.get("complete", False):
        raise ValueError(f"Value sample horizon {target_horizon} is incomplete")
    return label


def build_value_targets(sample: dict, *, target_horizon: int = 48) -> list[float]:
    """Build multi-task labels from one streamed value sample."""

    label = _future_label(sample, target_horizon)
    max_chains = float(label.get("maxChains", 0.0) or 0.0)
    chains10 = float(label.get("chains10Plus", 0.0) or 0.0)
    chains11 = float(label.get("chains11Plus", 0.0) or 0.0)
    chains12 = float(label.get("chains12Plus", 0.0) or 0.0)
    chains13 = float(label.get("chains13Plus", 0.0) or 0.0)
    total_score = float(label.get("totalScore", 0.0) or 0.0)
    topout = 1.0 if label.get("topout", False) else 0.0

    objective = (
        min((max_chains / 13.0) ** 3, 1.3)
        + min(chains10, 4.0) * 0.08
        + min(chains11, 3.0) * 0.18
        + min(chains12, 2.0) * 0.34
        + min(chains13, 1.0) * 0.75
        + min(total_score / 500_000.0, 1.0) * 0.12
        - topout * 0.45
    )

    return [
        objective,
        min(max_chains / 14.0, 1.2),
        min(chains10 / 3.0, 1.0),
        min(chains11 / 2.0, 1.0),
        min(chains12, 1.0),
        topout,
    ]


def encode_value_sample(
    sample: dict,
    *,
    target_horizon: int = 48,
    max_next_pairs: int = MAX_NEXT_PAIRS,
) -> EncodedValueSample:
    """Encode one streamed value sample into dense inputs and targets."""

    if sample.get("kind") != "search_value":
        raise ValueError("Expected a search_value sample")
    return EncodedValueSample(
        inputs=_encode_state(sample, max_next_pairs),
        targets=build_value_targets(sample, target_horizon=target_horizon),
    )
