"""Dataset loading and encoding for learned policies."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from training.action_vocab import ACTION_KEYS, ACTION_TO_INDEX

BOARD_HEIGHT = 14
BOARD_WIDTH = 6
BOARD_CELLS = BOARD_HEIGHT * BOARD_WIDTH
BOARD_SYMBOLS = [".", "R", "G", "B", "Y"]
PLAYABLE_COLORS = ["R", "G", "B", "Y"]
MAX_NEXT_PAIRS = 5

BOARD_SYMBOL_TO_INDEX = {symbol: index for index, symbol in enumerate(BOARD_SYMBOLS)}
PLAYABLE_COLOR_TO_INDEX = {symbol: index for index, symbol in enumerate(PLAYABLE_COLORS)}


@dataclass(slots=True)
class PolicyRecord:
    """Normalized training sample used by the learned policy."""

    kind: str
    board_rows: list[str]
    current_pair: dict[str, str]
    next_queue: list[dict[str, str]]
    best_action_key: str
    objective: str
    depth: int
    beam_width: int
    top_candidates: list[dict]
    sample_weight: float
    is_focus: bool
    focus_metadata: dict | None


@dataclass(slots=True)
class EncodedPolicyDataset:
    """Encoded tensors represented as Python lists before torch conversion."""

    inputs: list[list[float]]
    labels: list[int]
    sample_weights: list[float]
    teacher_probs: list[list[float]]
    teacher_masks: list[float]
    focus_masks: list[float]

    @property
    def input_dim(self) -> int:
        return len(self.inputs[0]) if self.inputs else 0


def load_json_dataset(path: str | Path) -> list[dict]:
    """Load a JSON dataset exported by the web app."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def _normalize_top_candidates(record: dict) -> list[dict]:
    if "topCandidates" in record and isinstance(record["topCandidates"], list):
        return list(record["topCandidates"])

    if "candidates" in record and isinstance(record["candidates"], list):
        candidates = [
            {
                "actionKey": candidate["actionKey"],
                "searchScore": candidate["searchScore"],
            }
            for candidate in record["candidates"][:3]
            if "actionKey" in candidate and "searchScore" in candidate
        ]
        return candidates

    return []


def _normalize_record(record: dict, default_weight: float, focus_weight: float) -> PolicyRecord:
    kind = record.get("kind", "")
    state = record.get("state", {})
    search = record.get("search", {})
    focus = record.get("focus")

    if "boardRows" not in state or "currentPair" not in state or "nextQueue" not in state:
        raise ValueError("Record is missing required state fields")

    sample_weight = focus_weight if kind == "search_policy_chain_focus" else default_weight

    return PolicyRecord(
        kind=kind,
        board_rows=list(state["boardRows"]),
        current_pair=dict(state["currentPair"]),
        next_queue=list(state["nextQueue"]),
        best_action_key=record["bestActionKey"],
        objective=search.get("objective", "unknown"),
        depth=int(search.get("settings", {}).get("depth", 3)),
        beam_width=int(search.get("settings", {}).get("beamWidth", 24)),
        top_candidates=_normalize_top_candidates(record),
        sample_weight=sample_weight,
        is_focus=kind == "search_policy_chain_focus",
        focus_metadata=focus if isinstance(focus, dict) else None,
    )


def normalize_policy_record(
    record: dict,
    *,
    default_weight: float = 1.0,
    focus_weight: float = 2.0,
) -> PolicyRecord:
    """Public wrapper for normalizing one exported search sample."""

    return _normalize_record(record, default_weight, focus_weight)


def load_policy_records(
    slim_path: str | Path,
    focus_path: str | Path | None = None,
    *,
    default_weight: float = 1.0,
    focus_weight: float = 2.0,
) -> tuple[list[PolicyRecord], list[PolicyRecord]]:
    """Load and normalize slim and optional chain-focus datasets."""

    slim_records = [
        _normalize_record(record, default_weight, focus_weight)
        for record in load_json_dataset(slim_path)
    ]
    focus_records: list[PolicyRecord] = []
    if focus_path is not None:
        focus_records = [
            _normalize_record(record, default_weight, focus_weight)
            for record in load_json_dataset(focus_path)
        ]
    return slim_records, focus_records


def split_records(
    records: list[PolicyRecord],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[PolicyRecord], list[PolicyRecord]]:
    """Deterministically split records into train and validation subsets."""

    if not records:
        return [], []

    copied = list(records)
    random.Random(seed).shuffle(copied)
    val_size = max(1, int(len(copied) * val_fraction)) if len(copied) > 1 else 0
    if val_size == 0:
        return copied, []
    return copied[val_size:], copied[:val_size]


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


def _encode_search_settings(record: PolicyRecord) -> list[float]:
    return [
        record.depth / 4.0,
        record.beam_width / 96.0,
    ]


def _encode_teacher_distribution(
    top_candidates: list[dict],
    *,
    temperature: float,
) -> tuple[list[float], float]:
    if not top_candidates:
        return [0.0] * len(ACTION_KEYS), 0.0

    logits = [candidate["searchScore"] / max(temperature, 1e-6) for candidate in top_candidates]
    max_logit = max(logits)
    exp_scores = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_scores)
    distribution = [0.0] * len(ACTION_KEYS)

    for candidate, exp_score in zip(top_candidates, exp_scores, strict=True):
        distribution[ACTION_TO_INDEX[candidate["actionKey"]]] = exp_score / total

    return distribution, 1.0


def encode_policy_record(
    record: PolicyRecord,
    *,
    max_next_pairs: int = MAX_NEXT_PAIRS,
    teacher_temperature: float = 1.0,
) -> tuple[list[float], int, float, list[float], float, float]:
    """Encode a normalized record into numeric features and labels."""

    vector = []
    vector.extend(_encode_board(record.board_rows))
    vector.extend(_encode_pair(record.current_pair))
    vector.extend(_encode_next_queue(record.next_queue, max_next_pairs))
    vector.extend(_encode_search_settings(record))

    label = ACTION_TO_INDEX[record.best_action_key]
    teacher_probs, teacher_mask = _encode_teacher_distribution(
        record.top_candidates,
        temperature=teacher_temperature,
    )
    focus_mask = 1.0 if record.is_focus else 0.0

    return vector, label, record.sample_weight, teacher_probs, teacher_mask, focus_mask


def encode_policy_state(
    *,
    board_rows: list[str],
    current_pair: dict[str, str],
    next_queue: list[dict[str, str]],
    depth: int = 3,
    beam_width: int = 24,
    max_next_pairs: int = MAX_NEXT_PAIRS,
) -> list[float]:
    """Encode a state-only payload for learned-policy inference."""

    vector = []
    vector.extend(_encode_board(list(board_rows)))
    vector.extend(_encode_pair(dict(current_pair)))
    vector.extend(_encode_next_queue(list(next_queue), max_next_pairs))
    vector.extend(
        _encode_search_settings(
            PolicyRecord(
                kind="inference_state",
                board_rows=list(board_rows),
                current_pair=dict(current_pair),
                next_queue=list(next_queue),
                best_action_key=ACTION_KEYS[0],
                objective="inference",
                depth=depth,
                beam_width=beam_width,
                top_candidates=[],
                sample_weight=1.0,
                is_focus=False,
                focus_metadata=None,
            )
        )
    )
    return vector


def build_encoded_dataset(
    records: list[PolicyRecord],
    *,
    max_next_pairs: int = MAX_NEXT_PAIRS,
    teacher_temperature: float = 1.0,
) -> EncodedPolicyDataset:
    """Encode a list of records into dense numeric arrays."""

    inputs: list[list[float]] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    teacher_probs: list[list[float]] = []
    teacher_masks: list[float] = []
    focus_masks: list[float] = []

    for record in records:
        vector, label, sample_weight, teacher_prob, teacher_mask, focus_mask = encode_policy_record(
            record,
            max_next_pairs=max_next_pairs,
            teacher_temperature=teacher_temperature,
        )
        inputs.append(vector)
        labels.append(label)
        sample_weights.append(sample_weight)
        teacher_probs.append(teacher_prob)
        teacher_masks.append(teacher_mask)
        focus_masks.append(focus_mask)

    return EncodedPolicyDataset(
        inputs=inputs,
        labels=labels,
        sample_weights=sample_weights,
        teacher_probs=teacher_probs,
        teacher_masks=teacher_masks,
        focus_masks=focus_masks,
    )
