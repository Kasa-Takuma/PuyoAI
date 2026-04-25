"""Dataset loading and encoding for policy-gradient rollouts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from training.action_vocab import ACTION_KEYS, ACTION_TO_INDEX
from training.data import MAX_NEXT_PAIRS, encode_policy_state


@dataclass(slots=True)
class EncodedRlDataset:
    """Encoded rollout steps used by policy-gradient training."""

    inputs: list[list[float]]
    actions: list[int]
    returns: list[float]
    advantages: list[float]
    behavior_log_probs: list[float]
    behavior_masks: list[float]
    legal_masks: list[list[float]]
    episode_count: int
    step_count: int
    return_mean: float
    return_std: float

    @property
    def input_dim(self) -> int:
        return len(self.inputs[0]) if self.inputs else 0


def _read_json_payloads(path: str | Path) -> Iterable[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {source}")
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
            return

        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            yield payload


def _iter_steps(payload: dict) -> tuple[int, list[tuple[dict, dict]]]:
    if payload.get("kind") == "policy_rl_episode":
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("policy_rl_episode.steps must be a list")
        return 1, [(payload, step) for step in steps if isinstance(step, dict)]

    if payload.get("kind") == "policy_rl_step":
        return 0, [({}, payload)]

    if "steps" in payload and isinstance(payload["steps"], list):
        return 1, [(payload, step) for step in payload["steps"] if isinstance(step, dict)]

    return 0, []


def _settings_for_step(episode: dict, step: dict) -> tuple[int, int]:
    state_settings = step.get("state", {}).get("settings", {})
    policy = episode.get("policy", {})
    depth = int(state_settings.get("depth", policy.get("depthFeature", 3)) or 3)
    beam_width = int(state_settings.get("beamWidth", policy.get("beamWidthFeature", 24)) or 24)
    return depth, beam_width


def _encode_step_input(step: dict, episode: dict, max_next_pairs: int) -> list[float]:
    state = step.get("state", {})
    depth, beam_width = _settings_for_step(episode, step)
    return encode_policy_state(
        board_rows=list(state["boardRows"]),
        current_pair=dict(state["currentPair"]),
        next_queue=list(state["nextQueue"]),
        depth=depth,
        beam_width=beam_width,
        max_next_pairs=max_next_pairs,
    )


def _legal_mask(step: dict) -> list[float]:
    legal_keys = step.get("legalActionKeys")
    if not isinstance(legal_keys, list) or not legal_keys:
        legal_keys = ACTION_KEYS

    mask = [0.0] * len(ACTION_KEYS)
    for action_key in legal_keys:
        index = ACTION_TO_INDEX.get(str(action_key))
        if index is not None:
            mask[index] = 1.0

    action_index = ACTION_TO_INDEX[step["actionKey"]]
    mask[action_index] = 1.0
    return mask


def _standardize(values: list[float], clip: float | None) -> tuple[list[float], float, float]:
    if not values:
        return [], 0.0, 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(variance, 1e-12))
    standardized = [(value - mean) / std for value in values]
    if clip is not None and clip > 0:
        standardized = [max(-clip, min(clip, value)) for value in standardized]
    return standardized, mean, std


def load_encoded_rl_dataset(
    paths: list[str | Path],
    *,
    max_next_pairs: int = MAX_NEXT_PAIRS,
    normalize_advantages: bool = True,
    advantage_clip: float | None = 5.0,
) -> EncodedRlDataset:
    """Load JSONL rollout episodes and encode them for policy-gradient training."""

    inputs: list[list[float]] = []
    actions: list[int] = []
    returns: list[float] = []
    behavior_log_probs: list[float] = []
    behavior_masks: list[float] = []
    legal_masks: list[list[float]] = []
    episode_count = 0

    for path in paths:
        for payload in _read_json_payloads(path):
            episode_increment, steps = _iter_steps(payload)
            episode_count += episode_increment
            for episode, step in steps:
                action_key = step.get("actionKey")
                if action_key not in ACTION_TO_INDEX:
                    raise ValueError(f"Unknown actionKey in rollout: {action_key!r}")
                if "return" not in step:
                    raise ValueError("Rollout step is missing return")

                inputs.append(_encode_step_input(step, episode, max_next_pairs))
                actions.append(ACTION_TO_INDEX[action_key])
                returns.append(float(step["return"]))
                legal_masks.append(_legal_mask(step))

                behavior_log_prob = step.get("behaviorLogProb")
                if behavior_log_prob is None:
                    behavior_log_probs.append(0.0)
                    behavior_masks.append(0.0)
                else:
                    behavior_log_probs.append(float(behavior_log_prob))
                    behavior_masks.append(1.0)

    if normalize_advantages:
        advantages, return_mean, return_std = _standardize(returns, advantage_clip)
    else:
        advantages = list(returns)
        return_mean = sum(returns) / len(returns) if returns else 0.0
        return_std = 1.0

    return EncodedRlDataset(
        inputs=inputs,
        actions=actions,
        returns=returns,
        advantages=advantages,
        behavior_log_probs=behavior_log_probs,
        behavior_masks=behavior_masks,
        legal_masks=legal_masks,
        episode_count=episode_count,
        step_count=len(inputs),
        return_mean=return_mean,
        return_std=return_std,
    )
