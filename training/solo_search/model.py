"""The 438-64-32-2 solo value model shared by JAX and JavaScript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

INPUT_DIM = 438
HIDDEN_DIMS = (64, 32)
OUTPUT_DIM = 2


def init_params(key: jax.Array) -> dict[str, jax.Array]:
    dimensions = (INPUT_DIM, *HIDDEN_DIMS, OUTPUT_DIM)
    keys = jax.random.split(key, len(dimensions) - 1)
    params: dict[str, jax.Array] = {}
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        params[f"w{index + 1}"] = jax.random.uniform(
            keys[index],
            (output_dim, input_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        )
        params[f"b{index + 1}"] = jnp.zeros((output_dim,), dtype=jnp.float32)
    return params


def encode_inputs(
    boards: jax.Array,
    next_pairs: jax.Array,
    known_counts: jax.Array,
) -> jax.Array:
    """Encode boards (N,6,14), pairs (N,2,2), and 0/1/2 known counts."""
    board_one_hot = jax.nn.one_hot(boards, 5, dtype=jnp.float32).reshape(boards.shape[0], -1)
    pair_indices = jnp.clip(next_pairs.astype(jnp.int32) - 1, 0, 3)
    pair_one_hot = jax.nn.one_hot(pair_indices, 4, dtype=jnp.float32).reshape(
        boards.shape[0], 2, 8
    )
    slots = jnp.arange(2, dtype=jnp.int32)[None, :]
    known = slots < known_counts[:, None]
    pair_one_hot = pair_one_hot * known[:, :, None]
    encoded_pairs = jnp.concatenate([pair_one_hot, known[:, :, None]], axis=2).reshape(
        boards.shape[0], 18
    )
    return jnp.concatenate([board_one_hot, encoded_pairs], axis=1)


def apply_model(params: dict[str, jax.Array], encoded: jax.Array) -> jax.Array:
    hidden1 = jax.nn.relu(encoded @ params["w1"].T + params["b1"])
    hidden2 = jax.nn.relu(hidden1 @ params["w2"].T + params["b2"])
    return hidden2 @ params["w3"].T + params["b3"]


def predict(
    params: dict[str, jax.Array],
    boards: jax.Array,
    next_pairs: jax.Array,
    known_counts: jax.Array,
) -> jax.Array:
    return apply_model(params, encode_inputs(boards, next_pairs, known_counts))


def params_from_web_model(raw_model: dict[str, Any]) -> dict[str, jax.Array]:
    layers = raw_model["layers"]
    if len(layers) != 3:
        raise ValueError("Solo value model must have three layers")
    params: dict[str, jax.Array] = {}
    expected = (INPUT_DIM, *HIDDEN_DIMS, OUTPUT_DIM)
    for index, layer in enumerate(layers):
        input_dim = expected[index]
        output_dim = expected[index + 1]
        if layer["inputDim"] != input_dim or layer["outputDim"] != output_dim:
            raise ValueError(f"Layer {index + 1} has unexpected dimensions")
        params[f"w{index + 1}"] = jnp.asarray(layer["weights"], dtype=jnp.float32).reshape(
            output_dim, input_dim
        )
        params[f"b{index + 1}"] = jnp.asarray(layer["bias"], dtype=jnp.float32)
    return params


def load_web_model(path: str | Path) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    raw_model = json.loads(Path(path).read_text(encoding="utf-8"))
    return params_from_web_model(raw_model), raw_model


def web_model_from_params(
    params: dict[str, jax.Array],
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dimensions = (INPUT_DIM, *HIDDEN_DIMS, OUTPUT_DIM)
    layers = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(
            {
                "inputDim": input_dim,
                "outputDim": output_dim,
                "activation": "relu" if index < 2 else "linear",
                "weights": np.asarray(params[f"w{index + 1}"], dtype=np.float32)
                .reshape(-1)
                .tolist(),
                "bias": np.asarray(params[f"b{index + 1}"], dtype=np.float32).tolist(),
            }
        )
    return {
        "format": "puyoai-solo-value-v1",
        "name": name,
        "inputDim": INPUT_DIM,
        "maxNextPairs": 2,
        "targetHorizon": 128,
        "targetNames": ["discounted_reward", "death_logit"],
        "architecture": [INPUT_DIM, *HIDDEN_DIMS, OUTPUT_DIM],
        "metadata": metadata or {},
        "layers": layers,
    }
