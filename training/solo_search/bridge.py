"""Persistent JSON-lines bridge used to compare JAX with JavaScript."""

from __future__ import annotations

import argparse
import base64
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .jax_search import search_batch
from .model import load_web_model, predict


def decode_boards(values: list[str]) -> np.ndarray:
    result = []
    for value in values:
        board = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
        if board.size != 84:
            raise ValueError(f"Expected 84 board cells, got {board.size}")
        result.append(board.reshape(6, 14))
    return np.stack(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    params, _ = load_web_model(args.model)
    print(json.dumps({"type": "ready", "jax": jax.__version__, "devices": [str(x) for x in jax.devices()]}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request["id"]
        try:
            boards = jnp.asarray(decode_boards(request["boards"]))
            if request["type"] == "predict":
                pairs = jnp.asarray(request["nextPairs"], dtype=jnp.uint8)
                known = jnp.asarray(request["knownCounts"], dtype=jnp.int32)
                output = predict(params, boards, pairs, known)
                jax.block_until_ready(output)
                response = {"id": request_id, "outputs": np.asarray(output).tolist()}
            elif request["type"] == "search":
                pairs = jnp.asarray(request["pairs"], dtype=jnp.uint8)
                output = search_batch(
                    boards,
                    pairs,
                    params,
                    beam_width=int(request.get("beamWidth", 32)),
                    preserve_roots=bool(request.get("preserveRootActions", True)),
                    value_at_leaf_only=bool(request.get("valueAtLeafOnly", False)),
                )
                jax.block_until_ready(output)
                root_scores = np.asarray(output[1])
                response = {
                    "id": request_id,
                    "actions": np.asarray(output[0]).tolist(),
                    "rootScores": np.where(np.isfinite(root_scores), root_scores, -1e30).tolist(),
                    "paths": np.asarray(output[2]).tolist(),
                    "expanded": np.asarray(output[3]).tolist(),
                }
            else:
                raise ValueError(f"Unknown request type: {request['type']}")
            print(json.dumps(response, separators=(",", ":")), flush=True)
        except Exception as error:
            print(json.dumps({"id": request_id, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
