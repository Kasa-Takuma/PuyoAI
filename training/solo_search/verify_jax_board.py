"""Compare JAX turn resolution with fixtures produced by the JavaScript core."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import time
from pathlib import Path

import jax
import numpy as np

from .jax_board import resolve_turn_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default="training/solo_search/artifacts/jax-fixtures.jsonl.gz",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def decode_board(encoded: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    if raw.size != 84:
        raise ValueError(f"Expected 84 board cells, got {raw.size}")
    return raw.reshape(6, 14)


def verify_batch(inputs: list[tuple[np.ndarray, dict]], compiled: bool) -> tuple[int, bool, float]:
    boards = np.stack([item[0] for item in inputs])
    cases = [item[1] for item in inputs]
    axes = np.asarray([case["axis"] for case in cases], dtype=np.uint8)
    children = np.asarray([case["child"] for case in cases], dtype=np.uint8)
    actions = np.asarray([case["actionId"] for case in cases], dtype=np.int32)
    started = time.perf_counter()
    output = resolve_turn_batch(boards, axes, children, actions)
    jax.block_until_ready(output)
    elapsed = time.perf_counter() - started
    output_np = [np.asarray(value) for value in output]
    for index, case in enumerate(cases):
        expected_board = decode_board(case["board"])
        if not np.array_equal(output_np[0][index], expected_board):
            mismatch = np.argwhere(output_np[0][index] != expected_board)[0].tolist()
            raise AssertionError(f"Board mismatch at batch row {index}, cell {mismatch}")
        actual = {
            "topout": bool(output_np[1][index]),
            "chains": int(output_np[2][index]),
            "score": int(output_np[3][index]),
            "allClear": bool(output_np[4][index]),
        }
        expected = {key: case[key] for key in actual}
        if actual != expected:
            raise AssertionError(f"Result mismatch at batch row {index}: {actual} != {expected}")
        if bool(output_np[5][index]):
            raise AssertionError(f"Chain loop overflow at batch row {index}")
    return len(inputs), compiled, elapsed


def main() -> None:
    args = parse_args()
    pending: list[tuple[np.ndarray, dict]] = []
    boards_seen = 0
    cases_seen = 0
    compile_seconds = 0.0
    steady_seconds = 0.0
    first_batch = True
    fixtures = Path(args.fixtures)
    open_fixtures = gzip.open if fixtures.suffix == ".gz" else open
    with open_fixtures(fixtures, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] != "board":
                continue
            boards_seen += 1
            source = decode_board(record["board"])
            pending.extend((source, case) for case in record["cases"])
            while len(pending) >= args.batch_size:
                batch = pending[: args.batch_size]
                del pending[: args.batch_size]
                count, _, elapsed = verify_batch(batch, not first_batch)
                cases_seen += count
                if first_batch:
                    compile_seconds += elapsed
                    first_batch = False
                else:
                    steady_seconds += elapsed
    if pending:
        count, _, elapsed = verify_batch(pending, not first_batch)
        cases_seen += count
        if first_batch:
            compile_seconds += elapsed
        else:
            steady_seconds += elapsed

    print(
        json.dumps(
            {
                "status": "passed",
                "fixtures": str(Path(args.fixtures).resolve()),
                "boards": boards_seen,
                "cases": cases_seen,
                "compileAndFirstBatchSeconds": compile_seconds,
                "steadySeconds": steady_seconds,
                "steadyPlacementsPerSecond": cases_seen / steady_seconds if steady_seconds else None,
                "jaxVersion": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
