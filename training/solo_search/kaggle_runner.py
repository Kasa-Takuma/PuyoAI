"""Entry point used by the reproducible Kaggle GPU jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# These must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.42")

import jax
import jax.numpy as jnp
import numpy as np

from .jax_board import resolve_turn_batch
from .jax_search import search_batch
from .model import init_params, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    parser.add_argument("--run-id", default="solo-smoke-20260914-01")
    parser.add_argument("--output-dir", default="/kaggle/working/solo-search")
    parser.add_argument("--fixtures", default="jax-fixtures.jsonl.gz")
    parser.add_argument("--source-commit", default="8db100a412568e12f1dadd239611c349e41bc353")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_resume_state() -> tuple[dict | None, dict[str, np.ndarray] | None]:
    candidates = sorted(Path("/kaggle/input").glob("**/resume-state.json"))
    if not candidates:
        return None, None
    path = candidates[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = path.parent / payload["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint was not found: {checkpoint}")
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != payload["checkpointSha256"]:
        raise ValueError(
            f"Resume checkpoint hash mismatch: {checkpoint_sha256} != {payload['checkpointSha256']}"
        )
    with np.load(checkpoint) as loaded:
        required = ("boards", "pairs", "processed", "rng_state")
        missing = [name for name in required if name not in loaded]
        if missing:
            raise ValueError(f"Resume checkpoint is missing arrays: {missing}")
        arrays = {name: np.asarray(loaded[name]) for name in required}
    return (
        {
            "path": str(path),
            "sha256": sha256(path),
            "checkpointPath": str(checkpoint),
            "checkpointSha256": checkpoint_sha256,
            "payload": payload,
        },
        arrays,
    )


def benchmark_call(function, *, warmup: bool = True, iterations: int = 3):
    started = time.perf_counter()
    output = function()
    jax.block_until_ready(output)
    first_seconds = time.perf_counter() - started
    steady = []
    if warmup:
        for _ in range(iterations):
            started = time.perf_counter()
            output = function()
            jax.block_until_ready(output)
            steady.append(time.perf_counter() - started)
    return output, first_seconds, steady


def gpu_snapshot() -> list[dict]:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        lines = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()
        result = []
        for line in lines:
            index, name, total, used, utilization = [item.strip() for item in line.split(",")]
            result.append(
                {
                    "index": int(index),
                    "name": name,
                    "memoryTotalMiB": int(total),
                    "memoryUsedMiB": int(used),
                    "utilizationPercent": int(utilization),
                }
            )
        return result
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def verify_fixtures(fixtures: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "solo_search.verify_jax_board",
        "--fixtures",
        str(fixtures),
        "--batch-size",
        "2048",
    ]
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fixture verification failed with exit {completed.returncode}: {completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def smoke_benchmarks(params) -> dict:
    rng = np.random.default_rng(20260914)
    resolution_count = 8192
    boards = jnp.zeros((resolution_count, 6, 14), dtype=jnp.uint8)
    axes = jnp.asarray(rng.integers(1, 5, size=resolution_count), dtype=jnp.uint8)
    children = jnp.asarray(rng.integers(1, 5, size=resolution_count), dtype=jnp.uint8)
    actions = jnp.asarray(rng.integers(0, 22, size=resolution_count), dtype=jnp.int32)
    _, board_first, board_steady = benchmark_call(
        lambda: resolve_turn_batch(boards, axes, children, actions)
    )

    model_count = 65536
    model_boards = jnp.zeros((model_count, 6, 14), dtype=jnp.uint8)
    model_pairs = jnp.asarray(rng.integers(1, 5, size=(model_count, 2, 2)), dtype=jnp.uint8)
    model_known = jnp.asarray(rng.integers(0, 3, size=model_count), dtype=jnp.int32)
    _, model_first, model_steady = benchmark_call(
        lambda: predict(params, model_boards, model_pairs, model_known)
    )

    search_results = []
    for games in (32, 64, 128):
        search_boards = jnp.zeros((games, 6, 14), dtype=jnp.uint8)
        search_pairs = jnp.asarray(rng.integers(1, 5, size=(games, 3, 2)), dtype=jnp.uint8)
        output, first, steady = benchmark_call(
            lambda: search_batch(search_boards, search_pairs, params, beam_width=32),
            iterations=2,
        )
        expanded = int(np.asarray(output[3]).sum())
        transfer_started = time.perf_counter()
        np.asarray(output[0])
        transfer_seconds = time.perf_counter() - transfer_started
        median_steady = float(np.median(steady))
        search_results.append(
            {
                "games": games,
                "compileAndFirstSeconds": first,
                "steadySeconds": steady,
                "expandedPlacements": expanded,
                "placementsPerSecond": expanded / median_steady,
                "hostTransferSeconds": transfer_seconds,
            }
        )
    return {
        "boardResolution": {
            "batch": resolution_count,
            "compileAndFirstSeconds": board_first,
            "steadySeconds": board_steady,
            "placementsPerSecond": resolution_count / float(np.median(board_steady)),
        },
        "model": {
            "batch": model_count,
            "compileAndFirstSeconds": model_first,
            "steadySeconds": model_steady,
            "evaluationsPerSecond": model_count / float(np.median(model_steady)),
        },
        "fullSearch": search_results,
    }


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = Path(args.fixtures)
    environment = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "devices": [str(device) for device in jax.devices()],
        "gpuBefore": gpu_snapshot(),
    }
    fixture_report = verify_fixtures(fixtures)
    params = init_params(jax.random.key(20260914))
    benchmarks = smoke_benchmarks(params)
    resume, resume_arrays = find_resume_state()

    state_path = output_dir / "state.npz"
    temporary_state = output_dir / "state.tmp.npz"
    state_arrays = resume_arrays or {
        "boards": np.zeros((64, 6, 14), dtype=np.uint8),
        "pairs": np.zeros((64, 3, 2), dtype=np.uint8),
        "processed": np.asarray(0, dtype=np.int64),
        "rng_state": np.asarray(20260914, dtype=np.uint64),
    }
    np.savez_compressed(
        temporary_state,
        **state_arrays,
    )
    os.replace(temporary_state, state_path)
    resume_state = {
        "format": "puyoai-solo-resume-v1",
        "runId": args.run_id,
        "checkpoint": state_path.name,
        "checkpointSha256": sha256(state_path),
        "processedPlacements": int(state_arrays["processed"]),
        "pendingTeacherSamples": 0,
    }
    atomic_json(output_dir / "resume-state.json", resume_state)
    ended_at = datetime.now(timezone.utc)
    report = {
        "format": "puyoai-solo-kaggle-run-v1",
        "runId": args.run_id,
        "mode": args.mode,
        "sourceCommit": args.source_commit,
        "startedAt": started_at.isoformat(),
        "endedAt": ended_at.isoformat(),
        "elapsedSeconds": time.perf_counter() - started_perf,
        "environment": {**environment, "gpuAfter": gpu_snapshot()},
        "fixtures": {
            "path": str(fixtures),
            "sha256": sha256(fixtures),
            "verification": fixture_report,
        },
        "benchmarks": benchmarks,
        "resumedFrom": resume,
        "resumeState": resume_state,
    }
    atomic_json(output_dir / "run-report.json", report)
    manifest = {
        "format": "puyoai-solo-output-manifest-v1",
        "runId": args.run_id,
        "files": [
            {"path": item.name, "bytes": item.stat().st_size, "sha256": sha256(item)}
            for item in sorted(output_dir.iterdir())
            if item.is_file()
        ],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
