"""Run fixed-seed solo-search games on the JAX device and save event traces."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .jax_board import BOARD_HEIGHT, BOARD_WIDTH, resolve_turn
from .jax_search import search_batch
from .model import load_web_model, params_from_web_model


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_seed_games(
    path: Path,
    max_turns: int,
    limit_games: int | None = None,
) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload.get("games", [])
    if limit_games is not None:
        games = games[:limit_games]
    if not games:
        raise ValueError(f"No games found in {path}")
    required_pairs = max_turns + 2
    selected = []
    for game in games:
        pairs = np.asarray(game["pairs"], dtype=np.uint8)
        if pairs.shape[0] < required_pairs or pairs.shape[1:] != (2,):
            raise ValueError(
                f"Seed {game.get('seed')} has {pairs.shape[0]} pairs; "
                f"need {required_pairs}"
            )
        selected.append(pairs[:required_pairs])
    return [str(game["seed"]) for game in games], np.stack(selected)


def _game_metrics(chains: np.ndarray, topout: bool, planned_turns: int) -> dict:
    high_turns = np.flatnonzero(chains >= 10) + 1
    gaps = np.diff(np.concatenate(([0], high_turns, [planned_turns + 1]))) - 1
    return {
        "plannedTurns": planned_turns,
        "executedTurns": int(np.count_nonzero(chains >= 0)),
        "topout": bool(topout),
        "events": int(np.count_nonzero(chains > 0)),
        "atLeast10": int(np.count_nonzero(chains >= 10)),
        "atLeast11": int(np.count_nonzero(chains >= 11)),
        "atLeast12": int(np.count_nonzero(chains >= 12)),
        "atLeast13": int(np.count_nonzero(chains >= 13)),
        "atLeast14": int(np.count_nonzero(chains >= 14)),
        "longestHighChainGap": int(gaps.max(initial=planned_turns)),
        "firstHighChainTurn": int(high_turns[0]) if high_turns.size else None,
        "repeatedHighChainSegments": int(high_turns.size >= 2),
    }


def run_simulation(
    seed_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    max_turns: int = 1000,
    beam_width: int = 32,
    preserve_roots: bool = True,
    limit_games: int | None = None,
) -> dict:
    seed_path = Path(seed_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds, pair_array = load_seed_games(seed_path, max_turns, limit_games)
    params, raw_model = load_web_model(model_path)
    game_count = len(seeds)

    boards = jnp.zeros((game_count, BOARD_WIDTH, BOARD_HEIGHT), dtype=jnp.uint8)
    active = np.ones((game_count,), dtype=np.bool_)
    action_trace = np.full((game_count, max_turns), -1, dtype=np.int32)
    chain_trace = np.full((game_count, max_turns), -1, dtype=np.int16)
    score_trace = np.zeros((game_count, max_turns), dtype=np.int32)
    topout_trace = np.zeros((game_count, max_turns), dtype=np.bool_)
    expanded_trace = np.zeros((game_count, max_turns), dtype=np.int32)
    started = time.perf_counter()
    for turn in range(max_turns):
        pair_batch = jnp.asarray(pair_array[:, turn : turn + 3, :])
        best, _, _, expanded = search_batch(
            boards,
            pair_batch,
            params,
            beam_width=beam_width,
            preserve_roots=preserve_roots,
            value_at_leaf_only=False,
            dedupe=True,
        )
        resolved = jax.vmap(resolve_turn)(
            boards,
            pair_batch[:, 0, 0],
            pair_batch[:, 0, 1],
            best,
        )
        jax.block_until_ready(resolved)
        best_np = np.asarray(best)
        topout_np = np.asarray(resolved[1])
        chains_np = np.asarray(resolved[2])
        score_np = np.asarray(resolved[3])
        expanded_np = np.asarray(expanded)
        action_trace[:, turn] = np.where(active, best_np, -1)
        chain_trace[:, turn] = np.where(active, chains_np, -1)
        score_trace[:, turn] = np.where(active, score_np, 0)
        topout_trace[:, turn] = active & topout_np
        expanded_trace[:, turn] = np.where(active, expanded_np, 0)
        active &= ~topout_np
        boards = resolved[0]
        if not np.any(active):
            break

    per_game = [
        _game_metrics(
            chain_trace[index],
            bool(np.any(topout_trace[index])),
            max_turns,
        )
        | {"seed": seeds[index]}
        for index in range(game_count)
    ]
    chain_values = chain_trace[chain_trace >= 0]
    summary = {
        "games": game_count,
        "plannedTurns": game_count * max_turns,
        "executedTurns": int(np.count_nonzero(chain_trace >= 0)),
        "topoutGames": int(sum(item["topout"] for item in per_game)),
        "topoutRate": float(np.mean([item["topout"] for item in per_game])),
        "events": int(np.count_nonzero(chain_values > 0)),
        "atLeast10": int(np.count_nonzero(chain_values >= 10)),
        "atLeast11": int(np.count_nonzero(chain_values >= 11)),
        "atLeast12": int(np.count_nonzero(chain_values >= 12)),
        "atLeast13": int(np.count_nonzero(chain_values >= 13)),
        "atLeast14": int(np.count_nonzero(chain_values >= 14)),
        "repeatedHighChainSegments": int(
            sum(item["repeatedHighChainSegments"] for item in per_game)
        ),
        "maxChain": int(chain_values.max(initial=0)),
        "expandedPlacements": int(expanded_trace.sum()),
    }
    trace_path = output_dir / "simulation.npz"
    temporary_trace = output_dir / "simulation.tmp.npz"
    np.savez_compressed(
        temporary_trace,
        actions=action_trace,
        chains=chain_trace,
        scores=score_trace,
        topout=topout_trace,
        expanded=expanded_trace,
        seeds=np.asarray(seeds),
    )
    temporary_trace.replace(trace_path)
    report = {
        "format": "puyoai-solo-jax-simulation-v2",
        "seedPath": str(seed_path.resolve()),
        "seedSha256": file_sha256(seed_path),
        "modelPath": str(model_path.resolve()),
        "modelSha256": file_sha256(model_path),
        "maxTurns": max_turns,
        "beamWidth": beam_width,
        "searchScoring": "structure-turn-prior-model30-v1",
        "preserveRootActions": preserve_roots,
        "jaxVersion": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "implementationSha256": {
            name: file_sha256(Path(__file__).resolve().parent / name)
            for name in (
                "jax_board.py",
                "jax_search.py",
                "jax_structure.py",
                "model.py",
                "simulate.py",
            )
        },
        "elapsedSeconds": time.perf_counter() - started,
        "summary": summary,
        "perGame": per_game,
        "trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
    }
    atomic_json(output_dir / "simulation-report.json", report)
    return report
