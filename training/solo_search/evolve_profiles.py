"""GPU-batched evolutionary tuning starting from chain_builder_v12_ac."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.82")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .jax_board import BOARD_HEIGHT, BOARD_WIDTH
from .jax_profile_search import PROFILE_KEYS, play_profile_batch_sharded

BASE_PROFILE = np.asarray(
    [
        738.0, 3.25039, 0.89989, -42_148.0, -150_194.0, -49_504.0,
        -70_464.0, -53_129.0, 105_948.0, 202_440.0, 512_657.0, 450_000.0,
        942.0, 403.0, 0.73346, 0.12942, 189.0, 198.0, 70.0, -221.0,
        -14.0, -77.0, 1.0, 1.0,
    ],
    dtype=np.float64,
)
SPREADS = np.asarray(
    [
        .08, .025, .06, .16, .16, .18, .18, .18, .16, .18, .20, .16,
        .08, .10, .10, .12, .12, .10, .10, .12, .12, .12, .10, .14,
    ],
    dtype=np.float64,
)
TURN_KEYS = PROFILE_KEYS[:12]
BOARD_KEYS = PROFILE_KEYS[12:22]


@dataclass
class Candidate:
    id: str
    values: np.ndarray
    parent_id: str | None
    source: str

    def profile_config(self) -> dict:
        values = dict(zip(PROFILE_KEYS, (float(value) for value in self.values)))
        return {
            "id": self.id,
            "label": self.id.replace("_", " ").title(),
            "baseProfileId": "chain_builder_v12_ac",
            "turnWeights": {key: values[key] for key in TURN_KEYS},
            "boardWeights": {key: values[key] for key in BOARD_KEYS},
            "bonusScales": {
                "largeChain": values["largeChainScale"],
                "v9b": values["v9bScale"],
            },
        }

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "parentId": self.parent_id,
            "source": self.source,
            "values": self.values.tolist(),
            "profileConfig": self.profile_config(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Candidate":
        return cls(
            str(payload["id"]),
            np.asarray(payload["values"], dtype=np.float64),
            payload.get("parentId"),
            str(payload.get("source", "resume")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--stage-turns", default="3000,6000,12000")
    parser.add_argument("--stage-keeps", default="12,4")
    parser.add_argument("--stage-games", default="2,3,4")
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--profile-batch", default="auto")
    parser.add_argument("--seed", default="auto")
    parser.add_argument("--output", default="/kaggle/working/v12ac-evolution/report.json")
    parser.add_argument("--resume-report")
    parser.add_argument("--min-improvement-pct", type=float, default=.03)
    parser.add_argument("--hall-of-fame", type=int, default=8)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    args.stage_turns = tuple(int(value) for value in args.stage_turns.split(","))
    args.stage_keeps = tuple(int(value) for value in args.stage_keeps.split(","))
    args.stage_games = tuple(int(value) for value in args.stage_games.split(","))
    if len(args.stage_turns) != 3 or len(args.stage_keeps) != 2 or len(args.stage_games) != 3:
        parser.error("stage-turns needs 3 values, stage-keeps 2, and stage-games 3")
    return args


def gpu_snapshot() -> list[dict]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return [
            {"name": line.rsplit(",", 1)[0].strip(), "memoryMiB": int(line.rsplit(",", 1)[1])}
            for line in output.strip().splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def choose_profile_batch(requested: str, gpus: list[dict]) -> int:
    if requested != "auto":
        return max(1, int(requested))
    if not gpus:
        return 2
    memory = gpus[0]["memoryMiB"]
    if memory >= 22_000:
        return 18
    if memory >= 14_000:
        return 12
    return 8


def hash_seed(text: str) -> int:
    state = 2166136261
    for byte in text.encode("utf-8"):
        state ^= byte
        state = (state * 16777619) & 0xFFFFFFFF
    return state or 0x12345678


def pair_stream(seed: str, turns: int) -> np.ndarray:
    initial = [(1, 2), (3, 4), (2, 1), (4, 3)]
    required_pairs = turns + 2
    if required_pairs <= len(initial):
        return np.asarray(initial[:required_pairs], dtype=np.uint8)
    state = hash_seed(f"sandbox:{seed}")
    values = []
    for _ in range((required_pairs - len(initial)) * 2):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        values.append(state % 4 + 1)
    generated = np.asarray(values, dtype=np.uint8).reshape(-1, 2)
    return np.concatenate([np.asarray(initial, dtype=np.uint8), generated], axis=0)


def round_metrics(metrics: dict) -> dict:
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def summarize_candidate(
    candidate: Candidate,
    chains: np.ndarray,
    scores: np.ndarray,
    topouts: np.ndarray,
    all_clears: np.ndarray,
    planned_turns: int,
) -> dict:
    live_chains = chains[chains >= 0]
    count = lambda low: int(np.count_nonzero(live_chains >= low))
    below7 = int(np.count_nonzero((live_chains > 0) & (live_chains < 7)))
    seven_to_nine = int(np.count_nonzero((live_chains >= 7) & (live_chains <= 9)))
    scale = 10_000 / planned_turns
    topout_count = int(np.count_nonzero(topouts))
    all_clear_count = int(np.count_nonzero(all_clears))
    metrics = {
        "id": candidate.id,
        "plannedTurns": planned_turns,
        "executedTurns": int(live_chains.size),
        "topouts": topout_count,
        "totalScore": int(scores.sum()),
        "scorePerPlannedTurn": float(scores.sum() / planned_turns),
        "bestChain": int(live_chains.max(initial=0)),
        "chainsBelow7": below7,
        "chains7To9": seven_to_nine,
        "chains10Plus": count(10),
        "chains11Plus": count(11),
        "chains12Plus": count(12),
        "chains13Plus": count(13),
        "allClears": all_clear_count,
        "below7Per10k": below7 * scale,
        "sevenToNinePer10k": seven_to_nine * scale,
        "tenPlusPer10k": count(10) * scale,
        "elevenPlusPer10k": count(11) * scale,
        "twelvePlusPer10k": count(12) * scale,
        "thirteenPlusPer10k": count(13) * scale,
        "allClearsPer10k": all_clear_count * scale,
    }
    metrics["objectiveScore"] = (
        metrics["tenPlusPer10k"] * 10
        + metrics["elevenPlusPer10k"] * 48
        + metrics["twelvePlusPer10k"] * 120
        + metrics["thirteenPlusPer10k"] * 260
        + metrics["allClearsPer10k"] * 8
        + metrics["scorePerPlannedTurn"] * .04
        - metrics["below7Per10k"] * .22
        - metrics["sevenToNinePer10k"] * .23
        - topout_count * 1200
    )
    return round_metrics(metrics)


def evaluate_candidates(
    candidates: list[Candidate],
    *,
    turns: int,
    games: int,
    seeds: list[str],
    beam_width: int,
    profile_batch: int,
) -> list[dict]:
    turns_per_game = (turns + games - 1) // games
    streams = np.stack([pair_stream(seed, turns_per_game) for seed in seeds])
    results = []
    device_count = max(1, jax.local_device_count())
    profiles_per_chunk = profile_batch * device_count
    for offset in range(0, len(candidates), profiles_per_chunk):
        chunk = candidates[offset : offset + profiles_per_chunk]
        actual = len(chunk)
        padded = chunk + [chunk[-1]] * (profiles_per_chunk - actual)
        weights = np.repeat(np.stack([item.values for item in padded]), games, axis=0)
        pair_batch = np.tile(streams, (profiles_per_chunk, 1, 1))
        weights = weights.reshape(device_count, profile_batch * games, -1)
        pair_batch = pair_batch.reshape(
            device_count, profile_batch * games, turns_per_game + 2, 2
        )
        boards = jnp.zeros(
            (device_count, profile_batch * games, BOARD_WIDTH, BOARD_HEIGHT),
            dtype=jnp.uint8,
        )
        started = time.perf_counter()
        traces = play_profile_batch_sharded(
            boards,
            jnp.asarray(pair_batch),
            jnp.asarray(weights),
            beam_width,
        )
        jax.block_until_ready(traces)
        chain_trace, score_trace, topout_trace, all_clear_trace = (
            np.asarray(value).transpose(0, 2, 1).reshape(profiles_per_chunk, games, -1)
            for value in traces
        )
        elapsed = time.perf_counter() - started
        for index, candidate in enumerate(chunk):
            metrics = summarize_candidate(
                candidate,
                chain_trace[index],
                score_trace[index],
                topout_trace[index],
                all_clear_trace[index],
                turns,
            )
            metrics["gpuBatchSeconds"] = round(elapsed, 3)
            results.append({"candidate": candidate, "summary": metrics})
            print(json.dumps({"stage": "candidate", **metrics}), flush=True)
    return sorted(results, key=lambda item: item["summary"]["objectiveScore"], reverse=True)


def normalized_distance(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1)
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def mutate(source: Candidate, spread: float, rng: np.random.Generator, candidate_id: str, source_name: str) -> Candidate:
    factors = 1 + rng.uniform(-1, 1, BASE_PROFILE.size) * SPREADS * spread
    values = source.values * factors
    values[1] = np.clip(values[1], 2.7, 3.8)
    values[14:16] = np.maximum(values[14:16], 0)
    values[22:24] = np.clip(values[22:24], .25, 2.5)
    return Candidate(candidate_id, values.astype(np.float64), source.id, source_name)


def make_generation(
    champion: Candidate,
    hall: list[Candidate],
    population: int,
    generation: int,
    stagnation: int,
    rng: np.random.Generator,
    seen: list[np.ndarray],
) -> list[Candidate]:
    generated = []
    boost = min(2.3, 1 + max(0, stagnation - 1) * .18)
    attempts = 0
    while len(generated) < population and attempts < population * 300:
        attempts += 1
        slot = len(generated) + 1
        ratio = slot / population
        if ratio <= .5:
            parent, spread, source = champion, .75, "champion_small"
        elif ratio <= .73:
            parent, spread, source = champion, 1.25, "champion_broad"
        elif ratio <= .9 and hall:
            parent, spread, source = hall[int(rng.integers(len(hall)))], 1.0, "hall_of_fame"
        else:
            parent, spread, source = champion, 1.85, "explore"
        token = hashlib.sha1(rng.bytes(12)).hexdigest()[:6]
        candidate = mutate(
            parent,
            spread * boost,
            rng,
            f"evolve_v12ac_g{generation:03d}_c{slot:03d}_{token}",
            f"{source}@x{spread * boost:.3f}",
        )
        if any(normalized_distance(candidate.values, other) < .006 for other in seen):
            continue
        generated.append(candidate)
        seen.append(candidate.values.copy())
    return generated


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact_result(result: dict) -> dict:
    return {
        "summary": result["summary"],
        **result["candidate"].to_json(),
    }


def main() -> None:
    args = parse_args()
    gpus = gpu_snapshot()
    profile_batch = choose_profile_batch(args.profile_batch, gpus)
    root_seed = str(uuid.uuid4()) if args.seed == "auto" else args.seed
    seed_number = int(hashlib.sha256(root_seed.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed_number)
    output = Path(args.output)
    baseline = Candidate("chain_builder_v12_ac", BASE_PROFILE.copy(), None, "baseline")
    prior = None
    if args.resume_report:
        prior = json.loads(Path(args.resume_report).read_text(encoding="utf-8"))
        champion = Candidate.from_json(prior["champion"])
        hall = [Candidate.from_json(item) for item in prior.get("hallOfFame", [])]
        generations = list(prior.get("generations", []))
        start_generation = max((item["generation"] for item in generations), default=0) + 1
        stagnation = int(prior.get("stagnation", 0))
    else:
        champion, hall, generations, start_generation, stagnation = baseline, [], [], 1, 0
    seen = [baseline.values.copy(), champion.values.copy(), *(item.values.copy() for item in hall)]
    report = {
        "kind": "puyoai_v12ac_gpu_evolution_report",
        "version": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "rootSeed": root_seed,
        "settings": {
            "generations": args.generations,
            "population": args.population,
            "stageTurns": args.stage_turns,
            "stageKeeps": args.stage_keeps,
            "stageGames": args.stage_games,
            "beamWidth": args.beam_width,
            "profileBatch": profile_batch,
            "minImprovementPct": args.min_improvement_pct,
        },
        "environment": {
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "gpus": gpus,
        },
        "resumedFrom": args.resume_report,
        "champion": champion.to_json(),
        "hallOfFame": [item.to_json() for item in hall],
        "generations": generations,
        "stagnation": stagnation,
    }
    print(json.dumps({"stage": "start", "output": str(output), **report["settings"], **report["environment"]}), flush=True)

    for generation in range(start_generation, start_generation + args.generations):
        generated = make_generation(champion, hall, args.population, generation, stagnation, rng, seen)
        protected = {item.id: item for item in [baseline, champion, *hall[:3]]}
        candidates = list(protected.values()) + generated
        record = {"generation": generation, "championAtStart": champion.id, "generated": [item.to_json() for item in generated], "stages": []}
        previous = None
        for stage_index, (turns, games) in enumerate(zip(args.stage_turns, args.stage_games)):
            if previous is not None:
                keep = args.stage_keeps[stage_index - 1]
                selected = [item["candidate"] for item in previous[:keep]]
                candidates = list({item.id: item for item in [*selected, *protected.values()]}.values())
            stage_seed = hashlib.sha256(f"{root_seed}:{generation}:{stage_index}:{rng.random()}".encode()).hexdigest()[:16]
            seeds = [f"v12ac-gpu:{generation}:s{stage_index + 1}:{stage_seed}:game-{index + 1}" for index in range(games)]
            print(json.dumps({"stage": "evaluation_start", "generation": generation, "tier": stage_index + 1, "candidates": len(candidates), "turns": turns, "games": games, "seeds": seeds}), flush=True)
            previous = evaluate_candidates(
                candidates,
                turns=turns,
                games=games,
                seeds=seeds,
                beam_width=args.beam_width,
                profile_batch=profile_batch,
            )
            record["stages"].append({
                "tier": stage_index + 1,
                "turns": turns,
                "games": games,
                "seeds": seeds,
                "top": [compact_result(item) for item in previous[:args.top]],
            })
            report["activeGeneration"] = record
            write_report(output, report)
            print(json.dumps({"stage": "evaluation_complete", "generation": generation, "tier": stage_index + 1, "top": [item["summary"] for item in previous[:args.top]]}), flush=True)

        champion_result = next((item for item in previous if item["candidate"].id == champion.id), None)
        best = previous[0]
        promoted = bool(
            best["candidate"].id != champion.id
            and champion_result
            and best["summary"]["objectiveScore"] >= champion_result["summary"]["objectiveScore"] * (1 + args.min_improvement_pct)
            and best["summary"]["elevenPlusPer10k"] >= champion_result["summary"]["elevenPlusPer10k"] * .95
            and best["summary"]["topouts"] <= champion_result["summary"]["topouts"]
        )
        if promoted:
            champion = best["candidate"]
            stagnation = 0
        else:
            stagnation += 1
        hall_map = {item.id: item for item in hall}
        hall_scores = {item.id: -np.inf for item in hall}
        for item in previous:
            if item["candidate"].parent_id is not None:
                hall_map[item["candidate"].id] = item["candidate"]
                hall_scores[item["candidate"].id] = item["summary"]["objectiveScore"]
        hall = sorted(hall_map.values(), key=lambda item: hall_scores.get(item.id, -np.inf), reverse=True)[:args.hall_of_fame]
        record["championChanged"] = promoted
        record["championAtEnd"] = champion.id
        generations.append(record)
        report.pop("activeGeneration", None)
        report["champion"] = champion.to_json() | {"lastStageSummary": next(item["summary"] for item in previous if item["candidate"].id == champion.id)}
        report["hallOfFame"] = [item.to_json() for item in hall]
        report["stagnation"] = stagnation
        write_report(output, report)
        print(json.dumps({"stage": "generation_complete", "generation": generation, "championChanged": promoted, "champion": report["champion"], "output": str(output)}), flush=True)

    report["completedAt"] = datetime.now(timezone.utc).isoformat()
    write_report(output, report)
    print(json.dumps({"stage": "complete", "output": str(output), "champion": report["champion"]}), flush=True)


if __name__ == "__main__":
    main()
