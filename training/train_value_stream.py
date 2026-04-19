"""Train a value function while a CLI search run streams samples."""

from __future__ import annotations

import argparse
import json
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyTorch is required to train the value function. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.data import MAX_NEXT_PAIRS
from training.model import (
    ValueMLP,
    choose_device,
    load_value_checkpoint,
    save_value_checkpoint,
)
from training.value_data import (
    VALUE_FEATURE_KEYS,
    VALUE_TARGET_NAMES,
    encode_value_sample,
)

DEFAULT_GENERATOR = "tools/generate-value-stream.js"
LOSS_WEIGHTS = [1.0, 0.5, 0.35, 0.45, 0.6, 0.35]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="models/value_mlp.pt",
        help="Checkpoint path. Default: models/value_mlp.pt",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional value checkpoint to continue training from.",
    )
    parser.add_argument("--turns", type=int, default=100_000)
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Split generation across N games. 0 means keep starting games until turns are reached.",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Node generators. --turns is split across them.",
    )
    parser.add_argument("--search-profile", default="chain_builder_v11")
    parser.add_argument("--seed", default="value-stream")
    parser.add_argument("--horizons", default="12,24,48")
    parser.add_argument("--target-horizon", type=int, default=48)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--train-every", type=int, default=32)
    parser.add_argument("--updates-per-tick", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-val-samples", type=int, default=10_000)
    parser.add_argument("--report-every", type=int, default=5_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--max-next-pairs", type=int, default=MAX_NEXT_PAIRS)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, mps, cuda",
    )
    return parser.parse_args()


def print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def split_total(total: int, parts: int) -> list[int]:
    """Split a non-negative total across parts while preserving the sum."""

    safe_total = max(0, total)
    safe_parts = max(1, parts)
    base = safe_total // safe_parts
    remainder = safe_total % safe_parts
    return [base + (1 if index < remainder else 0) for index in range(safe_parts)]


def build_generator_command(
    args: argparse.Namespace,
    *,
    turns: int,
    games: int,
    seed: str,
) -> list[str]:
    return [
        "node",
        args.generator,
        "--turns",
        str(max(1, turns)),
        "--games",
        str(max(0, games)),
        "--depth",
        str(max(1, min(4, args.depth))),
        "--beam-width",
        str(max(4, min(96, args.beam_width))),
        "--search-profile",
        args.search_profile,
        "--seed",
        seed,
        "--horizons",
        args.horizons,
        "--report-every",
        str(max(1, args.report_every)),
    ]


def build_generator_specs(args: argparse.Namespace) -> list[dict]:
    """Create one generator command per worker.

    The requested turn count is total work, not per-worker work. Games are also
    divided when provided; a game count of zero keeps each worker in open-ended
    game mode until its turn allocation is reached.
    """

    worker_count = max(1, args.workers)
    turn_chunks = [chunk for chunk in split_total(max(1, args.turns), worker_count) if chunk > 0]
    game_chunks = (
        split_total(max(0, args.games), len(turn_chunks))
        if args.games > 0
        else [0] * len(turn_chunks)
    )
    specs = []
    for index, turns in enumerate(turn_chunks):
        worker_id = index + 1
        seed = args.seed if len(turn_chunks) == 1 else f"{args.seed}:worker-{worker_id}"
        games = game_chunks[index] if index < len(game_chunks) else 0
        specs.append(
            {
                "worker_id": worker_id,
                "turns": turns,
                "games": games,
                "seed": seed,
                "command": build_generator_command(
                    args,
                    turns=turns,
                    games=games,
                    seed=seed,
                ),
            }
        )
    return specs


def read_generator_output(
    worker_id: int,
    process: subprocess.Popen,
    output_queue: queue.Queue,
) -> None:
    """Forward one generator's stdout lines to the training loop."""

    assert process.stdout is not None
    for line in process.stdout:
        output_queue.put((worker_id, line))
    output_queue.put((worker_id, None, process.wait()))


def make_batch(
    inputs_buffer: list[list[float]],
    targets_buffer: list[list[float]],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = [rng.randrange(len(inputs_buffer)) for _ in range(batch_size)]
    inputs = torch.tensor([inputs_buffer[index] for index in indices], dtype=torch.float32)
    targets = torch.tensor([targets_buffer[index] for index in indices], dtype=torch.float32)
    return inputs.to(device), targets.to(device)


def weighted_mse(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (F.mse_loss(predictions, targets, reduction="none") * weights).mean()


def train_updates(
    model: ValueMLP,
    optimizer: torch.optim.Optimizer,
    inputs_buffer: list[list[float]],
    targets_buffer: list[list[float]],
    *,
    batch_size: int,
    updates: int,
    rng: random.Random,
    device: torch.device,
    loss_weights: torch.Tensor,
) -> float | None:
    if len(inputs_buffer) < batch_size:
        return None

    model.train()
    losses = []
    for _ in range(updates):
        inputs, targets = make_batch(
            inputs_buffer,
            targets_buffer,
            batch_size=batch_size,
            rng=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = weighted_mse(predictions, targets, loss_weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(
    model: ValueMLP | None,
    val_inputs: list[list[float]],
    val_targets: list[list[float]],
    *,
    batch_size: int,
    device: torch.device,
    loss_weights: torch.Tensor,
) -> dict:
    if model is None or not val_inputs:
        return {
            "val_loss": None,
            "val_objective_mae": None,
            "val_max_chain_mae": None,
            "val_samples": len(val_inputs),
        }

    model.eval()
    total_loss = 0.0
    total_samples = 0
    objective_abs_error = 0.0
    max_chain_abs_error = 0.0

    for start in range(0, len(val_inputs), batch_size):
        inputs = torch.tensor(val_inputs[start : start + batch_size], dtype=torch.float32).to(device)
        targets = torch.tensor(val_targets[start : start + batch_size], dtype=torch.float32).to(device)
        predictions = model(inputs)
        loss = weighted_mse(predictions, targets, loss_weights)
        count = targets.shape[0]
        total_loss += float(loss.detach().cpu().item()) * count
        total_samples += count
        objective_abs_error += torch.abs(predictions[:, 0] - targets[:, 0]).sum().item()
        max_chain_abs_error += torch.abs(predictions[:, 1] - targets[:, 1]).sum().item()

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_objective_mae": objective_abs_error / max(total_samples, 1),
        "val_max_chain_mae": max_chain_abs_error * 14.0 / max(total_samples, 1),
        "val_samples": total_samples,
    }


def save_model(
    output_path: str | Path,
    *,
    model: ValueMLP,
    args: argparse.Namespace,
    input_dim: int,
    samples_seen: int,
    train_samples: int,
    val_samples: int,
    updates: int,
    best_val_loss: float | None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_value_checkpoint(
        output,
        model=model,
        metadata={
            "kind": "streamed_value_function",
            "samples_seen": samples_seen,
            "train_buffer_samples": train_samples,
            "val_samples": val_samples,
            "updates": updates,
            "best_val_loss": best_val_loss,
            "input_dim": input_dim,
            "max_next_pairs": args.max_next_pairs,
            "target_horizon": args.target_horizon,
            "target_names": VALUE_TARGET_NAMES,
            "feature_keys": VALUE_FEATURE_KEYS,
            "generator": {
                "turns": args.turns,
                "games": args.games,
                "workers": args.workers,
                "depth": args.depth,
                "beam_width": args.beam_width,
                "search_profile": args.search_profile,
                "seed": args.seed,
                "horizons": args.horizons,
            },
            "resume": args.resume,
        },
    )


def best_output_path(output_path: str | Path) -> Path:
    """Derive the companion checkpoint path used for the best validation model."""

    output = Path(output_path)
    return output.with_name(f"{output.stem}.best{output.suffix}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.replay_size < args.batch_size:
        raise SystemExit("--replay-size must be at least --batch-size.")
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in the range [0.0, 1.0).")

    device = choose_device(args.device)
    rng = random.Random(args.seed)
    generator_specs = build_generator_specs(args)
    loss_weights = torch.tensor(LOSS_WEIGHTS, dtype=torch.float32, device=device)
    resume_payload = None
    resumed_metadata = {}
    resume_updates = 0
    resume_samples_seen = 0
    resume_best_val_loss = None

    if args.resume is not None:
        resume_model_path = Path(args.resume)
        if not resume_model_path.exists():
            raise SystemExit(f"--resume checkpoint does not exist: {resume_model_path}")
        resumed_model, resume_payload = load_value_checkpoint(resume_model_path, device=device)
        resumed_metadata = dict(resume_payload.get("metadata", {}))
        resume_updates = int(resumed_metadata.get("updates", 0) or 0)
        resume_samples_seen = int(resumed_metadata.get("samples_seen", 0) or 0)
        metadata_best = resumed_metadata.get("best_val_loss")
        if metadata_best is not None:
            resume_best_val_loss = float(metadata_best)
    else:
        resumed_model = None

    print_json(
        {
            "stage": "start",
            "device": str(device),
            "resume": args.resume,
            "resume_updates": resume_updates,
            "resume_samples_seen": resume_samples_seen,
            "workers": len(generator_specs),
            "generator_specs": [
                {
                    "worker_id": spec["worker_id"],
                    "turns": spec["turns"],
                    "games": spec["games"],
                    "seed": spec["seed"],
                    "command": spec["command"],
                }
                for spec in generator_specs
            ],
            "output": args.output,
            "best_output": str(best_output_path(args.output)),
            "target_horizon": args.target_horizon,
            "replay_size": args.replay_size,
        }
    )

    processes: list[subprocess.Popen] = []
    output_queue: queue.Queue = queue.Queue()
    try:
        for spec in generator_specs:
            process = subprocess.Popen(
                spec["command"],
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            processes.append(process)
            threading.Thread(
                target=read_generator_output,
                args=(spec["worker_id"], process, output_queue),
                daemon=True,
            ).start()
    except FileNotFoundError as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise SystemExit(
            "Node.js is required for streaming generation. "
            "Install Node.js, or run the generator separately and add JSONL input support later."
        ) from exc

    model: ValueMLP | None = resumed_model
    optimizer: torch.optim.Optimizer | None = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if model is not None
        else None
    )
    train_inputs: list[list[float]] = []
    train_targets: list[list[float]] = []
    val_inputs: list[list[float]] = []
    val_targets: list[list[float]] = []
    samples_seen = 0
    skipped_samples = 0
    updates = resume_updates
    last_loss: float | None = None
    best_val_loss: float | None = resume_best_val_loss
    input_dim = model.input_dim if model is not None else 0
    started_at = time.perf_counter()
    active_generators = len(processes)
    generator_errors: list[dict] = []
    interrupted = False
    if model is not None:
        print_json(
            {
                "stage": "resume",
                "checkpoint": args.resume,
                "input_dim": input_dim,
                "updates": updates,
                "best_val_loss": best_val_loss,
                "metadata": resumed_metadata,
            }
        )

    while active_generators > 0:
        try:
            worker_id, line, *return_code = output_queue.get()
        except KeyboardInterrupt:
            interrupted = True
            print_json(
                {
                    "stage": "interrupted",
                    "samples_seen": samples_seen,
                    "message": "Stopping generators and saving the latest model if available.",
                }
            )
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            break

        if line is None:
            active_generators -= 1
            code = return_code[0] if return_code else 0
            if code != 0:
                generator_errors.append({"worker_id": worker_id, "return_code": code})
            continue

        stripped = line.strip()
        if not stripped:
            continue
        try:
            sample = json.loads(stripped)
            encoded = encode_value_sample(
                sample,
                target_horizon=args.target_horizon,
                max_next_pairs=args.max_next_pairs,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            skipped_samples += 1
            print_json(
                {
                    "stage": "skip",
                    "worker_id": worker_id,
                    "samples_seen": samples_seen,
                    "skipped_samples": skipped_samples,
                    "error": str(exc),
                }
            )
            continue

        if model is None:
            input_dim = encoded.input_dim
            model = ValueMLP(
                input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                output_dim=len(VALUE_TARGET_NAMES),
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            print_json(
                {
                    "stage": "model",
                    "input_dim": input_dim,
                    "target_names": VALUE_TARGET_NAMES,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                }
            )
        elif encoded.input_dim != input_dim:
            raise SystemExit(
                f"Encoded input_dim changed from {input_dim} to {encoded.input_dim}. "
                "Check --max-next-pairs and checkpoint compatibility."
            )

        samples_seen += 1
        use_for_val = (
            len(val_inputs) < args.max_val_samples and rng.random() < args.val_fraction
        )
        if use_for_val:
            val_inputs.append(encoded.inputs)
            val_targets.append(encoded.targets)
        else:
            train_inputs.append(encoded.inputs)
            train_targets.append(encoded.targets)
            if len(train_inputs) > args.replay_size:
                overflow = len(train_inputs) - args.replay_size
                del train_inputs[:overflow]
                del train_targets[:overflow]

        if (
            optimizer is not None
            and samples_seen % max(1, args.train_every) == 0
            and len(train_inputs) >= args.batch_size
        ):
            loss = train_updates(
                model,
                optimizer,
                train_inputs,
                train_targets,
                batch_size=args.batch_size,
                updates=max(1, args.updates_per_tick),
                rng=rng,
                device=device,
                loss_weights=loss_weights,
            )
            if loss is not None:
                last_loss = loss
                updates += max(1, args.updates_per_tick)

        should_report = samples_seen % max(1, args.report_every) == 0
        should_save = samples_seen % max(1, args.save_every) == 0
        if should_report or should_save:
            metrics = evaluate(
                model,
                val_inputs,
                val_targets,
                batch_size=args.batch_size,
                device=device,
                loss_weights=loss_weights,
            )
            val_loss = metrics["val_loss"]
            improved = val_loss is not None and (
                best_val_loss is None or val_loss <= best_val_loss
            )
            if improved:
                best_val_loss = float(val_loss)

            saved_latest = False
            saved_best = False
            if should_save and model is not None:
                save_model(
                    args.output,
                    model=model,
                    args=args,
                    input_dim=input_dim,
                    samples_seen=samples_seen,
                    train_samples=len(train_inputs),
                    val_samples=len(val_inputs),
                    updates=updates,
                    best_val_loss=best_val_loss,
                )
                saved_latest = True

                if improved:
                    save_model(
                        best_output_path(args.output),
                        model=model,
                        args=args,
                        input_dim=input_dim,
                        samples_seen=samples_seen,
                        train_samples=len(train_inputs),
                        val_samples=len(val_inputs),
                        updates=updates,
                        best_val_loss=best_val_loss,
                    )
                    saved_best = True

            print_json(
                {
                    "stage": "progress",
                    "samples_seen": samples_seen,
                    "skipped_samples": skipped_samples,
                    "train_buffer_samples": len(train_inputs),
                    "updates": updates,
                    "last_train_loss": last_loss,
                    "elapsed_sec": round(time.perf_counter() - started_at, 1),
                    **metrics,
                    "saved": saved_latest,
                    "saved_latest": saved_latest,
                    "saved_best": saved_best,
                }
            )

    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    if generator_errors and not interrupted:
        raise SystemExit(f"Value stream generator failed: {generator_errors}")
    if model is None:
        raise SystemExit("No value samples were received from the generator.")

    metrics = evaluate(
        model,
        val_inputs,
        val_targets,
        batch_size=args.batch_size,
        device=device,
        loss_weights=loss_weights,
    )
    val_loss = metrics["val_loss"]
    final_improved = val_loss is not None and (
        best_val_loss is None or val_loss <= best_val_loss
    )
    if final_improved:
        best_val_loss = float(val_loss)

    save_model(
        args.output,
        model=model,
        args=args,
        input_dim=input_dim,
        samples_seen=samples_seen,
        train_samples=len(train_inputs),
        val_samples=len(val_inputs),
        updates=updates,
        best_val_loss=best_val_loss,
    )
    if final_improved:
        save_model(
            best_output_path(args.output),
            model=model,
            args=args,
            input_dim=input_dim,
            samples_seen=samples_seen,
            train_samples=len(train_inputs),
            val_samples=len(val_inputs),
            updates=updates,
            best_val_loss=best_val_loss,
        )

    print_json(
        {
            "stage": "complete",
            "samples_seen": samples_seen,
            "skipped_samples": skipped_samples,
            "train_buffer_samples": len(train_inputs),
            "updates": updates,
            "saved_to": args.output,
            "best_saved_to": str(best_output_path(args.output)),
            "elapsed_sec": round(time.perf_counter() - started_at, 1),
            **metrics,
        }
    )


if __name__ == "__main__":
    main()
