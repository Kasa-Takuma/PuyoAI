"""Train the solo value MLP from completed fixed-policy trajectories."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import apply_model, encode_inputs, init_params, load_web_model, web_model_from_params


@dataclass
class Dataset:
    boards: np.ndarray
    next_pairs: np.ndarray
    targets: np.ndarray
    deaths: np.ndarray
    games: np.ndarray

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            self.boards[mask],
            self.next_pairs[mask],
            self.targets[mask],
            self.deaths[mask],
            self.games[mask],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-data",
        default="training/solo_search/artifacts/initial-v13-trajectories.jsonl.gz",
    )
    parser.add_argument("--new-data", action="append", default=[])
    parser.add_argument("--initial-fraction", type=float, default=None)
    parser.add_argument("--resume-web-model")
    parser.add_argument("--output-dir", default="training/solo_search/artifacts/model-initial")
    parser.add_argument("--name", default="solo_value_initial")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory_data(paths: list[Path]) -> Dataset:
    boards: list[np.ndarray] = []
    next_pairs: list[np.ndarray] = []
    targets: list[float] = []
    deaths: list[float] = []
    games: list[str] = []
    for path in paths:
        open_trajectory = gzip.open if path.suffix == ".gz" else open
        with open_trajectory(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") != "sample":
                    continue
                board = np.frombuffer(base64.b64decode(record["board"]), dtype=np.uint8)
                if board.size != 84:
                    raise ValueError(f"Bad board in {path}: {board.size} cells")
                encoded_pairs = np.zeros((2, 2), dtype=np.uint8)
                supplied = np.asarray(record["nextPairs"], dtype=np.uint8)
                encoded_pairs[: supplied.shape[0]] = supplied[:2]
                boards.append(board.reshape(6, 14))
                next_pairs.append(encoded_pairs)
                targets.append(float(record["targetValue"]))
                deaths.append(float(bool(record["deathWithin128"])))
                games.append(str(record["seed"]))
    if not boards:
        raise ValueError("No completed trajectory samples were found")
    return Dataset(
        boards=np.stack(boards),
        next_pairs=np.stack(next_pairs),
        targets=np.asarray(targets, dtype=np.float32),
        deaths=np.asarray(deaths, dtype=np.float32),
        games=np.asarray(games),
    )


def validation_game(seed: str) -> bool:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10 == 0


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    validation_mask = np.asarray([validation_game(seed) for seed in dataset.games])
    if not np.any(validation_mask) or np.all(validation_mask):
        unique = sorted(set(dataset.games.tolist()))
        validation_games = set(unique[::10] or unique[:1])
        validation_mask = np.asarray([seed in validation_games for seed in dataset.games])
    return dataset.subset(~validation_mask), dataset.subset(validation_mask)


def permute_colors(key: jax.Array, boards: jax.Array, pairs: jax.Array):
    permutation = jax.random.permutation(key, jnp.arange(1, 5, dtype=jnp.uint8))
    mapping = jnp.concatenate([jnp.zeros((1,), dtype=jnp.uint8), permutation])
    return mapping[boards], mapping[pairs]


def huber(error: jax.Array) -> jax.Array:
    absolute = jnp.abs(error)
    return jnp.where(absolute <= 1, 0.5 * error**2, absolute - 0.5)


def loss_and_parts(params, boards, pairs, known, targets, deaths):
    output = apply_model(params, encode_inputs(boards, pairs, known))
    value_loss = jnp.mean(huber(output[:, 0] - targets))
    death_loss = jnp.mean(jax.nn.softplus(output[:, 1]) - deaths * output[:, 1])
    return value_loss + 0.1 * death_loss, (value_loss, death_loss)


loss_gradient = jax.jit(jax.value_and_grad(loss_and_parts, has_aux=True))


@jax.jit
def adam_update(params, gradients, first, second, step, learning_rate):
    beta1 = 0.9
    beta2 = 0.999
    first = jax.tree.map(lambda old, grad: beta1 * old + (1 - beta1) * grad, first, gradients)
    second = jax.tree.map(
        lambda old, grad: beta2 * old + (1 - beta2) * grad * grad, second, gradients
    )
    first_hat = jax.tree.map(lambda value: value / (1 - beta1**step), first)
    second_hat = jax.tree.map(lambda value: value / (1 - beta2**step), second)
    params = jax.tree.map(
        lambda value, mean, variance: value
        - learning_rate * mean / (jnp.sqrt(variance) + 1e-8),
        params,
        first_hat,
        second_hat,
    )
    return params, first, second


@jax.jit
def validation_loss(params, boards, pairs, known, targets, deaths):
    return loss_and_parts(params, boards, pairs, known, targets, deaths)


def sample_epoch_indices(
    rng: np.random.Generator,
    initial_count: int,
    new_count: int,
    initial_fraction: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if new_count == 0:
        indices = rng.permutation(initial_count)
        return indices, np.zeros_like(indices, dtype=np.bool_)
    fraction = 0.3 if initial_fraction is None else float(np.clip(initial_fraction, 0, 1))
    total = new_count
    wanted_initial = round(total * fraction / max(1e-9, 1 - fraction))
    initial_indices = rng.choice(initial_count, size=wanted_initial, replace=wanted_initial > initial_count)
    new_indices = rng.permutation(new_count)
    indices = np.concatenate([initial_indices, new_indices])
    source_new = np.concatenate(
        [np.zeros(initial_indices.size, dtype=np.bool_), np.ones(new_indices.size, dtype=np.bool_)]
    )
    order = rng.permutation(indices.size)
    return indices[order], source_new[order]


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, params, first, second, step: int, epoch: int) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    arrays = {f"param_{key}": np.asarray(value) for key, value in params.items()}
    arrays.update({f"adam_first_{key}": np.asarray(value) for key, value in first.items()})
    arrays.update({f"adam_second_{key}": np.asarray(value) for key, value in second.items()})
    arrays["step"] = np.asarray(step)
    arrays["epoch"] = np.asarray(epoch)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = Path(args.initial_data)
    new_paths = [Path(item) for item in args.new_data]
    initial_train, initial_validation = split_dataset(load_trajectory_data([initial_path]))
    if new_paths:
        new_train, new_validation = split_dataset(load_trajectory_data(new_paths))
    else:
        new_train = new_validation = None

    key = jax.random.key(args.seed)
    if args.resume_web_model:
        params, _ = load_web_model(args.resume_web_model)
    else:
        params = init_params(key)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float("inf")
    best_params = params
    stale_epochs = 0
    step = 0
    started = time.perf_counter()

    validation_sets = [initial_validation] + ([new_validation] if new_validation else [])
    validation_boards = jnp.asarray(np.concatenate([item.boards for item in validation_sets]))
    validation_pairs = jnp.asarray(np.concatenate([item.next_pairs for item in validation_sets]))
    validation_targets = jnp.asarray(np.concatenate([item.targets for item in validation_sets]))
    validation_deaths = jnp.asarray(np.concatenate([item.deaths for item in validation_sets]))
    validation_known = jnp.arange(validation_boards.shape[0], dtype=jnp.int32) % 3

    for epoch in range(1, args.epochs + 1):
        indices, source_new = sample_epoch_indices(
            rng,
            len(initial_train.boards),
            len(new_train.boards) if new_train else 0,
            args.initial_fraction,
        )
        epoch_losses = []
        for start in range(0, indices.size, args.batch_size):
            selection = indices[start : start + args.batch_size]
            selection_new = source_new[start : start + args.batch_size]
            if new_train:
                boards = np.empty((selection.size, 6, 14), dtype=np.uint8)
                pairs = np.empty((selection.size, 2, 2), dtype=np.uint8)
                targets = np.empty(selection.size, dtype=np.float32)
                deaths = np.empty(selection.size, dtype=np.float32)
                old_mask = ~selection_new
                boards[old_mask] = initial_train.boards[selection[old_mask]]
                pairs[old_mask] = initial_train.next_pairs[selection[old_mask]]
                targets[old_mask] = initial_train.targets[selection[old_mask]]
                deaths[old_mask] = initial_train.deaths[selection[old_mask]]
                boards[selection_new] = new_train.boards[selection[selection_new]]
                pairs[selection_new] = new_train.next_pairs[selection[selection_new]]
                targets[selection_new] = new_train.targets[selection[selection_new]]
                deaths[selection_new] = new_train.deaths[selection[selection_new]]
            else:
                boards = initial_train.boards[selection]
                pairs = initial_train.next_pairs[selection]
                targets = initial_train.targets[selection]
                deaths = initial_train.deaths[selection]
            key, color_key = jax.random.split(key)
            board_batch, pair_batch = permute_colors(
                color_key, jnp.asarray(boards), jnp.asarray(pairs)
            )
            known = jnp.asarray(rng.integers(0, 3, size=selection.size), dtype=jnp.int32)
            (loss, parts), gradients = loss_gradient(
                params,
                board_batch,
                pair_batch,
                known,
                jnp.asarray(targets),
                jnp.asarray(deaths),
            )
            step += 1
            params, first, second = adam_update(
                params, gradients, first, second, step, args.learning_rate
            )
            epoch_losses.append((float(loss), float(parts[0]), float(parts[1])))

        (valid_total, valid_parts) = validation_loss(
            params,
            validation_boards,
            validation_pairs,
            validation_known,
            validation_targets,
            validation_deaths,
        )
        record = {
            "epoch": epoch,
            "trainLoss": float(np.mean([item[0] for item in epoch_losses])),
            "trainValueHuber": float(np.mean([item[1] for item in epoch_losses])),
            "trainDeathBce": float(np.mean([item[2] for item in epoch_losses])),
            "validationLoss": float(valid_total),
            "validationValueHuber": float(valid_parts[0]),
            "validationDeathBce": float(valid_parts[1]),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        save_checkpoint(output_dir / "latest.npz", params, first, second, step, epoch)
        if record["validationLoss"] < best_loss - 1e-6:
            best_loss = record["validationLoss"]
            best_params = jax.tree.map(lambda value: jnp.array(value), params)
            stale_epochs = 0
            save_checkpoint(output_dir / "best.npz", params, first, second, step, epoch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    metadata = {
        "trainingSeed": args.seed,
        "discount": 0.99,
        "horizon": 128,
        "initialData": str(initial_path.resolve()),
        "initialDataSha256": file_sha256(initial_path),
        "newData": [
            {"path": str(item.resolve()), "sha256": file_sha256(item)} for item in new_paths
        ],
        "initialFraction": args.initial_fraction,
        "trainSamples": len(initial_train.boards)
        + (len(new_train.boards) if new_train else 0),
        "validationSamples": int(validation_boards.shape[0]),
        "epochsCompleted": len(history),
        "bestValidationLoss": best_loss,
        "jaxVersion": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "elapsedSeconds": time.perf_counter() - started,
    }
    web_model = web_model_from_params(best_params, name=args.name, metadata=metadata)
    atomic_json(output_dir / "solo_value.web.json", web_model)
    metadata["webModelSha256"] = file_sha256(output_dir / "solo_value.web.json")
    atomic_json(output_dir / "training-report.json", {"metadata": metadata, "history": history})
    print(json.dumps({"status": "complete", **metadata}, indent=2))


if __name__ == "__main__":
    main()
