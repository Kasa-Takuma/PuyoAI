"""Train a learned policy that imitates the search AI."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyTorch is required to train the learned policy. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.action_vocab import ACTION_KEYS
from training.data import (
    MAX_NEXT_PAIRS,
    build_encoded_dataset,
    load_policy_records,
    split_records,
)
from training.model import PolicyMLP, choose_device, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", required=True, help="Path to Export Slim JSON")
    parser.add_argument(
        "--focus",
        default=None,
        help="Optional path to Export 6+ Focus JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Checkpoint path, for example models/policy_mlp.pt",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--focus-weight", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=16.0)
    parser.add_argument("--max-next-pairs", type=int, default=MAX_NEXT_PAIRS)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, mps, cuda",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_tensor_dataset(encoded_dataset):
    return TensorDataset(
        torch.tensor(encoded_dataset.inputs, dtype=torch.float32),
        torch.tensor(encoded_dataset.labels, dtype=torch.long),
        torch.tensor(encoded_dataset.sample_weights, dtype=torch.float32),
        torch.tensor(encoded_dataset.teacher_probs, dtype=torch.float32),
        torch.tensor(encoded_dataset.teacher_masks, dtype=torch.float32),
        torch.tensor(encoded_dataset.focus_masks, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device, *, distill_weight: float) -> dict:
    model.eval()
    total = 0
    total_loss = 0.0
    total_focus = 0
    correct_top1 = 0
    correct_top3 = 0
    focus_top1 = 0

    for inputs, labels, sample_weights, teacher_probs, teacher_masks, focus_masks in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)
        teacher_probs = teacher_probs.to(device)
        teacher_masks = teacher_masks.to(device)
        focus_masks = focus_masks.to(device)

        logits = model(inputs)
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        distill_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)
        loss = ((ce_loss * sample_weights) + (distill_loss * teacher_masks * distill_weight)).mean()

        predictions = logits.argmax(dim=-1)
        top3 = logits.topk(k=3, dim=-1).indices

        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct_top1 += (predictions == labels).sum().item()
        correct_top3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

        focus_count = int(focus_masks.sum().item())
        total_focus += focus_count
        if focus_count > 0:
            focus_top1 += ((predictions == labels).float() * focus_masks).sum().item()

    metrics = {
        "loss": total_loss / max(total, 1),
        "top1": correct_top1 / max(total, 1),
        "top3": correct_top3 / max(total, 1),
        "samples": total,
        "focus_top1": focus_top1 / max(total_focus, 1) if total_focus else None,
        "focus_samples": total_focus,
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.max_next_pairs <= 0:
        raise SystemExit("--max-next-pairs must be at least 1.")
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in the range [0.0, 1.0).")

    set_global_seed(args.seed)

    device = choose_device(args.device)

    slim_records, focus_records = load_policy_records(
        args.slim,
        args.focus,
        focus_weight=args.focus_weight,
    )

    slim_train, slim_val = split_records(
        slim_records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    focus_train, focus_val = split_records(
        focus_records,
        val_fraction=args.val_fraction,
        seed=args.seed + 1,
    )

    train_records = slim_train + focus_train
    val_records = slim_val + focus_val

    if not train_records:
        raise SystemExit("No training records found.")

    train_encoded = build_encoded_dataset(
        train_records,
        max_next_pairs=args.max_next_pairs,
        teacher_temperature=args.teacher_temperature,
    )
    val_encoded = build_encoded_dataset(
        val_records,
        max_next_pairs=args.max_next_pairs,
        teacher_temperature=args.teacher_temperature,
    )

    train_loader = DataLoader(
        build_tensor_dataset(train_encoded),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        build_tensor_dataset(val_encoded),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PolicyMLP(
        train_encoded.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        json.dumps(
            {
                "stage": "dataset",
                "device": str(device),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "slim_train_records": len(slim_train),
                "slim_val_records": len(slim_val),
                "focus_train_records": len(focus_train),
                "focus_val_records": len(focus_val),
                "input_dim": train_encoded.input_dim,
                "max_next_pairs": args.max_next_pairs,
            },
            ensure_ascii=False,
        )
    )

    best_val_top1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for inputs, labels, sample_weights, teacher_probs, teacher_masks, _focus_masks in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            teacher_probs = teacher_probs.to(device)
            teacher_masks = teacher_masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce_loss = F.cross_entropy(logits, labels, reduction="none")
            distill_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            loss = ((ce_loss * sample_weights) + (distill_loss * teacher_masks * args.distill_weight)).mean()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = epoch_loss / max(sample_count, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            distill_weight=args.distill_weight,
        ) if val_records else {
            "loss": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "samples": 0,
            "focus_top1": None,
            "focus_samples": 0,
        }

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_focus_top1": val_metrics["focus_top1"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if val_metrics["top1"] >= best_val_top1:
            best_val_top1 = val_metrics["top1"]
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                output_path,
                model=model,
                metadata={
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "slim_train_records": len(slim_train),
                    "slim_val_records": len(slim_val),
                    "focus_train_records": len(focus_train),
                    "focus_val_records": len(focus_val),
                    "max_next_pairs": args.max_next_pairs,
                    "focus_weight": args.focus_weight,
                    "teacher_temperature": args.teacher_temperature,
                    "distill_weight": args.distill_weight,
                    "best_val_top1": best_val_top1,
                    "history": history,
                    "action_keys": ACTION_KEYS,
                },
            )

    print(
        json.dumps(
            {
                "saved_to": args.output,
                "best_val_top1": best_val_top1,
                "device": str(device),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
