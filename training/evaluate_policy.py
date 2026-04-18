"""Evaluate a learned policy checkpoint against exported search datasets."""

from __future__ import annotations

import argparse
import json
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
        "PyTorch is required to evaluate the learned policy. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.data import MAX_NEXT_PAIRS, build_encoded_dataset, load_policy_records
from training.model import choose_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--slim", required=True, help="Path to Export Slim JSON")
    parser.add_argument(
        "--focus",
        default=None,
        help="Optional path to Export 6+ Focus JSON",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, mps, cuda",
    )
    parser.add_argument(
        "--max-next-pairs",
        type=int,
        default=None,
        help="Override the checkpoint's encoded next count",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=None,
        help="Override the checkpoint's teacher temperature for evaluation loss",
    )
    return parser.parse_args()


def build_tensor_dataset(encoded_dataset) -> TensorDataset:
    return TensorDataset(
        torch.tensor(encoded_dataset.inputs, dtype=torch.float32),
        torch.tensor(encoded_dataset.labels, dtype=torch.long),
        torch.tensor(encoded_dataset.sample_weights, dtype=torch.float32),
        torch.tensor(encoded_dataset.teacher_probs, dtype=torch.float32),
        torch.tensor(encoded_dataset.teacher_masks, dtype=torch.float32),
        torch.tensor(encoded_dataset.focus_masks, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    distill_weight: float,
) -> dict:
    model.eval()
    total = 0
    total_loss = 0.0
    total_focus = 0
    correct_top1 = 0
    correct_top3 = 0
    focus_top1 = 0.0

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

        batch_size = labels.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        correct_top1 += (predictions == labels).sum().item()
        correct_top3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

        focus_count = int(focus_masks.sum().item())
        total_focus += focus_count
        if focus_count > 0:
            focus_top1 += ((predictions == labels).float() * focus_masks).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "top1": correct_top1 / max(total, 1),
        "top3": correct_top3 / max(total, 1),
        "samples": total,
        "focus_top1": focus_top1 / max(total_focus, 1) if total_focus else None,
        "focus_samples": total_focus,
    }


def evaluate_records(
    model,
    records,
    *,
    device: torch.device,
    batch_size: int,
    max_next_pairs: int,
    teacher_temperature: float,
    distill_weight: float,
) -> dict:
    if not records:
        return {
            "loss": None,
            "top1": None,
            "top3": None,
            "samples": 0,
            "focus_top1": None,
            "focus_samples": 0,
        }

    encoded = build_encoded_dataset(
        records,
        max_next_pairs=max_next_pairs,
        teacher_temperature=teacher_temperature,
    )
    loader = DataLoader(
        build_tensor_dataset(encoded),
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_loader(model, loader, device, distill_weight=distill_weight)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device=device)

    metadata = payload.get("metadata", {})
    max_next_pairs = args.max_next_pairs or metadata.get("max_next_pairs", MAX_NEXT_PAIRS)
    teacher_temperature = (
        args.teacher_temperature
        if args.teacher_temperature is not None
        else metadata.get("teacher_temperature", 16.0)
    )
    distill_weight = float(metadata.get("distill_weight", 0.0))

    slim_records, focus_records = load_policy_records(
        args.slim,
        args.focus,
        focus_weight=float(metadata.get("focus_weight", 2.0)),
    )
    combined_records = [*slim_records, *focus_records]

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "max_next_pairs": max_next_pairs,
        "teacher_temperature": teacher_temperature,
        "distill_weight": distill_weight,
        "checkpoint_metadata": metadata,
        "metrics": {
            "combined": evaluate_records(
                model,
                combined_records,
                device=device,
                batch_size=args.batch_size,
                max_next_pairs=max_next_pairs,
                teacher_temperature=teacher_temperature,
                distill_weight=distill_weight,
            ),
            "slim": evaluate_records(
                model,
                slim_records,
                device=device,
                batch_size=args.batch_size,
                max_next_pairs=max_next_pairs,
                teacher_temperature=teacher_temperature,
                distill_weight=distill_weight,
            ),
            "focus": evaluate_records(
                model,
                focus_records,
                device=device,
                batch_size=args.batch_size,
                max_next_pairs=max_next_pairs,
                teacher_temperature=teacher_temperature,
                distill_weight=distill_weight,
            ),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
