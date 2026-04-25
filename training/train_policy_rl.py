"""Fine-tune a learned policy from stochastic self-play rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyTorch is required to fine-tune the learned policy. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.action_vocab import ACTION_KEYS
from training.data import MAX_NEXT_PAIRS
from training.model import PolicyMLP, choose_device, load_checkpoint, save_checkpoint
from training.rl_data import load_encoded_rl_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts",
        nargs="+",
        required=True,
        help="One or more JSONL files from tools/generate-rl-rollouts.js",
    )
    parser.add_argument(
        "--init",
        default=None,
        help="Optional policy checkpoint to initialize from.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Checkpoint path, for example models/policy_rl.pt",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--kl-anchor-weight", type=float, default=0.02)
    parser.add_argument(
        "--ppo-clip",
        type=float,
        default=0.2,
        help="Importance-ratio clipping against rollout behavior log-probs. 0 disables clipping.",
    )
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--max-next-pairs", type=int, default=MAX_NEXT_PAIRS)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, mps, cuda",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_tensor_dataset(encoded):
    return TensorDataset(
        torch.tensor(encoded.inputs, dtype=torch.float32),
        torch.tensor(encoded.actions, dtype=torch.long),
        torch.tensor(encoded.advantages, dtype=torch.float32),
        torch.tensor(encoded.returns, dtype=torch.float32),
        torch.tensor(encoded.behavior_log_probs, dtype=torch.float32),
        torch.tensor(encoded.behavior_masks, dtype=torch.float32),
        torch.tensor(encoded.legal_masks, dtype=torch.bool),
    )


def masked_log_softmax(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~legal_masks, -1.0e9)
    return F.log_softmax(masked_logits, dim=-1)


def policy_gradient_loss(
    selected_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    behavior_masks: torch.Tensor,
    *,
    ppo_clip: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if ppo_clip <= 0 or behavior_masks.sum().item() == 0:
        return -(advantages * selected_log_probs).mean(), None

    ratios = torch.exp(selected_log_probs - behavior_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip)
    clipped_objective = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    unclipped_objective = advantages * selected_log_probs
    objective = torch.where(behavior_masks > 0, clipped_objective, unclipped_objective)
    approx_kl = ((behavior_log_probs - selected_log_probs) * behavior_masks).sum() / behavior_masks.sum()
    return -objective.mean(), approx_kl


def masked_entropy(log_probs: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp() * legal_masks.float()
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def anchor_probs(anchor_model: PolicyMLP | None, inputs: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor | None:
    if anchor_model is None:
        return None
    anchor_logits = anchor_model(inputs)
    return masked_log_softmax(anchor_logits, legal_masks).exp()


def train_epoch(
    model: PolicyMLP,
    anchor_model: PolicyMLP | None,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    *,
    entropy_weight: float,
    kl_anchor_weight: float,
    ppo_clip: float,
) -> dict:
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0
    total_anchor_kl = 0.0
    total_behavior_kl = 0.0
    behavior_kl_batches = 0
    total_samples = 0

    for (
        inputs,
        actions,
        advantages,
        _returns,
        behavior_log_probs,
        behavior_masks,
        legal_masks,
    ) in loader:
        inputs = inputs.to(device)
        actions = actions.to(device)
        advantages = advantages.to(device)
        behavior_log_probs = behavior_log_probs.to(device)
        behavior_masks = behavior_masks.to(device)
        legal_masks = legal_masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        log_probs = masked_log_softmax(logits, legal_masks)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss, behavior_kl = policy_gradient_loss(
            selected_log_probs,
            advantages,
            behavior_log_probs,
            behavior_masks,
            ppo_clip=ppo_clip,
        )
        entropy = masked_entropy(log_probs, legal_masks).mean()

        anchor_distribution = anchor_probs(anchor_model, inputs, legal_masks)
        if anchor_distribution is None or kl_anchor_weight <= 0:
            anchor_kl = torch.zeros((), device=device)
        else:
            anchor_kl = F.kl_div(log_probs, anchor_distribution, reduction="batchmean")

        loss = policy_loss - entropy_weight * entropy + kl_anchor_weight * anchor_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = inputs.shape[0]
        total_samples += batch_size
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_policy_loss += float(policy_loss.detach().cpu().item()) * batch_size
        total_entropy += float(entropy.detach().cpu().item()) * batch_size
        total_anchor_kl += float(anchor_kl.detach().cpu().item()) * batch_size
        if behavior_kl is not None:
            total_behavior_kl += float(behavior_kl.detach().cpu().item())
            behavior_kl_batches += 1

    return {
        "loss": total_loss / max(total_samples, 1),
        "policy_loss": total_policy_loss / max(total_samples, 1),
        "entropy": total_entropy / max(total_samples, 1),
        "anchor_kl": total_anchor_kl / max(total_samples, 1),
        "behavior_kl": total_behavior_kl / max(behavior_kl_batches, 1)
        if behavior_kl_batches
        else None,
        "samples": total_samples,
    }


def make_model(args: argparse.Namespace, input_dim: int, device: torch.device) -> tuple[PolicyMLP, dict, PolicyMLP | None]:
    if args.init is None:
        model = PolicyMLP(
            input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        return model, {}, None

    model, payload = load_checkpoint(args.init, device=device)
    if model.input_dim != input_dim:
        raise SystemExit(
            f"Checkpoint input_dim {model.input_dim} does not match rollout input_dim {input_dim}."
        )

    anchor_model = None
    if args.kl_anchor_weight > 0:
        anchor_model, _anchor_payload = load_checkpoint(args.init, device=device)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)

    return model, payload, anchor_model


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.max_next_pairs <= 0:
        raise SystemExit("--max-next-pairs must be at least 1.")

    set_global_seed(args.seed)
    device = choose_device(args.device)
    encoded = load_encoded_rl_dataset(
        args.rollouts,
        max_next_pairs=args.max_next_pairs,
        normalize_advantages=not args.no_normalize_advantages,
        advantage_clip=args.advantage_clip,
    )
    if encoded.step_count == 0:
        raise SystemExit("No rollout steps found.")

    model, init_payload, anchor_model = make_model(args, encoded.input_dim, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        build_tensor_dataset(encoded),
        batch_size=args.batch_size,
        shuffle=True,
    )

    print_json(
        {
            "stage": "dataset",
            "device": str(device),
            "rollouts": args.rollouts,
            "episodes": encoded.episode_count,
            "steps": encoded.step_count,
            "input_dim": encoded.input_dim,
            "return_mean": encoded.return_mean,
            "return_std": encoded.return_std,
            "init": args.init,
            "anchor": args.init if anchor_model is not None else None,
        }
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model,
            anchor_model,
            optimizer,
            loader,
            device,
            entropy_weight=args.entropy_weight,
            kl_anchor_weight=args.kl_anchor_weight,
            ppo_clip=args.ppo_clip,
        )
        epoch_metrics = {"epoch": epoch, **metrics}
        history.append(epoch_metrics)
        print_json(epoch_metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_path,
        model=model,
        metadata={
            "objective": "policy_gradient_rl",
            "kind": "policy_gradient_rl",
            "init_checkpoint": args.init,
            "init_metadata": init_payload.get("metadata", {}) if init_payload else {},
            "rollouts": args.rollouts,
            "episodes": encoded.episode_count,
            "steps": encoded.step_count,
            "return_mean": encoded.return_mean,
            "return_std": encoded.return_std,
            "max_next_pairs": args.max_next_pairs,
            "entropy_weight": args.entropy_weight,
            "kl_anchor_weight": args.kl_anchor_weight,
            "ppo_clip": args.ppo_clip,
            "advantage_clip": args.advantage_clip,
            "normalized_advantages": not args.no_normalize_advantages,
            "history": history,
            "action_keys": ACTION_KEYS,
        },
    )

    print_json(
        {
            "stage": "complete",
            "saved_to": str(output_path),
            "epochs": args.epochs,
            "steps": encoded.step_count,
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
