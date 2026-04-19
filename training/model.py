"""PyTorch model definitions for learned PuyoAI policies and values."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from training.action_vocab import ACTION_KEYS


class PolicyMLP(nn.Module):
    """A compact MLP that imitates the search policy from encoded states."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.12,
        output_dim: int = len(ACTION_KEYS),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ValueMLP(nn.Module):
    """A compact MLP that estimates future chain value from encoded states."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        output_dim: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def choose_device(requested: str = "auto") -> torch.device:
    """Choose the best available torch device."""

    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    output_path: str | Path,
    *,
    model: PolicyMLP,
    metadata: dict,
) -> None:
    """Persist a model checkpoint with config and metadata."""

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "output_dim": model.output_dim,
        },
        "action_keys": ACTION_KEYS,
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def save_value_checkpoint(
    output_path: str | Path,
    *,
    model: ValueMLP,
    metadata: dict,
) -> None:
    """Persist a value model checkpoint with config and metadata."""

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "output_dim": model.output_dim,
        },
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyMLP, dict]:
    """Load a saved policy checkpoint and return the model plus payload."""

    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["model_config"]
    model = PolicyMLP(
        config["input_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_value_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ValueMLP, dict]:
    """Load a saved value checkpoint and return the model plus payload."""

    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["model_config"]
    model = ValueMLP(
        config["input_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
