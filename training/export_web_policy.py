"""Export a trained PyTorch policy checkpoint to a browser-friendly JSON asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyTorch is required to export the learned policy. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path, for example models/policy_mlp.web.json",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places to keep for exported weights",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional model name shown in the web UI",
    )
    return parser.parse_args()


def round_tensor_values(tensor, precision: int) -> list[float]:
    return [round(float(value), precision) for value in tensor.reshape(-1).tolist()]


def export_layers(model, precision: int) -> list[dict]:
    linear_layers = [module for module in model.network if isinstance(module, nn.Linear)]
    exported = []
    for index, layer in enumerate(linear_layers):
        exported.append(
            {
                "index": index,
                "inputDim": layer.in_features,
                "outputDim": layer.out_features,
                "activation": "linear" if index == len(linear_layers) - 1 else "gelu",
                "weights": round_tensor_values(layer.weight.detach().cpu(), precision),
                "bias": round_tensor_values(layer.bias.detach().cpu(), precision),
            }
        )
    return exported


def main() -> None:
    args = parse_args()
    model, payload = load_checkpoint(args.checkpoint, device="cpu")
    metadata = payload.get("metadata", {})

    export_payload = {
        "format": "puyoai-policy-web-v1",
        "name": args.name or Path(args.checkpoint).stem,
        "actionKeys": payload.get("action_keys", []),
        "maxNextPairs": metadata.get("max_next_pairs", 5),
        "objective": metadata.get("objective", "learned_policy_mlp"),
        "modelConfig": payload.get("model_config", {}),
        "metadata": metadata,
        "layers": export_layers(model, args.precision),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(export_payload, handle, ensure_ascii=False, separators=(",", ":"))

    print(
        json.dumps(
            {
                "exported_to": str(output_path),
                "layer_count": len(export_payload["layers"]),
                "action_count": len(export_payload["actionKeys"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
