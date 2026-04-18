"""Run a learned policy checkpoint on one exported sample or raw state JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyTorch is required to run learned-policy inference. "
        "Install a PyTorch-supported Python environment and run "
        "`pip install -r requirements-ml.txt`."
    ) from exc

from training.action_vocab import ACTION_KEYS
from training.data import MAX_NEXT_PAIRS, encode_policy_state, normalize_policy_record
from training.model import choose_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a dataset JSON or a state JSON object",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="If --input is a JSON array, choose which sample to inspect",
    )
    parser.add_argument("--top-k", type=int, default=5)
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
    parser.add_argument("--depth", type=int, default=None, help="Override depth for raw state input")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Override beam width for raw state input",
    )
    return parser.parse_args()


def load_input_payload(path: str | Path, index: int) -> tuple[dict, dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if index < 0 or index >= len(payload):
            raise SystemExit(f"--index {index} is out of range for {len(payload)} samples.")
        return payload[index], {"source": "dataset_array", "selected_index": index, "total_samples": len(payload)}

    if not isinstance(payload, dict):
        raise SystemExit("Prediction input must be a JSON object or array.")

    return payload, {"source": "json_object"}


def build_input_vector(raw_payload: dict, *, max_next_pairs: int, depth: int, beam_width: int) -> tuple[list[float], dict]:
    if "bestActionKey" in raw_payload and "state" in raw_payload:
        record = normalize_policy_record(raw_payload, focus_weight=1.0)
        vector = encode_policy_state(
            board_rows=record.board_rows,
            current_pair=record.current_pair,
            next_queue=record.next_queue,
            depth=record.depth,
            beam_width=record.beam_width,
            max_next_pairs=max_next_pairs,
        )
        return vector, {
            "input_kind": record.kind,
            "objective": record.objective,
            "target_action": record.best_action_key,
            "teacher_candidates": record.top_candidates,
        }

    if "state" in raw_payload:
        state = raw_payload["state"]
        search = raw_payload.get("search", {})
        settings = search.get("settings", {})
        vector = encode_policy_state(
            board_rows=state["boardRows"],
            current_pair=state["currentPair"],
            next_queue=state["nextQueue"],
            depth=int(settings.get("depth", depth)),
            beam_width=int(settings.get("beamWidth", beam_width)),
            max_next_pairs=max_next_pairs,
        )
        return vector, {
            "input_kind": "state_wrapper",
            "objective": search.get("objective", "inference"),
            "target_action": None,
            "teacher_candidates": None,
        }

    if {"boardRows", "currentPair", "nextQueue"} <= raw_payload.keys():
        vector = encode_policy_state(
            board_rows=raw_payload["boardRows"],
            current_pair=raw_payload["currentPair"],
            next_queue=raw_payload["nextQueue"],
            depth=depth,
            beam_width=beam_width,
            max_next_pairs=max_next_pairs,
        )
        return vector, {
            "input_kind": "raw_state",
            "objective": "inference",
            "target_action": None,
            "teacher_candidates": None,
        }

    raise SystemExit(
        "Unsupported prediction input. Expected an exported sample, "
        "a {'state': ...} wrapper, or a raw {'boardRows', 'currentPair', 'nextQueue'} object."
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device=device)
    metadata = payload.get("metadata", {})
    max_next_pairs = args.max_next_pairs or metadata.get("max_next_pairs", MAX_NEXT_PAIRS)
    default_depth = int(metadata.get("depth", 3))
    default_beam_width = int(metadata.get("beam_width", 24))

    raw_payload, source_info = load_input_payload(args.input, args.index)
    vector, input_info = build_input_vector(
        raw_payload,
        max_next_pairs=max_next_pairs,
        depth=args.depth if args.depth is not None else default_depth,
        beam_width=args.beam_width if args.beam_width is not None else default_beam_width,
    )

    with torch.no_grad():
        logits = model(torch.tensor([vector], dtype=torch.float32, device=device))
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()

    top_k = max(1, min(args.top_k, len(ACTION_KEYS)))
    top_probabilities, top_indices = probabilities.topk(k=top_k)
    predictions = [
        {
            "rank": rank + 1,
            "actionKey": ACTION_KEYS[index],
            "probability": float(probability),
        }
        for rank, (probability, index) in enumerate(
            zip(top_probabilities.tolist(), top_indices.tolist(), strict=True)
        )
    ]

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "max_next_pairs": max_next_pairs,
                "source_info": source_info,
                "input_info": input_info,
                "predictions": predictions,
                "predicted_action": predictions[0]["actionKey"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
