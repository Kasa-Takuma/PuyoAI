"""Train, evaluate, export, and register multiple learned policies in one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))


SUITE_CONFIG_PATH = Path(__file__).with_name("model_suite.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", required=True, help="Path to Export Slim JSON")
    parser.add_argument(
        "--focus",
        default=None,
        help="Optional path to Export 6+ Focus JSON",
    )
    parser.add_argument(
        "--suite",
        default=str(SUITE_CONFIG_PATH),
        help="Path to a suite JSON config",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for generated checkpoints, web models, and manifest",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional explicit manifest path. Defaults to <models-dir>/manifest.json",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the sub-commands",
    )
    parser.add_argument(
        "--model-ids",
        nargs="*",
        default=None,
        help="Optional subset of model ids from the suite config",
    )
    return parser.parse_args()


def load_suite_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("models"), list):
        raise SystemExit("Suite config must contain a top-level 'models' array.")
    return config


def kebab_to_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def append_optional_flag(command: list[str], flag_name: str, value) -> None:
    if value is None:
        return
    command.extend([kebab_to_flag(flag_name), str(value)])


def run_json_command(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    stdout = completed.stdout.strip()
    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        lines = [line for line in stdout.splitlines() if line.strip()]
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Expected JSON output from {' '.join(command)}") from exc


def build_train_command(
    python_executable: str,
    *,
    slim_path: str,
    focus_path: str | None,
    checkpoint_path: Path,
    settings: dict,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "training.train_policy",
        "--slim",
        slim_path,
        "--output",
        str(checkpoint_path),
    ]
    if focus_path:
        command.extend(["--focus", focus_path])

    for key, value in settings.items():
        append_optional_flag(command, key, value)

    return command


def build_evaluate_command(
    python_executable: str,
    *,
    slim_path: str,
    focus_path: str | None,
    checkpoint_path: Path,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "training.evaluate_policy",
        "--checkpoint",
        str(checkpoint_path),
        "--slim",
        slim_path,
    ]
    if focus_path:
        command.extend(["--focus", focus_path])
    return command


def build_export_command(
    python_executable: str,
    *,
    checkpoint_path: Path,
    web_output_path: Path,
    name: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "training.export_web_policy",
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(web_output_path),
        "--name",
        name,
    ]


def main() -> None:
    args = parse_args()
    suite_config = load_suite_config(args.suite)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = Path(args.manifest_output) if args.manifest_output else models_dir / "manifest.json"

    defaults = dict(suite_config.get("defaults", {}))
    requested_ids = set(args.model_ids or [])

    selected_models = [
        model
        for model in suite_config["models"]
        if not requested_ids or model.get("id") in requested_ids
    ]
    if not selected_models:
        raise SystemExit("No suite models matched the requested ids.")

    manifest_models = []

    for model in selected_models:
        model_id = model["id"]
        label = model.get("label", model_id)
        description = model.get("description", "")
        train_settings = {**defaults, **model.get("train", {})}
        checkpoint_path = models_dir / f"{model_id}.pt"
        web_output_path = models_dir / f"{model_id}.web.json"

        print(f"== Training {model_id} ==")
        run_json_command(
            build_train_command(
                args.python,
                slim_path=args.slim,
                focus_path=args.focus,
                checkpoint_path=checkpoint_path,
                settings=train_settings,
            )
        )

        print(f"== Evaluating {model_id} ==")
        evaluation = run_json_command(
            build_evaluate_command(
                args.python,
                slim_path=args.slim,
                focus_path=args.focus,
                checkpoint_path=checkpoint_path,
            )
        )

        print(f"== Exporting {model_id} ==")
        run_json_command(
            build_export_command(
                args.python,
                checkpoint_path=checkpoint_path,
                web_output_path=web_output_path,
                name=label,
            )
        )

        manifest_models.append(
            {
                "id": model_id,
                "label": label,
                "description": description,
                "path": f"./{web_output_path.name}",
                "checkpointPath": f"./{checkpoint_path.name}",
                "train": train_settings,
                "evaluation": evaluation.get("metrics", {}),
            }
        )

    default_model_id = suite_config.get("defaultModelId")
    if not any(model["id"] == default_model_id for model in manifest_models):
        default_model_id = manifest_models[0]["id"]

    manifest = {
        "format": "puyoai-model-manifest-v1",
        "defaultModelId": default_model_id,
        "models": manifest_models,
    }
    with manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "manifest": str(manifest_output),
                "model_count": len(manifest_models),
                "defaultModelId": default_model_id,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
