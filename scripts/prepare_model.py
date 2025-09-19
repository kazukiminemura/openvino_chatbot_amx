#!/usr/bin/env python
"""Utilities to export and quantize a causal language model for OpenVINO INT8/AMX inference."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Hugging Face causal language model to OpenVINO IR and apply INT8 weight "
            "compression so it can leverage Intel AMX on supported CPUs."
        )
    )
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model repository on Hugging Face Hub to export."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/int8_chatbot_model"),
        help="Directory where the exported OpenVINO model should be stored."
    )
    parser.add_argument(
        "--precision",
        choices=["int8", "int4", "bf16"],
        default="int8",
        help="Weight format to request from optimum-cli export."
    )
    parser.add_argument(
        "--task",
        default="text-generation-with-past",
        help="Task argument forwarded to optimum-cli (use text-generation for simple GPT2-like models)."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust-remote-code flag when exporting community models that require custom code."
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
        help="Optional path for HF_HOME / model cache to avoid re-downloading."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the export step if the output directory already contains model.xml."
    )
    return parser.parse_args()


def ensure_optimum_cli() -> str:
    cli_path = shutil.which("optimum-cli")
    if cli_path is None:
        raise RuntimeError(
            "optimum-cli not found. Install it with `pip install optimum-intel[openvino]` before running this script."
        )
    return cli_path


def export_with_optimum(cli_path: str, args: argparse.Namespace) -> None:
    cmd = [
        cli_path,
        "export",
        "openvino",
        "--model",
        args.model_id,
    ]

    if args.task:
        cmd.extend(["--task", args.task])

    if args.precision.lower() in {"int8", "int4"}:
        cmd.extend(["--weight-format", args.precision.lower()])
    elif args.precision.lower() == "bf16":
        cmd.extend(["--precision", "bf16"])

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    cmd.append(str(args.output_dir))

    env = os.environ.copy()
    if args.model_cache is not None:
        env["HF_HOME"] = str(args.model_cache)

    print("[optimum-cli]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def write_runtime_hints(output_dir: Path, precision: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "ov_cache"
    cache_dir.mkdir(exist_ok=True)

    ov_cpu_precision = {
        "int8": "int8",
        "int4": "int8",
        "bf16": "bf16",
    }[precision.lower()]

    ov_config = {
        "INFERENCE_PRECISION_HINT": ov_cpu_precision,
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "CACHE_DIR": str(cache_dir.resolve()),
        "ENABLE_CPU_PINNING": "YES",
    }

    config_path = output_dir / "ov_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(ov_config, fh, indent=2)
    print(f"Saved OpenVINO runtime hints to {config_path}")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.skip_existing:
        ov_xml = args.output_dir / "openvino_model.xml"
        fallback_xml = args.output_dir / "model.xml"
        if ov_xml.exists() or fallback_xml.exists():
            print("Existing OpenVINO model detected; skipping export as requested.")
            write_runtime_hints(args.output_dir, args.precision)
            return

    optimum_cli = ensure_optimum_cli()
    export_with_optimum(optimum_cli, args)
    write_runtime_hints(args.output_dir, args.precision)
    print("Model preparation complete. You can now run scripts/chatbot.py with this export directory.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"optimum-cli returned non-zero exit status {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
