from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all G0-G3 methods on one protocol split."
    )
    parser.add_argument(
        "--split",
        choices=["pilot", "calibration", "standard_test", "unseen_test"],
        default="pilot",
    )
    parser.add_argument("--backend", choices=["fake", "qwen"], default="fake")
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = PROJECT_ROOT / "data" / f"{args.split}_variants.jsonl"
    for method in ("G0", "G1", "G2", "G3"):
        output = PROJECT_ROOT / "results" / f"{args.backend}_{method}_{args.split}.jsonl"
        command = [
            PYTHON,
            str(PROJECT_ROOT / "tools" / "run_experiment.py"),
            "--variants",
            str(variants),
            "--method",
            method,
            "--backend",
            args.backend,
            "--output",
            str(output),
            "--max-total-tokens",
            str(args.max_total_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.backend == "qwen":
            command.extend(["--model-name", args.model_name])
            if args.no_4bit:
                command.append("--no-4bit")
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
