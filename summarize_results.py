from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import aurc, summarize_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment JSONL files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--pattern",
        default="qwen_*_pilot.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "qwen_pilot_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary: Dict[str, Any] = {}
    for path in sorted(args.results_dir.glob(args.pattern)):
        rows: List[Dict[str, Any]] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_condition = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        method = rows[0]["method"] if rows else path.stem
        summary[method] = {
            condition: {
                **summarize_metrics(values),
                "aurc": aurc(values),
            }
            for condition, values in sorted(by_condition.items())
        }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"methods": sorted(summary), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
