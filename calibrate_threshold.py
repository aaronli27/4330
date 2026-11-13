from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a gate threshold using calibration results."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--min-c0-coverage", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "calibrated_threshold.json",
    )
    return parser.parse_args()


def mean(rows: List[Dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows) / max(len(rows), 1)


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = sorted(
        {
            round(float(row.get("selection_score", 0.0)), 4)
            for row in rows
        }
        | {0.0, 0.5, 0.7, 0.8, 0.9, 1.0}
    )
    evaluated = []
    for threshold in candidates:
        accepted = [
            row
            for row in rows
            if row.get("status") == "answer"
            and float(row.get("selection_score", 0.0)) >= threshold
        ]
        c0 = [row for row in accepted if row["condition"] == "C0"]
        missing = [
            row
            for row in accepted
            if row["condition"] in {"C1", "C3"}
        ]
        c0_coverage = len(c0) / max(
            sum(row["condition"] == "C0" for row in rows), 1
        )
        unsafe = mean(missing, "unsafe_answer") if missing else 0.0
        c0_f1 = mean(c0, "answer_f1") if c0 else 0.0
        evaluated.append(
            {
                "threshold": threshold,
                "c0_coverage": c0_coverage,
                "missing_unsafe_rate": unsafe,
                "c0_answer_f1": c0_f1,
            }
        )

    eligible = [
        item
        for item in evaluated
        if item["c0_coverage"] >= args.min_c0_coverage
    ]
    if not eligible:
        raise SystemExit("No threshold satisfies the minimum C0 coverage")
    selected = min(
        eligible,
        key=lambda item: (
            item["missing_unsafe_rate"],
            -item["c0_answer_f1"],
            item["threshold"],
        ),
    )
    payload = {
        "source_results": str(args.results),
        "minimum_c0_coverage": args.min_c0_coverage,
        "selected": selected,
        "candidates": evaluated,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
