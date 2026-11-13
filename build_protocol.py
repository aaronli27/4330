from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prepare import (  # noqa: E402
    load_valid_records,
    save_json,
    save_split_ids,
    stratified_split,
    summarize_records,
    write_jsonl,
)
from src.perturbations import build_distractor_pool, build_variants  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the controlled RAG protocol.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"D:\项目\大作业\datasets\hotpotqa\hotpot_dev_distractor_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--pilot", type=int, default=50)
    parser.add_argument("--calibration", type=int, default=100)
    parser.add_argument("--standard-test", type=int, default=100)
    parser.add_argument("--unseen-test", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, excluded = load_valid_records(args.dataset)
    requested = {
        "pilot": args.pilot,
        "calibration": args.calibration,
        "standard_test": args.standard_test,
        "unseen_test": args.unseen_test,
    }
    splits = stratified_split(records, requested, seed=args.seed)
    # Use calibration-only paragraphs as the shared hard-distractor source.
    # This keeps the standard and unseen test questions out of the donor pool.
    hard_pool = build_distractor_pool(splits["calibration"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_split_ids(splits, args.output_dir / "split_ids.json")
    save_json(
        {
            "seed": args.seed,
            "dataset": str(args.dataset),
            "valid_records": len(records),
            "excluded_records": len(excluded),
            "summary": summarize_records(records),
            "splits": {name: len(values) for name, values in splits.items()},
            "hard_distractor_pool": len(hard_pool),
            "hard_distractor_pool_source_split": "calibration",
            "excluded": excluded,
        },
        args.output_dir / "protocol_manifest.json",
    )

    variant_counts = {}
    for split_name, split_records in splits.items():
        variants = build_variants(
            split_records,
            hard_pool=hard_pool,
            seed=args.seed + list(splits).index(split_name) * 100000,
        )
        write_jsonl(
            (variant.to_dict() for variant in variants),
            args.output_dir / f"{split_name}_variants.jsonl",
        )
        variant_counts[split_name] = len(variants)
    variant_counts["pilot"] = variant_counts.get("pilot", 0)
    print(
        json.dumps(
            {
                "valid_records": len(records),
                "excluded_records": len(excluded),
                "splits": {name: len(values) for name, values in splits.items()},
                "variant_counts": variant_counts,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

