from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25  # noqa: E402
from src.data_prepare import load_valid_records, write_jsonl  # noqa: E402
from src.perturbations import build_distractor_pool, build_variants  # noqa: E402
from src.pipeline import CandidatePool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate closed-pool BM25 recall.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"D:\项目\大作业\datasets\hotpotqa\hotpot_dev_distractor_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_retrieval_smoke.jsonl",
    )
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, _ = load_valid_records(args.dataset)
    records = records[: args.questions]
    hard_pool = build_distractor_pool(records, max_paragraphs=5000)
    variants = build_variants(records, hard_pool, seed=20261019)
    rows = []
    for variant in variants:
        pool = CandidatePool(variant.context)
        hits = pool.retrieve(variant.question, args.top_k)
        hit_ids = {paragraph.paragraph_id for paragraph in hits}
        gold_ids = {
            paragraph.paragraph_id
            for paragraph in variant.context
            if paragraph.title in {title for title, _ in variant.supporting_facts}
        }
        rows.append(
            {
                "record_id": variant.record_id,
                "condition": variant.condition,
                "top_k": args.top_k,
                "support_recall": (
                    len(hit_ids & gold_ids) / len(gold_ids) if gold_ids else 0.0
                ),
                "all_support_retrieved": int(gold_ids <= hit_ids),
                "retrieved_ids": [paragraph.paragraph_id for paragraph in hits],
            }
        )
    write_jsonl(rows, args.output)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = {
        condition: {
            "rows": len(values),
            "mean_support_recall": sum(
                value["support_recall"] for value in values
            )
            / len(values),
            "all_support_retrieved_rate": sum(
                value["all_support_retrieved"] for value in values
            )
            / len(values),
        }
        for condition, values in grouped.items()
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(summary_path),
                "by_condition": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

