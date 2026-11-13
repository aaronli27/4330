from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prepare import write_jsonl  # noqa: E402
from src.metrics import aurc, evaluate_result, summarize_metrics  # noqa: E402
from src.model_backend import FakeBackend, QwenBackend  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.schemas import parse_variant  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one controlled RAG method.")
    parser.add_argument(
        "--variants",
        type=Path,
        default=PROJECT_ROOT / "data" / "pilot_variants.jsonl",
    )
    parser.add_argument("--method", choices=["G0", "G1", "G2", "G3"], required=True)
    parser.add_argument("--condition", choices=["C0", "C1", "C2", "C3"])
    parser.add_argument("--backend", choices=["fake", "qwen"], default="fake")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-retrieval-rounds", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def read_variants(path: Path, limit: int = 0, condition: str | None = None) -> Iterable:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                variant = parse_variant(json.loads(line))
                if condition and variant.condition != condition:
                    continue
                if limit and emitted >= limit:
                    break
                emitted += 1
                yield variant


def main() -> int:
    args = parse_args()
    if args.backend == "fake":
        backend = FakeBackend()
    else:
        backend = QwenBackend(
            model_name=args.model_name,
            load_in_4bit=not args.no_4bit,
        )
    pipeline = RAGPipeline(
        backend=backend,
        top_k=args.top_k,
        max_total_tokens=args.max_total_tokens,
        max_new_tokens=args.max_new_tokens,
        max_retrieval_rounds=args.max_retrieval_rounds,
        gate_threshold=args.gate_threshold,
    )

    rows: List[dict] = []
    for variant in read_variants(args.variants, args.limit, args.condition):
        result = pipeline.run(variant, args.method)
        row = result.to_dict()
        row.update(
            {
                key: value
                for key, value in evaluate_result(variant, result).items()
                if key not in row
            }
        )
        rows.append(row)
    write_jsonl(rows, args.output)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    print(
        json.dumps(
            {
                "method": args.method,
                "backend": args.backend,
                "rows": len(rows),
                "output": str(args.output),
                "summary_by_condition": {
                    condition: {
                        **summarize_metrics(values),
                        "aurc": aurc(values),
                    }
                    for condition, values in grouped.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
