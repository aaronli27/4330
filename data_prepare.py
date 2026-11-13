from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .schemas import HotpotRecord, parse_record, validate_record


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_valid_records(
    path: Path,
) -> Tuple[List[HotpotRecord], List[Dict[str, Any]]]:
    raw_records = load_json(path)
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a JSON list, got {type(raw_records).__name__}")

    valid: List[HotpotRecord] = []
    excluded: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw in raw_records:
        record_id = str(raw.get("_id", ""))
        if record_id in seen_ids:
            excluded.append({"record_id": record_id, "errors": ["duplicate_id"]})
            continue
        seen_ids.add(record_id)
        record = parse_record(raw)
        errors = validate_record(record)
        if errors:
            excluded.append({"record_id": record_id, "errors": errors})
        else:
            valid.append(record)
    return valid, excluded


def _allocate_counts(total: int, requested: Mapping[str, int]) -> Dict[str, int]:
    requested_total = sum(requested.values())
    if requested_total > total:
        raise ValueError(
            f"Requested {requested_total} records, but only {total} are available"
        )
    if requested_total == 0:
        return {key: 0 for key in requested}
    return dict(requested)


def stratified_split(
    records: Sequence[HotpotRecord],
    requested: Mapping[str, int],
    seed: int = 20261019,
) -> Dict[str, List[HotpotRecord]]:
    """Create deterministic type-stratified splits without question leakage."""
    counts = _allocate_counts(len(records), requested)
    by_type: Dict[str, List[HotpotRecord]] = defaultdict(list)
    for record in records:
        by_type[record.question_type].append(record)

    rng = random.Random(seed)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    result: Dict[str, List[HotpotRecord]] = {key: [] for key in requested}
    remaining_by_type = {name: len(bucket) for name, bucket in by_type.items()}
    type_names = sorted(by_type)
    remaining_total = len(records)

    for split_name, split_count in counts.items():
        if split_count == 0:
            continue
        raw_targets = {
            type_name: split_count
            * remaining_by_type[type_name]
            / max(remaining_total, 1)
            for type_name in type_names
        }
        allocation = {
            type_name: min(
                remaining_by_type[type_name], int(raw_targets[type_name])
            )
            for type_name in type_names
        }
        slots_left = split_count - sum(allocation.values())
        ranked_types = sorted(
            type_names,
            key=lambda name: (
                raw_targets[name] - allocation[name],
                remaining_by_type[name],
            ),
            reverse=True,
        )
        for type_name in ranked_types:
            if slots_left <= 0:
                break
            if allocation[type_name] < remaining_by_type[type_name]:
                allocation[type_name] += 1
                slots_left -= 1
        if slots_left:
            raise AssertionError(
                f"Could not allocate {split_count} records to {split_name}"
            )

        for type_name in type_names:
            take = allocation[type_name]
            result[split_name].extend(by_type[type_name][:take])
            del by_type[type_name][:take]
            remaining_by_type[type_name] -= take
        remaining_total -= split_count

    assigned = [record.record_id for values in result.values() for record in values]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("A record was assigned to multiple splits")
    return result


def summarize_records(records: Sequence[HotpotRecord]) -> Dict[str, Any]:
    context_lengths = Counter(len(record.context) for record in records)
    question_types = Counter(record.question_type for record in records)
    levels = Counter(record.level for record in records)
    support_counts = Counter(len(record.supporting_facts) for record in records)
    return {
        "records": len(records),
        "unique_ids": len({record.record_id for record in records}),
        "question_types": dict(sorted(question_types.items())),
        "levels": dict(sorted(levels.items())),
        "context_lengths": dict(sorted(context_lengths.items())),
        "supporting_fact_counts": dict(sorted(support_counts.items())),
    }


def save_split_ids(splits: Mapping[str, Sequence[HotpotRecord]], path: Path) -> None:
    payload = {
        split_name: [record.record_id for record in records]
        for split_name, records in splits.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(items: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

