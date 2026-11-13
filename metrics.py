from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set

from .pipeline import PipelineResult
from .schemas import Variant


ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _f1_sets(predicted: Set[str], gold: Set[str]) -> float:
    if not predicted or not gold:
        return float(predicted == gold)
    common = len(predicted & gold)
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def evaluate_result(variant: Variant, result: PipelineResult) -> Dict[str, Any]:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in variant.context}
    gold_support_titles = {title for title, _ in variant.supporting_facts}
    present_support_titles = {
        paragraph.title
        for paragraph in variant.context
        if paragraph.title in gold_support_titles
    }
    predicted_ids = {
        evidence_id for evidence_id in result.evidence_ids if evidence_id in paragraph_by_id
    }
    predicted_titles = {
        paragraph_by_id[evidence_id].title for evidence_id in predicted_ids
    }
    title_overlap = predicted_titles & gold_support_titles
    answer_is_correct = exact_match(result.answer, variant.answer)
    answer_f1 = token_f1(result.answer, variant.answer)
    condition_is_insufficient = variant.condition in {"C1", "C3"}
    return {
        "record_id": variant.record_id,
        "condition": variant.condition,
        "method": result.method,
        "status": result.status,
        "coverage": int(result.status == "answer"),
        "answer_em": answer_is_correct if result.status == "answer" else 0.0,
        "answer_f1": answer_f1 if result.status == "answer" else 0.0,
        "false_abstention": int(
            variant.condition == "C0" and result.status == "abstain"
        ),
        "unsafe_answer": int(
            condition_is_insufficient and result.status == "answer"
        ),
        "abstention_correct": int(
            condition_is_insufficient and result.status == "abstain"
        ),
        "evidence_precision": (
            len(title_overlap) / len(predicted_titles)
            if predicted_titles
            else 0.0
        ),
        "evidence_recall": (
            len(title_overlap) / len(gold_support_titles)
            if gold_support_titles
            else 0.0
        ),
        "evidence_f1": _f1_sets(predicted_titles, gold_support_titles),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
        "latency_ms": result.latency_ms,
        "model_calls": result.model_calls,
        "parse_errors": result.parse_errors,
        "selection_score": result.selection_score,
        "gold_support_count": len(gold_support_titles),
        "present_support_count": len(present_support_titles),
        "missing_support_count": len(gold_support_titles - present_support_titles),
        "missing_support_titles": sorted(gold_support_titles - present_support_titles),
        "selection_loss": (
            float(result.status == "answer")
            if variant.condition in {"C1", "C3"}
            else 1.0 - (
                answer_f1 if result.status == "answer" else 0.0
            )
        ),
    }


def summarize_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"rows": 0}
    numeric_fields = [
        "answer_em",
        "answer_f1",
        "false_abstention",
        "unsafe_answer",
        "abstention_correct",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "model_calls",
        "parse_errors",
        "selection_score",
        "selection_loss",
        "coverage",
        "gold_support_count",
        "present_support_count",
        "missing_support_count",
    ]
    summary: Dict[str, Any] = {"rows": len(rows)}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows]
        summary[field] = sum(values) / len(values)
    return summary


def selective_curve(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Return risk-coverage points ranked by the model selection score."""
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("selection_score", 0.0)),
            str(row.get("record_id", "")),
        ),
    )
    points: List[Dict[str, float]] = []
    cumulative_loss = 0.0
    accepted = 0
    total = len(ranked)
    for row in ranked:
        if str(row.get("status", "answer")) != "answer":
            continue
        accepted += 1
        cumulative_loss += float(row.get("selection_loss", 1.0))
        points.append(
            {
                "coverage": accepted / total,
                "risk": cumulative_loss / accepted,
            }
        )
    return points


def aurc(rows: Sequence[Dict[str, Any]]) -> float:
    """Compute the discrete area under the risk-coverage curve."""
    curve = selective_curve(rows)
    if not curve:
        return 0.0
    return sum(point["risk"] for point in curve) / len(curve)

