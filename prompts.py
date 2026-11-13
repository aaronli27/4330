from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .schemas import Paragraph


def format_evidence(paragraphs: Sequence[Paragraph]) -> str:
    chunks = []
    for paragraph in paragraphs:
        chunks.append(f"[{paragraph.paragraph_id}] {paragraph.title}\n{paragraph.text}")
    return "\n\n".join(chunks)


def build_decomposition_prompt(question: str) -> str:
    return (
        "TASK: DECOMPOSE\n"
        "Break the multi-hop question into at most two answerable subquestions. "
        "Return JSON only: {\"subquestions\": [\"...\"]}.\n"
        f"Question: {question}"
    )


def build_global_gate_prompt(question: str, evidence: Sequence[Paragraph]) -> str:
    return (
        "TASK: GLOBAL_GATE\n"
        "Judge whether the supplied evidence is sufficient to answer the question "
        "without relying on unsupported memory. Return JSON only with fields "
        "sufficient (boolean), confidence (number from 0 to 1), and reason (string).\n"
        f"Question: {question}\nEvidence:\n{format_evidence(evidence)}"
    )


def build_local_gate_prompt(
    question: str,
    subquestions: Sequence[str],
    evidence: Sequence[Paragraph],
) -> str:
    subquestion_text = "\n".join(
        f"{index + 1}. {subquestion}"
        for index, subquestion in enumerate(subquestions)
    )
    return (
        "TASK: LOCAL_GATE\n"
        "Judge evidence coverage for each subquestion. Return JSON only with "
        "sufficient (boolean), confidence (number from 0 to 1), "
        "covered_subquestions (array of booleans), and reason (string). "
        "The overall result is sufficient only when every "
        "required subquestion is covered.\n"
        f"Question: {question}\nSubquestions:\n{subquestion_text}\n"
        f"Evidence:\n{format_evidence(evidence)}"
    )


def build_answer_prompt(
    question: str,
    evidence: Sequence[Paragraph],
) -> str:
    return (
        "TASK: ANSWER\n"
        "Answer only from the supplied evidence. Return JSON only with fields "
        "status (answer or abstain), answer (string), and evidence_ids (array "
        "of paragraph IDs). Use abstain when the evidence is insufficient.\n"
        f"Question: {question}\nEvidence:\n{format_evidence(evidence)}"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a JSON object")


def parse_decomposition(text: str, fallback_question: str) -> List[str]:
    try:
        value = _extract_json(text)
        raw = value.get("subquestions", [])
        if isinstance(raw, list):
            cleaned = [str(item).strip() for item in raw if str(item).strip()]
            if cleaned:
                return cleaned[:2]
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return [fallback_question]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "covered"}
    return False


def parse_gate(text: str) -> Dict[str, Any]:
    try:
        value = _extract_json(text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return {
            "sufficient": False,
            "confidence": 0.0,
            "covered_subquestions": [],
            "parse_error": True,
        }
    sufficient = _as_bool(value.get("sufficient", False))
    covered = value.get("covered_subquestions", [])
    if not isinstance(covered, list):
        covered = []
    raw_confidence = value.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "sufficient": sufficient,
        "confidence": max(0.0, min(1.0, confidence)),
        "covered_subquestions": [_as_bool(item) for item in covered],
        "reason": str(value.get("reason", "")),
        "parse_error": False,
    }


def parse_answer(text: str, fallback_evidence_ids: Sequence[str]) -> Dict[str, Any]:
    try:
        value = _extract_json(text)
        status = str(value.get("status", "answer")).lower()
        if status not in {"answer", "abstain"}:
            status = "answer"
        raw_answer = value.get("answer", "")
        answer = "" if raw_answer is None else str(raw_answer).strip()
        evidence_ids = value.get("evidence_ids", list(fallback_evidence_ids))
        if not isinstance(evidence_ids, list):
            evidence_ids = list(fallback_evidence_ids)
        return {
            "status": status,
            "answer": answer,
            "evidence_ids": [str(item) for item in evidence_ids],
            "parse_error": False,
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {
            "status": "answer",
            "answer": text.strip(),
            "evidence_ids": list(fallback_evidence_ids),
            "parse_error": True,
        }

