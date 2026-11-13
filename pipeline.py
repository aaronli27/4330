from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .bm25_retriever import BM25, RetrievalHit
from .model_backend import GenerationResult, ModelBackend
from .prompts import (
    build_answer_prompt,
    build_decomposition_prompt,
    build_global_gate_prompt,
    build_local_gate_prompt,
    parse_answer,
    parse_decomposition,
    parse_gate,
)
from .schemas import Paragraph, Variant


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TokenBudget:
    limit: int
    used: int = 0

    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self, estimated_input_tokens: int, max_new_tokens: int) -> int:
        available = self.remaining() - estimated_input_tokens
        if available <= 0:
            raise BudgetExceeded("No token budget remains for a model call")
        return min(max_new_tokens, available)

    def add(self, result: GenerationResult) -> None:
        if self.used + result.total_tokens > self.limit:
            raise BudgetExceeded(
                "Model call exceeded the shared token budget after exact counting"
            )
        self.used += result.total_tokens


@dataclass
class PipelineResult:
    record_id: str
    condition: str
    method: str
    status: str
    answer: str
    evidence_ids: List[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_calls: int
    selection_score: float
    trace: List[Dict[str, Any]] = field(default_factory=list)
    parse_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "condition": self.condition,
            "method": self.method,
            "status": self.status,
            "answer": self.answer,
            "evidence_ids": self.evidence_ids,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "latency_ms": self.latency_ms,
            "model_calls": self.model_calls,
            "selection_score": self.selection_score,
            "trace": self.trace,
            "parse_errors": self.parse_errors,
        }


class CandidatePool:
    def __init__(self, paragraphs: Sequence[Paragraph]) -> None:
        self.paragraphs = tuple(paragraphs)
        self.by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
        self.index = BM25(
            [(paragraph.paragraph_id, paragraph.text) for paragraph in paragraphs]
        )

    def retrieve(self, query: str, top_k: int) -> List[Paragraph]:
        hits = self.index.search(query, top_k=top_k)
        return [self.by_id[hit.doc_id] for hit in hits]

    @staticmethod
    def merge(paragraph_groups: Sequence[Sequence[Paragraph]]) -> List[Paragraph]:
        selected: Dict[str, Paragraph] = {}
        for group in paragraph_groups:
            for paragraph in group:
                selected.setdefault(paragraph.paragraph_id, paragraph)
        return list(selected.values())


class RAGPipeline:
    def __init__(
        self,
        backend: ModelBackend,
        top_k: int = 5,
        max_total_tokens: int = 2048,
        max_new_tokens: int = 256,
        max_retrieval_rounds: int = 2,
        gate_threshold: float = 0.5,
    ) -> None:
        self.backend = backend
        self.top_k = top_k
        self.max_total_tokens = max_total_tokens
        self.max_new_tokens = max_new_tokens
        self.max_retrieval_rounds = max_retrieval_rounds
        self.gate_threshold = max(0.0, min(1.0, gate_threshold))

    def _call(
        self,
        prompt: str,
        budget: TokenBudget,
        trace: List[Dict[str, Any]],
        stage: str,
    ) -> GenerationResult:
        estimated_input = self.backend.count_tokens(prompt)
        allowed_new = budget.reserve(estimated_input, self.max_new_tokens)
        result = self.backend.generate(prompt, max_new_tokens=allowed_new)
        budget.add(result)
        trace.append(
            {
                "stage": stage,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
                "raw_output": result.text,
            }
        )
        return result

    def _finish(
        self,
        variant: Variant,
        method: str,
        parsed: Dict[str, Any],
        budget: TokenBudget,
        trace: List[Dict[str, Any]],
        parse_errors: int,
        selection_score: float = 0.0,
    ) -> PipelineResult:
        return PipelineResult(
            record_id=variant.record_id,
            condition=variant.condition,
            method=method,
            status=parsed.get("status", "abstain"),
            answer=parsed.get("answer", ""),
            evidence_ids=list(parsed.get("evidence_ids", [])),
            input_tokens=sum(item["input_tokens"] for item in trace),
            output_tokens=sum(item["output_tokens"] for item in trace),
            latency_ms=sum(item["latency_ms"] for item in trace),
            model_calls=len(trace),
            selection_score=selection_score,
            trace=trace,
            parse_errors=parse_errors,
        )

    def _answer(
        self,
        variant: Variant,
        method: str,
        evidence: Sequence[Paragraph],
        budget: TokenBudget,
        trace: List[Dict[str, Any]],
        parse_errors: int,
        selection_score: float = 1.0,
    ) -> PipelineResult:
        try:
            result = self._call(
                build_answer_prompt(variant.question, evidence),
                budget,
                trace,
                "answer",
            )
        except BudgetExceeded:
            return self._finish(
                variant,
                method,
                {"status": "abstain", "answer": "", "evidence_ids": []},
                budget,
                trace,
                parse_errors,
            )
        parsed = parse_answer(
            result.text,
            [paragraph.paragraph_id for paragraph in evidence],
        )
        parse_errors += int(parsed.get("parse_error", False))
        return self._finish(
            variant, method, parsed, budget, trace, parse_errors, selection_score
        )

    def run(self, variant: Variant, method: str) -> PipelineResult:
        if method not in {"G0", "G1", "G2", "G3"}:
            raise ValueError(f"Unknown method: {method}")
        pool = CandidatePool(variant.context)
        budget = TokenBudget(self.max_total_tokens)
        trace: List[Dict[str, Any]] = []
        parse_errors = 0

        if method == "G0":
            evidence = pool.retrieve(variant.question, self.top_k)
            return self._answer(
                variant, method, evidence, budget, trace, parse_errors, 1.0
            )

        try:
            decomposition_result = self._call(
                build_decomposition_prompt(variant.question),
                budget,
                trace,
                "decompose",
            )
            subquestions = parse_decomposition(
                decomposition_result.text, variant.question
            )
        except BudgetExceeded:
            return self._finish(
                variant,
                method,
                {"status": "abstain", "answer": "", "evidence_ids": []},
                budget,
                trace,
                parse_errors,
            )

        if method == "G1":
            evidence = pool.merge(
                [pool.retrieve(subquestion, self.top_k) for subquestion in subquestions]
            )
            return self._answer(
                variant, method, evidence, budget, trace, parse_errors, 1.0
            )

        if method == "G2":
            evidence = pool.retrieve(variant.question, self.top_k)
            try:
                gate_result = self._call(
                    build_global_gate_prompt(variant.question, evidence),
                    budget,
                    trace,
                    "global_gate_round_1",
                )
                gate = parse_gate(gate_result.text)
                parse_errors += int(gate.get("parse_error", False))
                accepted = gate["sufficient"] and gate["confidence"] >= self.gate_threshold
                if not accepted:
                    if self.max_retrieval_rounds < 2:
                        return self._finish(
                            variant,
                            method,
                            {"status": "abstain", "answer": "", "evidence_ids": []},
                            budget,
                            trace,
                            parse_errors,
                            gate["confidence"],
                        )
                    evidence = pool.retrieve(variant.question, min(len(pool.paragraphs), self.top_k * 2))
                    gate_result = self._call(
                        build_global_gate_prompt(variant.question, evidence),
                        budget,
                        trace,
                        "global_gate_round_2",
                    )
                    gate = parse_gate(gate_result.text)
                    parse_errors += int(gate.get("parse_error", False))
                accepted = gate["sufficient"] and gate["confidence"] >= self.gate_threshold
                if not accepted:
                    return self._finish(
                        variant,
                        method,
                        {"status": "abstain", "answer": "", "evidence_ids": []},
                        budget,
                        trace,
                        parse_errors,
                        gate["confidence"],
                    )
                return self._answer(
                    variant,
                    method,
                    evidence,
                    budget,
                    trace,
                    parse_errors,
                    gate["confidence"],
                )
            except BudgetExceeded:
                return self._finish(
                    variant,
                    method,
                    {"status": "abstain", "answer": "", "evidence_ids": []},
                    budget,
                    trace,
                    parse_errors,
                )

        evidence_groups = [pool.retrieve(subquestion, self.top_k) for subquestion in subquestions]
        evidence = pool.merge(evidence_groups)
        try:
            gate_result = self._call(
                build_local_gate_prompt(variant.question, subquestions, evidence),
                budget,
                trace,
                "local_gate_round_1",
            )
            gate = parse_gate(gate_result.text)
            parse_errors += int(gate.get("parse_error", False))
            accepted = gate["sufficient"] and gate["confidence"] >= self.gate_threshold
            if not accepted:
                if self.max_retrieval_rounds < 2:
                    return self._finish(
                        variant,
                        method,
                        {"status": "abstain", "answer": "", "evidence_ids": []},
                        budget,
                        trace,
                        parse_errors,
                        gate["confidence"],
                    )
                missing = [
                    subquestion
                    for index, subquestion in enumerate(subquestions)
                    if index >= len(gate["covered_subquestions"])
                    or not gate["covered_subquestions"][index]
                ]
                if not missing:
                    missing = list(subquestions)
                evidence = pool.merge(
                    [
                        evidence,
                        *[
                            pool.retrieve(subquestion, min(len(pool.paragraphs), self.top_k * 2))
                            for subquestion in missing
                        ],
                    ]
                )
                gate_result = self._call(
                    build_local_gate_prompt(variant.question, subquestions, evidence),
                    budget,
                    trace,
                    "local_gate_round_2",
                )
                gate = parse_gate(gate_result.text)
                parse_errors += int(gate.get("parse_error", False))
            accepted = gate["sufficient"] and gate["confidence"] >= self.gate_threshold
            if not accepted:
                return self._finish(
                    variant,
                    method,
                    {"status": "abstain", "answer": "", "evidence_ids": []},
                    budget,
                    trace,
                    parse_errors,
                    gate["confidence"],
                )
            return self._answer(
                variant,
                method,
                evidence,
                budget,
                trace,
                parse_errors,
                gate["confidence"],
            )
        except BudgetExceeded:
            return self._finish(
                variant,
                method,
                {"status": "abstain", "answer": "", "evidence_ids": []},
                budget,
                trace,
                parse_errors,
            )

