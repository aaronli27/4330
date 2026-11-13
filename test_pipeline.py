from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_backend import FakeBackend, GenerationResult  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.prompts import parse_gate  # noqa: E402
from src.perturbations import build_distractor_pool, build_variants  # noqa: E402
from src.data_prepare import load_valid_records  # noqa: E402


DATASET = Path(
    r"D:\项目\大作业\datasets\hotpotqa\hotpot_dev_distractor_v1.json"
)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records, _ = load_valid_records(DATASET)
        pool = build_distractor_pool(records, max_paragraphs=500)
        cls.variants = build_variants(records[:1], pool, seed=20261019)

    def test_all_methods_emit_structured_result(self) -> None:
        pipeline = RAGPipeline(
            backend=FakeBackend(),
            top_k=3,
            max_total_tokens=1024,
            max_new_tokens=128,
        )
        for method in ("G0", "G1", "G2", "G3"):
            result = pipeline.run(self.variants[0], method)
            self.assertEqual(result.method, method)
            self.assertIn(result.status, {"answer", "abstain"})
            self.assertGreaterEqual(result.model_calls, 1)
            self.assertGreater(result.input_tokens, 0)
            self.assertGreaterEqual(result.selection_score, 0.0)
            self.assertLessEqual(result.selection_score, 1.0)

    def test_budget_can_force_abstention(self) -> None:
        pipeline = RAGPipeline(
            backend=FakeBackend(),
            top_k=3,
            max_total_tokens=5,
            max_new_tokens=3,
        )
        result = pipeline.run(self.variants[0], "G3")
        self.assertEqual(result.status, "abstain")

    def test_gate_parser_handles_string_booleans(self) -> None:
        parsed = parse_gate(
            '{"sufficient": "false", "covered_subquestions": ["true", "false"]}'
        )
        self.assertFalse(parsed["sufficient"])
        self.assertEqual(parsed["covered_subquestions"], [True, False])

    def test_malformed_gate_output_abstains_without_crashing(self) -> None:
        def malformed(prompt: str) -> str:
            if "TASK: DECOMPOSE" in prompt:
                return '{"subquestions": ["first", "second"]}'
            if "TASK: GLOBAL_GATE" in prompt or "TASK: LOCAL_GATE" in prompt:
                return "not json"
            return '{"status": "answer", "answer": "x", "evidence_ids": []}'

        pipeline = RAGPipeline(
            backend=FakeBackend(responder=malformed),
            top_k=3,
            max_total_tokens=4096,
            max_new_tokens=128,
        )
        for method in ("G2", "G3"):
            result = pipeline.run(self.variants[0], method)
            self.assertEqual(result.status, "abstain")
            self.assertGreaterEqual(result.parse_errors, 1)

    def test_exact_budget_overrun_abstains(self) -> None:
        class OverrunBackend:
            def count_tokens(self, text: str) -> int:
                return 1

            def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
                return GenerationResult(
                    text='{"status":"answer","answer":"x","evidence_ids":[]}',
                    input_tokens=100,
                    output_tokens=1,
                    latency_ms=0.0,
                )

        pipeline = RAGPipeline(
            backend=OverrunBackend(),
            top_k=3,
            max_total_tokens=50,
            max_new_tokens=3,
        )
        result = pipeline.run(self.variants[0], "G0")
        self.assertEqual(result.status, "abstain")

    def test_gate_threshold_can_force_abstention(self) -> None:
        pipeline = RAGPipeline(
            backend=FakeBackend(),
            top_k=3,
            max_total_tokens=4096,
            max_new_tokens=128,
            gate_threshold=0.95,
        )
        result = pipeline.run(self.variants[0], "G3")
        self.assertEqual(result.status, "abstain")


if __name__ == "__main__":
    unittest.main()
