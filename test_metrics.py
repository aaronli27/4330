from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import aurc, exact_match, token_f1  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_answer_normalization(self) -> None:
        self.assertEqual(exact_match("The United States", "United States"), 1.0)
        self.assertEqual(token_f1("a red fox", "red fox"), 1.0)
        self.assertEqual(token_f1("blue car", "red car"), 0.5)

    def test_aurc_uses_selection_score_order(self) -> None:
        rows = [
            {
                "record_id": "a",
                "status": "answer",
                "selection_score": 0.9,
                "selection_loss": 0.0,
            },
            {
                "record_id": "b",
                "status": "abstain",
                "selection_score": 0.1,
                "selection_loss": 0.0,
            },
        ]
        self.assertAlmostEqual(aurc(rows), 0.0)


if __name__ == "__main__":
    unittest.main()
