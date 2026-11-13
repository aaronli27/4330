from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25  # noqa: E402
from src.data_prepare import load_valid_records, stratified_split  # noqa: E402
from src.perturbations import build_distractor_pool, build_variants  # noqa: E402


DATASET = Path(
    r"D:\项目\大作业\datasets\hotpotqa\hotpot_dev_distractor_v1.json"
)


class CoreProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, cls.excluded = load_valid_records(DATASET)

    def test_dataset_validation_matches_local_audit(self) -> None:
        self.assertEqual(len(self.records), 7344)
        self.assertEqual(len(self.excluded), 61)
        self.assertEqual(len({record.record_id for record in self.records}), 7344)

    def test_stratified_split_has_no_leakage(self) -> None:
        splits = stratified_split(
            self.records,
            {
                "pilot": 50,
                "calibration": 100,
                "standard_test": 100,
                "unseen_test": 100,
            },
            seed=20261019,
        )
        all_ids = [
            record.record_id for values in splits.values() for record in values
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual({name: len(values) for name, values in splits.items()},
                         {"pilot": 50, "calibration": 100, "standard_test": 100, "unseen_test": 100})

    def test_bm25_returns_deterministic_hits(self) -> None:
        index = BM25(
            [
                ("a", "Scott Derrickson is an American filmmaker."),
                ("b", "Ed Wood was an American filmmaker and director."),
                ("c", "A football club in England."),
            ]
        )
        first = index.search("American filmmaker", top_k=3)
        second = index.search("American filmmaker", top_k=3)
        self.assertEqual(first, second)
        self.assertEqual([hit.doc_id for hit in first[:2]], ["a", "b"])

    def test_variants_preserve_candidate_count(self) -> None:
        records = self.records[:3]
        pool = build_distractor_pool(self.records, max_paragraphs=2000)
        variants = build_variants(records, pool, seed=20261019)
        self.assertEqual(len(variants), 12)
        for variant in variants:
            self.assertEqual(len(variant.context), 10)
            titles = [paragraph.title for paragraph in variant.context]
            self.assertEqual(len(titles), len(set(titles)))
            if variant.condition in {"C0", "C2"}:
                self.assertTrue(
                    all(
                        title in titles
                        for title, _ in variant.supporting_facts
                    )
                )
            if variant.condition in {"C1", "C3"}:
                missing = variant.perturbation["missing_support_title"]
                self.assertNotIn(missing, titles)


if __name__ == "__main__":
    unittest.main()

