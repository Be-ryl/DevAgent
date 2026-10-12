"""Tests for deterministic BM25 offline evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

from devagent.evaluation.retrieval_eval import (
    CaseResult,
    RetrievalCase,
    calculate_metrics,
    load_cases,
    summarize,
)


def make_case(case_id: str, case_type: str) -> RetrievalCase:
    """Build a small labeled case for metric tests."""
    return RetrievalCase(
        id=case_id,
        topic_id=case_id,
        query="query",
        expected_source="policy.md",
        evidence_substring="evidence",
        type=case_type,
    )


class RetrievalMetricsTest(unittest.TestCase):
    """Tests for Recall@K, MRR, and per-type grouping."""

    def test_recall_and_mrr_are_calculated_from_source_ranks(self) -> None:
        results = [
            CaseResult(make_case("one", "exact"), 1, ["policy.md"]),
            CaseResult(make_case("two", "exact"), 2, ["other.md", "policy.md"]),
            CaseResult(make_case("three", "exact"), None, ["other.md"]),
        ]

        metrics = calculate_metrics(results)

        self.assertEqual(metrics.total, 3)
        self.assertEqual(metrics.hits, 2)
        self.assertAlmostEqual(metrics.recall_at_k, 2 / 3)
        self.assertAlmostEqual(metrics.mrr, 0.5)

    def test_metrics_are_grouped_by_case_type(self) -> None:
        results = [
            CaseResult(make_case("exact-hit", "exact"), 1, ["policy.md"]),
            CaseResult(make_case("exact-miss", "exact"), None, []),
            CaseResult(make_case("typo-hit", "typo"), 2, ["other.md", "policy.md"]),
        ]

        summary = summarize(results)

        self.assertEqual(summary.by_type["exact"].recall_at_k, 0.5)
        self.assertEqual(summary.by_type["exact"].mrr, 0.5)
        self.assertEqual(summary.by_type["typo"].recall_at_k, 1.0)
        self.assertEqual(summary.by_type["typo"].mrr, 0.5)
        self.assertEqual(summary.by_type["noise"].total, 0)


class RetrievalCaseLoadingTest(unittest.TestCase):
    """Tests for JSONL structure and evidence-grounding validation."""

    def write_case(self, path: Path, **overrides) -> None:
        payload = {
            "id": "case-01",
            "topic_id": "topic-01",
            "query": "政策问题",
            "expected_source": "policy.md",
            "evidence_substring": "明确证据",
            "type": "exact",
        }
        payload.update(overrides)
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_valid_jsonl_loads_when_evidence_exists_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            policies = root / "policies"
            policies.mkdir()
            (policies / "policy.md").write_text(
                "# 政策\n\n这里包含明确证据。",
                encoding="utf-8",
            )
            cases_path = root / "cases.jsonl"
            self.write_case(cases_path)

            cases = load_cases(cases_path, policies)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_source, "policy.md")

    def test_missing_evidence_substring_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            policies = root / "policies"
            policies.mkdir()
            (policies / "policy.md").write_text("# 政策\n", encoding="utf-8")
            cases_path = root / "cases.jsonl"
            self.write_case(cases_path)

            with self.assertRaisesRegex(ValueError, "evidence substring not found"):
                load_cases(cases_path, policies)

    def test_malformed_jsonl_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            policies = root / "policies"
            policies.mkdir()
            (policies / "policy.md").write_text(
                "# 政策\n\n明确证据",
                encoding="utf-8",
            )
            cases_path = root / "cases.jsonl"
            cases_path.write_text('{"id":\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_cases(cases_path, policies)


if __name__ == "__main__":
    unittest.main()
