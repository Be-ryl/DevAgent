"""Tests for chunk-level BM25, Dense, and Hybrid offline evaluation."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from devagent.evaluation.retrieval_eval import (
    CaseResult,
    RetrievalCase,
    build_report,
    calculate_percentile,
    calculate_metrics,
    evaluate_cases,
    evaluate_modes,
    load_cases,
    main,
    summarize,
    warmup_services,
    write_report,
)
from devagent.rag.service import RetrievalHit, chunk_documents, load_documents


def make_case(
    case_id: str,
    case_type: str = "exact",
    *,
    expected_chunk_id: str = "target",
    difficulty: str = "medium",
    challenge_type: str = "exact_term",
) -> RetrievalCase:
    return RetrievalCase(
        id=case_id,
        topic_id=case_id,
        query=f"query-{case_id}",
        expected_source="policy.md",
        expected_chunk_id=expected_chunk_id,
        evidence_substring="evidence",
        type=case_type,
        difficulty=difficulty,
        challenge_type=challenge_type,
    )


def make_hit(chunk_id: str, rank: int, source: str = "policy.md") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source=source,
        content=f"content-{chunk_id}",
        score=1.0 / rank,
        rank=rank,
        retrieved_by=["test"],
    )


class FakeService:
    def __init__(self, results_by_query=None) -> None:
        self.results_by_query = results_by_query or {}
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        self.queries.append(query)
        return self.results_by_query.get(query, [])[:top_k]


class AlternatingService:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        self.calls += 1
        chunk_id = "first" if self.calls % 2 else "second"
        return [make_hit(chunk_id, 1)]


class RetrievalMetricsTest(unittest.TestCase):
    def test_rank_one_two_three_mrr_and_recall_are_calculated(self) -> None:
        results = [
            CaseResult(make_case("one"), 1, ["policy.md"]),
            CaseResult(make_case("two"), 2, ["policy.md"]),
            CaseResult(make_case("three"), 3, ["policy.md"]),
            CaseResult(make_case("miss"), None, ["other.md"]),
        ]

        metrics = calculate_metrics(results)

        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.hits_at_1, 1)
        self.assertEqual(metrics.hits_at_3, 3)
        self.assertAlmostEqual(metrics.recall_at_1, 0.25)
        self.assertAlmostEqual(metrics.recall_at_3, 0.75)
        self.assertAlmostEqual(metrics.mrr, (1 + 0.5 + 1 / 3) / 4)

    def test_type_difficulty_and_challenge_groups_are_correct(self) -> None:
        results = [
            CaseResult(
                make_case("easy", "exact", difficulty="easy"),
                1,
                ["policy.md"],
            ),
            CaseResult(
                make_case(
                    "hard",
                    "hard",
                    difficulty="hard",
                    challenge_type="similar_policy_distractor",
                ),
                None,
                [],
            ),
        ]

        summary = summarize(results)

        self.assertEqual(summary.by_type["exact"].recall_at_1, 1.0)
        self.assertEqual(summary.by_type["hard"].recall_at_3, 0.0)
        self.assertEqual(summary.by_difficulty["easy"].mrr, 1.0)
        self.assertEqual(summary.by_difficulty["hard"].mrr, 0.0)
        self.assertEqual(
            summary.by_challenge_type["similar_policy_distractor"].total,
            1,
        )

    def test_same_source_wrong_chunk_is_not_a_hit(self) -> None:
        case = make_case("same-source", expected_chunk_id="correct")
        service = FakeService(
            {case.query: [make_hit("wrong", 1, source="policy.md")]}
        )

        result = evaluate_cases(service, [case], top_k=3)[0]

        self.assertIsNone(result.rank)
        self.assertEqual(result.actual_sources, ["policy.md"])

    def test_target_beyond_top_k_is_a_miss(self) -> None:
        case = make_case("cutoff", expected_chunk_id="target")
        service = FakeService(
            {
                case.query: [
                    make_hit("one", 1),
                    make_hit("two", 2),
                    make_hit("three", 3),
                    make_hit("target", 4),
                ]
            }
        )

        result = evaluate_cases(service, [case], top_k=3)[0]

        self.assertIsNone(result.rank)
        self.assertEqual(result.actual_chunk_ids, ["one", "two", "three"])

    def test_all_modes_run_the_same_query_batch(self) -> None:
        cases = [make_case("a"), make_case("b")]
        services = {
            "bm25": FakeService(),
            "dense": FakeService(),
            "hybrid": FakeService(),
        }

        results = evaluate_modes(services, cases, top_k=3)

        expected_queries = [case.query for case in cases]
        self.assertEqual(set(results), {"bm25", "dense", "hybrid"})
        for service in services.values():
            self.assertEqual(service.queries, expected_queries)

    def test_warmup_uses_first_queries_without_adding_scored_results(self) -> None:
        cases = [make_case("a"), make_case("b")]
        services = {
            "bm25": FakeService(),
            "dense": FakeService(),
            "hybrid": FakeService(),
        }

        warmup_services(services, cases, top_k=3, warmup_query_count=1)
        results = evaluate_modes(services, cases, top_k=3, latency_runs=2)

        for mode, service in services.items():
            self.assertEqual(
                service.queries,
                [
                    cases[0].query,
                    cases[0].query,
                    cases[0].query,
                    cases[1].query,
                    cases[1].query,
                ],
            )
            self.assertEqual(len(results[mode]), 2)
            self.assertEqual(calculate_metrics(results[mode]).total, 2)
            self.assertTrue(
                all(len(result.latency_samples_ms) == 2 for result in results[mode])
            )

    def test_zero_warmup_disables_warmup_calls(self) -> None:
        service = FakeService()

        warmup_services(
            {"bm25": service},
            [make_case("a")],
            top_k=3,
            warmup_query_count=0,
        )

        self.assertEqual(service.queries, [])

    @patch(
        "devagent.evaluation.retrieval_eval.time.perf_counter",
        side_effect=[0.0, 0.001, 1.0, 1.003, 2.0, 2.009],
    )
    def test_repeated_timing_samples_and_percentiles_are_calculated(
        self,
        mock_perf_counter,
    ) -> None:
        case = make_case("timed")
        service = FakeService({case.query: [make_hit("target", 1)]})

        result = evaluate_cases(service, [case], top_k=3, latency_runs=3)[0]
        metrics = calculate_metrics([result])

        self.assertEqual(service.queries, [case.query] * 3)
        self.assertEqual(len(result.latency_samples_ms), 3)
        for actual, expected in zip(
            result.latency_samples_ms,
            [1.0, 3.0, 9.0],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.average_latency_ms, 13.0 / 3)
        self.assertAlmostEqual(result.p50_latency_ms, 3.0)
        self.assertAlmostEqual(result.p95_latency_ms, 8.4)
        self.assertAlmostEqual(metrics.average_latency_ms, 13.0 / 3)
        self.assertAlmostEqual(metrics.p50_latency_ms, 3.0)
        self.assertAlmostEqual(metrics.p95_latency_ms, 8.4)
        self.assertEqual(mock_perf_counter.call_count, 6)

    def test_percentile_uses_linear_interpolation(self) -> None:
        samples = [1.0, 3.0, 9.0]

        self.assertEqual(calculate_percentile(samples, 0.50), 3.0)
        self.assertAlmostEqual(calculate_percentile(samples, 0.95), 8.4)

    def test_mrr_calculation_does_not_read_the_timing_clock(self) -> None:
        case = make_case("mrr")
        service = FakeService({case.query: [make_hit("target", 1)]})

        with patch(
            "devagent.evaluation.retrieval_eval.time.perf_counter",
            side_effect=[10.0, 10.002],
        ) as mock_perf_counter:
            result = evaluate_cases(service, [case], top_k=3)[0]
            metrics = calculate_metrics([result])

        self.assertEqual(metrics.mrr, 1.0)
        self.assertAlmostEqual(result.latency_ms, 2.0)
        self.assertEqual(mock_perf_counter.call_count, 2)

    def test_unstable_repeated_results_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unstable retrieval ordering"):
            evaluate_cases(
                AlternatingService(),
                [make_case("unstable")],
                top_k=3,
                latency_runs=2,
            )

    def test_non_positive_latency_runs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "latency_runs must be positive"):
            evaluate_cases(FakeService(), [make_case("a")], 3, latency_runs=0)


class RetrievalCaseLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.policies = self.root / "policies"
        self.policies.mkdir()
        (self.policies / "policy.md").write_text(
            "# 政策\n\n这里包含明确证据。\n\n另一个政策段落。",
            encoding="utf-8",
        )
        self.chunks = chunk_documents(load_documents(self.policies))
        self.target = next(
            chunk for chunk in self.chunks if "明确证据" in chunk.content
        )
        self.cases_path = self.root / "cases.jsonl"

    def write_case(self, **overrides) -> None:
        payload = {
            "id": "case-01",
            "topic_id": "topic-01",
            "query": "政策问题",
            "expected_source": "policy.md",
            "expected_chunk_id": self.target.chunk_id,
            "evidence_substring": "明确证据",
            "type": "exact",
            "difficulty": "easy",
            "challenge_type": "exact_term",
        }
        payload.update(overrides)
        self.cases_path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_valid_case_preserves_source_chunk_and_evidence(self) -> None:
        self.write_case()

        cases = load_cases(self.cases_path, self.policies)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_source, "policy.md")
        self.assertEqual(cases[0].expected_chunk_id, self.target.chunk_id)

    def test_unknown_expected_source_is_rejected(self) -> None:
        self.write_case(expected_source="missing.md")

        with self.assertRaisesRegex(ValueError, "unknown expected source"):
            load_cases(self.cases_path, self.policies)

    def test_unknown_expected_chunk_is_rejected(self) -> None:
        self.write_case(expected_chunk_id="missing-chunk")

        with self.assertRaisesRegex(ValueError, "unknown expected chunk"):
            load_cases(self.cases_path, self.policies)

    def test_evidence_outside_target_chunk_is_rejected(self) -> None:
        other = next(chunk for chunk in self.chunks if chunk != self.target)
        self.write_case(expected_chunk_id=other.chunk_id)

        with self.assertRaisesRegex(ValueError, "evidence substring not found"):
            load_cases(self.cases_path, self.policies)

    def test_source_must_match_target_chunk(self) -> None:
        (self.policies / "other.md").write_text(
            "# 其他\n\n其他证据。",
            encoding="utf-8",
        )
        other_chunk = next(
            chunk
            for chunk in chunk_documents(load_documents(self.policies))
            if chunk.source == "other.md"
        )
        self.write_case(expected_chunk_id=other_chunk.chunk_id)

        with self.assertRaisesRegex(ValueError, "source does not match"):
            load_cases(self.cases_path, self.policies)

    def test_malformed_jsonl_is_rejected(self) -> None:
        self.cases_path.write_text('{"id":\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid JSONL"):
            load_cases(self.cases_path, self.policies)


class RetrievalReportTest(unittest.TestCase):
    def test_json_report_contains_required_metadata_groups_and_failures(self) -> None:
        case = make_case("report-case", expected_chunk_id="target")
        passed = CaseResult(
            case,
            1,
            ["policy.md"],
            ["target"],
            [{"chunk_id": "target", "rank": 1}],
            1.25,
            [1.0, 1.5],
            1.25,
            1.25,
            1.475,
        )
        failed = CaseResult(
            case,
            None,
            ["policy.md"],
            ["wrong"],
            [],
            2.5,
            [2.0, 3.0],
            2.5,
            2.5,
            2.95,
        )
        results = {"bm25": [passed], "dense": [failed], "hybrid": [passed]}
        summaries = {mode: summarize(items) for mode, items in results.items()}
        args = argparse.Namespace(
            knowledge_dir=Path("/tmp/knowledge"),
            cases=Path("/tmp/cases.jsonl"),
            dense_index_dir=Path("/tmp/index"),
            embedding_model="fake-model",
            top_k=3,
            candidate_top_k=20,
            rrf_k=60,
            warmup_query_count=5,
            latency_runs=5,
        )

        report = build_report(
            args=args,
            cases=[case],
            results_by_mode=results,
            summaries=summaries,
        )

        required = {
            "run_time",
            "knowledge_directory",
            "cases_path",
            "embedding_model",
            "top_k",
            "candidate_top_k",
            "rrf_k",
            "warmup_query_count",
            "latency_runs",
            "timing_method",
            "case_count",
            "case_distribution",
            "overall",
            "grouped",
            "cases",
            "failures",
        }
        self.assertTrue(required.issubset(report))
        self.assertEqual(set(report["overall"]), {"bm25", "dense", "hybrid"})
        self.assertIn("type", report["grouped"])
        self.assertIn("difficulty", report["grouped"])
        self.assertIn("challenge_type", report["grouped"])
        self.assertEqual(report["failures"][0]["mode"], "dense")
        self.assertEqual(report["timing_method"], "perf_counter")
        bm25_metrics = report["overall"]["bm25"]
        self.assertIn("average_latency_ms", bm25_metrics)
        self.assertIn("p50_latency_ms", bm25_metrics)
        self.assertIn("p95_latency_ms", bm25_metrics)
        case_timing = report["cases"][0]["modes"]["bm25"]
        self.assertEqual(case_timing["latency_samples_ms"], [1.0, 1.5])
        self.assertEqual(case_timing["average_latency_ms"], 1.25)
        self.assertEqual(case_timing["p50_latency_ms"], 1.25)
        self.assertEqual(case_timing["p95_latency_ms"], 1.475)
        json.dumps(report, ensure_ascii=False)

    def test_write_report_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.json"
            write_report(report_path, {"run": "first", "obsolete": True})
            write_report(report_path, {"run": "second"})

            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, {"run": "second"})

    def test_cli_rejects_non_positive_latency_runs(self) -> None:
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            with redirect_stderr(stderr):
                status = main(
                    [
                        "--knowledge-dir",
                        temporary_directory,
                        "--cases",
                        str(Path(temporary_directory) / "cases.jsonl"),
                        "--latency-runs",
                        "0",
                    ]
                )

        self.assertEqual(status, 2)
        self.assertIn("must be positive", stderr.getvalue())

    def test_offline_evaluation_closes_hybrid_service(self) -> None:
        stdout = StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            knowledge_directory = root / "knowledge"
            dense_index_directory = root / "index"
            knowledge_directory.mkdir()
            dense_index_directory.mkdir()
            with patch(
                "devagent.evaluation.retrieval_eval.load_cases",
                return_value=[],
            ), patch(
                "devagent.evaluation.retrieval_eval.BM25RagService.from_directory"
            ), patch(
                "devagent.rag.dense_index.DenseRagService.from_index"
            ), patch(
                "devagent.evaluation.retrieval_eval.HybridRagService"
            ) as mock_hybrid_service, redirect_stdout(stdout):
                status = main(
                    [
                        "--retrieval-mode",
                        "hybrid",
                        "--knowledge-dir",
                        str(knowledge_directory),
                        "--cases",
                        str(root / "cases.jsonl"),
                        "--dense-index-dir",
                        str(dense_index_directory),
                        "--warmup-query-count",
                        "0",
                    ]
                )

        self.assertEqual(status, 0)
        mock_hybrid_service.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
