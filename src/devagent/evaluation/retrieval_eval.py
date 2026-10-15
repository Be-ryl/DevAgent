"""Offline chunk-level comparison for BM25, Dense, and Hybrid retrieval."""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Protocol

from devagent.rag.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K, HybridRagService
from devagent.rag.service import (
    BM25RagService,
    Chunk,
    RetrievalHit,
    chunk_documents,
    load_documents,
)


CASE_TYPES = ("exact", "paraphrase", "mixed", "hard")
LEGACY_CASE_TYPES = ("typo", "noise")
DIFFICULTIES = ("easy", "medium", "hard")
CHALLENGE_TYPES = (
    "exact_term",
    "semantic_paraphrase",
    "condition_difference",
    "similar_policy_distractor",
    "numeric_constraint",
    "noisy_query",
    "multi_intent",
)
MODE_LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
DEFAULT_WARMUP_QUERY_COUNT = 5
DEFAULT_LATENCY_RUNS = 5


class RetrievalService(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return ranked results for one query."""


@dataclass(frozen=True)
class RetrievalCase:
    """One manually reviewed, chunk-level retrieval question."""

    id: str
    topic_id: str
    query: str
    expected_source: str
    evidence_substring: str
    type: str
    expected_chunk_id: str | None = None
    difficulty: str = "medium"
    challenge_type: str = "exact_term"


@dataclass(frozen=True)
class CaseResult:
    """Observed target-chunk rank and returned candidates for one mode."""

    case: RetrievalCase
    rank: int | None
    actual_sources: list[str]
    actual_chunk_ids: list[str] = field(default_factory=list)
    top_results: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    latency_samples_ms: list[float] = field(default_factory=list)
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass(frozen=True)
class Metrics:
    """Chunk-level Recall@1, Recall@3, MRR, and timing aggregates."""

    total: int
    hits_at_1: int
    hits_at_3: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float

    @property
    def hits(self) -> int:
        """Backward-compatible alias for Hits@3."""
        return self.hits_at_3

    @property
    def recall_at_k(self) -> float:
        """Backward-compatible alias for Recall@3."""
        return self.recall_at_3


@dataclass(frozen=True)
class EvaluationSummary:
    """Overall metrics plus every requested grouping dimension."""

    overall: Metrics
    by_type: dict[str, Metrics]
    by_difficulty: dict[str, Metrics]
    by_challenge_type: dict[str, Metrics]


def _load_index_chunks(chunks_path: Path | str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(chunks_path).open("r", encoding="utf-8") as chunk_file:
        for line_number, line in enumerate(chunk_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                chunks.append(Chunk(**payload))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"invalid chunks JSONL at line {line_number}"
                ) from error
    return chunks


def load_cases(
    cases_path: Path | str,
    knowledge_directory: Path | str,
    chunks_path: Path | str | None = None,
) -> list[RetrievalCase]:
    """Load cases and fail fast on invalid source, chunk, or evidence labels."""
    documents = load_documents(knowledge_directory)
    sources = {document.source for document in documents}
    chunks = (
        _load_index_chunks(chunks_path)
        if chunks_path is not None
        else chunk_documents(documents)
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    topic_truths: dict[str, tuple[str, str, str]] = {}
    required_fields = {
        "id",
        "topic_id",
        "query",
        "expected_source",
        "evidence_substring",
        "type",
    }

    with Path(cases_path).open("r", encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if not isinstance(payload, dict) or not required_fields.issubset(payload):
                raise ValueError(f"missing case fields at line {line_number}")
            string_fields = required_fields | {
                key
                for key in ("expected_chunk_id", "difficulty", "challenge_type")
                if key in payload
            }
            if any(
                not isinstance(payload[field], str) or not payload[field].strip()
                for field in string_fields
            ):
                raise ValueError(
                    f"case fields must be non-empty strings at line {line_number}"
                )

            if payload["id"] in seen_ids:
                raise ValueError(f"duplicate case id: {payload['id']}")
            if payload["type"] not in CASE_TYPES + LEGACY_CASE_TYPES:
                raise ValueError(
                    f"unsupported case type for {payload['id']}: {payload['type']}"
                )
            difficulty = payload.get("difficulty", "medium")
            challenge_type = payload.get("challenge_type", "exact_term")
            if difficulty not in DIFFICULTIES:
                raise ValueError(
                    f"unsupported difficulty for {payload['id']}: {difficulty}"
                )
            if challenge_type not in CHALLENGE_TYPES:
                raise ValueError(
                    f"unsupported challenge type for {payload['id']}: {challenge_type}"
                )
            expected_source = payload["expected_source"]
            if expected_source not in sources:
                raise ValueError(
                    f"unknown expected source for {payload['id']}: {expected_source}"
                )

            expected_chunk_id = payload.get("expected_chunk_id")
            evidence = payload["evidence_substring"]
            if expected_chunk_id is None:
                matching_chunks = [
                    chunk
                    for chunk in chunks_by_source.get(expected_source, [])
                    if evidence in chunk.content
                ]
                if not matching_chunks:
                    raise ValueError(
                        f"evidence substring not found for {payload['id']} "
                        f"in {expected_source}"
                    )
                if len(matching_chunks) != 1:
                    raise ValueError(
                        f"legacy case {payload['id']} must identify exactly one evidence chunk"
                    )
                expected_chunk_id = matching_chunks[0].chunk_id
            target_chunk = chunks_by_id.get(expected_chunk_id)
            if target_chunk is None:
                raise ValueError(
                    f"unknown expected chunk for {payload['id']}: {expected_chunk_id}"
                )
            if target_chunk.source != expected_source:
                raise ValueError(
                    f"expected source does not match target chunk for {payload['id']}"
                )
            if evidence not in target_chunk.content:
                raise ValueError(
                    f"evidence substring not found in target chunk for {payload['id']}"
                )

            case = RetrievalCase(
                id=payload["id"],
                topic_id=payload["topic_id"],
                query=payload["query"],
                expected_source=expected_source,
                expected_chunk_id=expected_chunk_id,
                evidence_substring=evidence,
                type=payload["type"],
                difficulty=difficulty,
                challenge_type=challenge_type,
            )
            truth = (expected_source, expected_chunk_id, evidence)
            previous_truth = topic_truths.setdefault(case.topic_id, truth)
            if previous_truth != truth:
                raise ValueError(f"inconsistent labels for topic: {case.topic_id}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


def _hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(hit).items()
        if value is not None
    }


def calculate_percentile(samples: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile over sorted samples."""
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")
    if not samples:
        return 0.0
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * weight


def _latency_statistics(samples: Sequence[float]) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    return (
        sum(samples) / len(samples),
        calculate_percentile(samples, 0.50),
        calculate_percentile(samples, 0.95),
    )


def warmup_services(
    services: Mapping[str, RetrievalService],
    cases: Sequence[RetrievalCase],
    top_k: int,
    warmup_query_count: int,
) -> None:
    """Warm every selected mode with the first deterministic query subset."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must be non-negative")
    warmup_cases = cases[:warmup_query_count]
    for service in services.values():
        for case in warmup_cases:
            service.search(case.query, top_k=top_k)


def evaluate_cases(
    service: RetrievalService,
    cases: Sequence[RetrievalCase],
    top_k: int,
    *,
    latency_runs: int = 1,
) -> list[CaseResult]:
    """Search repeatedly, time only retrieval, and score the first result."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if latency_runs <= 0:
        raise ValueError("latency_runs must be positive")
    results: list[CaseResult] = []
    for case in cases:
        latency_samples_ms: list[float] = []
        first_hits: list[RetrievalHit] | None = None
        first_chunk_ids: list[str] | None = None
        for run_number in range(1, latency_runs + 1):
            started = time.perf_counter()
            hits = service.search(case.query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            latency_samples_ms.append(elapsed_ms)
            chunk_ids = [hit.chunk_id for hit in hits]
            if first_hits is None:
                first_hits = hits
                first_chunk_ids = chunk_ids
            elif chunk_ids != first_chunk_ids:
                raise RuntimeError(
                    f"unstable retrieval ordering for case {case.id} "
                    f"on latency run {run_number}"
                )

        hits = first_hits or []
        average_latency_ms, p50_latency_ms, p95_latency_ms = (
            _latency_statistics(latency_samples_ms)
        )
        rank = next(
            (
                position
                for position, hit in enumerate(hits, start=1)
                if hit.chunk_id == case.expected_chunk_id
            ),
            None,
        )
        results.append(
            CaseResult(
                case=case,
                rank=rank,
                actual_sources=[hit.source for hit in hits],
                actual_chunk_ids=[hit.chunk_id for hit in hits],
                top_results=[_hit_payload(hit) for hit in hits[:3]],
                latency_ms=average_latency_ms,
                latency_samples_ms=latency_samples_ms,
                average_latency_ms=average_latency_ms,
                p50_latency_ms=p50_latency_ms,
                p95_latency_ms=p95_latency_ms,
            )
        )
    return results


def evaluate_modes(
    services: Mapping[str, RetrievalService],
    cases: Sequence[RetrievalCase],
    top_k: int,
    *,
    latency_runs: int = 1,
) -> dict[str, list[CaseResult]]:
    """Run every selected service against the exact same immutable case list."""
    return {
        mode: evaluate_cases(service, cases, top_k, latency_runs=latency_runs)
        for mode, service in services.items()
    }


def calculate_metrics(results: Sequence[CaseResult]) -> Metrics:
    """Calculate chunk-level Recall@1, Recall@3, and cutoff MRR."""
    total = len(results)
    hits_at_1 = sum(result.rank == 1 for result in results)
    hits_at_3 = sum(
        result.rank is not None and result.rank <= 3 for result in results
    )
    reciprocal_rank_sum = sum(
        1.0 / result.rank for result in results if result.rank is not None
    )
    latency_samples_ms = [
        sample
        for result in results
        for sample in (
            result.latency_samples_ms
            if result.latency_samples_ms
            else [result.latency_ms]
        )
    ]
    average_latency_ms, p50_latency_ms, p95_latency_ms = _latency_statistics(
        latency_samples_ms
    )
    return Metrics(
        total=total,
        hits_at_1=hits_at_1,
        hits_at_3=hits_at_3,
        recall_at_1=hits_at_1 / total if total else 0.0,
        recall_at_3=hits_at_3 / total if total else 0.0,
        mrr=reciprocal_rank_sum / total if total else 0.0,
        average_latency_ms=average_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def _group_metrics(
    results: Sequence[CaseResult],
    attribute: str,
) -> dict[str, Metrics]:
    known_values = {
        "type": CASE_TYPES + LEGACY_CASE_TYPES,
        "difficulty": DIFFICULTIES,
        "challenge_type": CHALLENGE_TYPES,
    }
    values = known_values.get(
        attribute,
        tuple(sorted({getattr(result.case, attribute) for result in results})),
    )
    return {
        value: calculate_metrics(
            [result for result in results if getattr(result.case, attribute) == value]
        )
        for value in values
    }


def summarize(results: Sequence[CaseResult]) -> EvaluationSummary:
    """Calculate overall and type, difficulty, and challenge group metrics."""
    return EvaluationSummary(
        overall=calculate_metrics(results),
        by_type=_group_metrics(results, "type"),
        by_difficulty=_group_metrics(results, "difficulty"),
        by_challenge_type=_group_metrics(results, "challenge_type"),
    )


def _metrics_payload(metrics: Metrics) -> dict[str, Any]:
    return asdict(metrics)


def _print_table(
    title: str,
    summaries: Mapping[str, EvaluationSummary],
    group: str | None = None,
    value: str | None = None,
) -> None:
    print(f"\n{title}")
    print(
        "Mode      Recall@1 Recall@3 MRR    "
        "Avg(ms)  P50(ms)  P95(ms)  Hits@3"
    )
    for mode, summary in summaries.items():
        metrics = summary.overall
        if group is not None and value is not None:
            metrics = getattr(summary, f"by_{group}")[value]
        print(
            f"{MODE_LABELS[mode]:<10}"
            f"{metrics.recall_at_1:<9.4f}"
            f"{metrics.recall_at_3:<9.4f}"
            f"{metrics.mrr:<7.4f}"
            f"{metrics.average_latency_ms:<9.3f}"
            f"{metrics.p50_latency_ms:<9.3f}"
            f"{metrics.p95_latency_ms:<9.3f}"
            f"{metrics.hits_at_3}/{metrics.total}"
        )


def _print_verbose(
    cases: Sequence[RetrievalCase],
    results_by_mode: Mapping[str, Sequence[CaseResult]],
) -> None:
    indexed = {
        mode: {result.case.id: result for result in results}
        for mode, results in results_by_mode.items()
    }
    for case in cases:
        print(f"\nCASE {case.id}: {case.query}")
        print(f"target={case.expected_chunk_id} source={case.expected_source}")
        for mode in results_by_mode:
            result = indexed[mode][case.id]
            status = "PASS" if result.rank is not None else "FAIL"
            print(f"  {MODE_LABELS[mode]} {status} rank={result.rank or '-'}")
            for hit in result.top_results:
                bm25_rank = (
                    hit["rank"] if mode == "bm25" else hit.get("bm25_rank", "-")
                )
                dense_rank = (
                    hit["rank"] if mode == "dense" else hit.get("dense_rank", "-")
                )
                details = (
                    f"bm25_rank={bm25_rank} "
                    f"dense_rank={dense_rank} "
                    f"rrf_score={hit.get('rrf_score', '-')}"
                )
                print(
                    f"    #{hit['rank']} {hit['chunk_id']} "
                    f"source={hit['source']} {details}"
                )


def build_report(
    *,
    args: argparse.Namespace,
    cases: Sequence[RetrievalCase],
    results_by_mode: Mapping[str, Sequence[CaseResult]],
    summaries: Mapping[str, EvaluationSummary],
) -> dict[str, Any]:
    """Build the complete machine-readable comparison report."""
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for group in ("type", "difficulty", "challenge_type"):
        grouped[group] = {}
        values = sorted({getattr(case, group) for case in cases})
        for value in values:
            grouped[group][value] = {
                mode: _metrics_payload(getattr(summary, f"by_{group}")[value])
                for mode, summary in summaries.items()
            }

    indexed = {
        mode: {result.case.id: result for result in results}
        for mode, results in results_by_mode.items()
    }
    case_results = []
    failures = []
    for case in cases:
        modes: dict[str, Any] = {}
        for mode in results_by_mode:
            result = indexed[mode][case.id]
            mode_payload = {
                "rank": result.rank,
                "passed": result.rank is not None,
                "latency_ms": result.latency_ms,
                "latency_samples_ms": result.latency_samples_ms,
                "average_latency_ms": result.average_latency_ms,
                "p50_latency_ms": result.p50_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "top_3": result.top_results,
            }
            modes[mode] = mode_payload
            if result.rank is None:
                failures.append(
                    {
                        "case_id": case.id,
                        "mode": mode,
                        "expected_chunk_id": case.expected_chunk_id,
                        "top_3_chunk_ids": result.actual_chunk_ids[:3],
                    }
                )
        case_results.append({"case": asdict(case), "modes": modes})

    return {
        "run_time": datetime.now(timezone.utc).isoformat(),
        "knowledge_directory": str(args.knowledge_dir.resolve()),
        "cases_path": str(args.cases.resolve()),
        "dense_index_directory": (
            str(args.dense_index_dir.resolve()) if args.dense_index_dir else None
        ),
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "candidate_top_k": args.candidate_top_k,
        "rrf_k": args.rrf_k,
        "warmup_query_count": getattr(
            args,
            "warmup_query_count",
            DEFAULT_WARMUP_QUERY_COUNT,
        ),
        "latency_runs": getattr(args, "latency_runs", DEFAULT_LATENCY_RUNS),
        "timing_method": "perf_counter",
        "case_count": len(cases),
        "case_distribution": {
            group: dict(sorted(Counter(getattr(case, group) for case in cases).items()))
            for group in ("type", "difficulty", "challenge_type", "topic_id")
        },
        "overall": {
            mode: _metrics_payload(summary.overall)
            for mode, summary in summaries.items()
        },
        "grouped": grouped,
        "cases": case_results,
        "failures": failures,
    }


def write_report(report_path: Path, report: Mapping[str, Any]) -> None:
    """Overwrite one JSON report path with the latest evaluation."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _selected_modes(retrieval_mode: str) -> tuple[str, ...]:
    return (
        ("bm25", "dense", "hybrid")
        if retrieval_mode == "all"
        else (retrieval_mode,)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run chunk-level offline retrieval evaluation and optional JSON report."""
    parser = argparse.ArgumentParser(description="Compare local retrieval modes.")
    parser.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense", "hybrid", "all"),
        default="bm25",
    )
    parser.add_argument("--knowledge-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--warmup-query-count",
        type=int,
        default=DEFAULT_WARMUP_QUERY_COUNT,
    )
    parser.add_argument("--latency-runs", type=int, default=DEFAULT_LATENCY_RUNS)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    modes = _selected_modes(args.retrieval_mode)
    if not args.knowledge_dir.is_dir():
        print(
            f"Error: knowledge directory is not a directory: {args.knowledge_dir}",
            file=sys.stderr,
        )
        return 2
    if any(mode in {"dense", "hybrid"} for mode in modes):
        if args.dense_index_dir is None or not args.dense_index_dir.is_dir():
            print("Error: a valid --dense-index-dir is required", file=sys.stderr)
            return 2
    if (
        args.top_k <= 0
        or args.candidate_top_k <= 0
        or args.rrf_k <= 0
        or args.latency_runs <= 0
        or args.warmup_query_count < 0
    ):
        print(
            "Error: retrieval limits, rrf-k, and latency-runs must be positive; "
            "warmup-query-count must be non-negative",
            file=sys.stderr,
        )
        return 2

    hybrid_service: HybridRagService | None = None
    try:
        chunks_path = (
            args.dense_index_dir / "chunks.jsonl"
            if args.dense_index_dir is not None
            else None
        )
        cases = load_cases(args.cases, args.knowledge_dir, chunks_path=chunks_path)
        services: dict[str, RetrievalService] = {}
        bm25_service: RetrievalService | None = None
        dense_service: RetrievalService | None = None
        if any(mode in {"bm25", "hybrid"} for mode in modes):
            bm25_service = BM25RagService.from_directory(args.knowledge_dir)
        if any(mode in {"dense", "hybrid"} for mode in modes):
            from devagent.rag.dense_index import DenseRagService

            dense_service = DenseRagService.from_index(
                args.dense_index_dir,
                model_name=args.embedding_model,
                knowledge_directory=args.knowledge_dir,
            )
        if "bm25" in modes and bm25_service is not None:
            services["bm25"] = bm25_service
        if "dense" in modes and dense_service is not None:
            services["dense"] = dense_service
        if "hybrid" in modes and bm25_service is not None and dense_service is not None:
            hybrid_service = HybridRagService(
                bm25_service,
                dense_service,
                candidate_top_k=args.candidate_top_k,
                rrf_k=args.rrf_k,
            )
            services["hybrid"] = hybrid_service

        print(f"Warmup queries: {args.warmup_query_count}")
        print(f"Latency runs per query: {args.latency_runs}")
        warmup_services(
            services,
            cases,
            args.top_k,
            args.warmup_query_count,
        )
        results_by_mode = evaluate_modes(
            services,
            cases,
            args.top_k,
            latency_runs=args.latency_runs,
        )
        summaries = {
            mode: summarize(results) for mode, results in results_by_mode.items()
        }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if hybrid_service is not None:
            hybrid_service.close()

    if args.verbose:
        _print_verbose(cases, results_by_mode)
    _print_table("Overall", summaries)
    for group in ("type", "difficulty", "challenge_type"):
        for value in sorted({getattr(case, group) for case in cases}):
            _print_table(
                f"{group.replace('_', ' ').title()}: {value}",
                summaries,
                group,
                value,
            )

    if args.report is not None:
        report = build_report(
            args=args,
            cases=cases,
            results_by_mode=results_by_mode,
            summaries=summaries,
        )
        write_report(args.report, report)
        print(f"\nReport: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
