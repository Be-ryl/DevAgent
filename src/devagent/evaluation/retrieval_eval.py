"""Offline source-recall evaluation for the local BM25 retriever."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from devagent.rag.service import BM25RagService, load_documents


CASE_TYPES = ("exact", "paraphrase", "typo", "noise")


@dataclass(frozen=True)
class RetrievalCase:
    """One manually labeled retrieval question."""

    id: str
    topic_id: str
    query: str
    expected_source: str
    evidence_substring: str
    type: str


@dataclass(frozen=True)
class CaseResult:
    """Observed source rank for one retrieval case."""

    case: RetrievalCase
    rank: int | None
    actual_sources: list[str]


@dataclass(frozen=True)
class Metrics:
    """Recall and reciprocal-rank aggregates."""

    total: int
    hits: int
    recall_at_k: float
    mrr: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Overall and per-type retrieval metrics."""

    overall: Metrics
    by_type: dict[str, Metrics]


def load_cases(
    cases_path: Path | str,
    knowledge_directory: Path | str,
) -> list[RetrievalCase]:
    """Load JSONL cases and validate every evidence label against Markdown."""
    sources = {
        document.source: document.content
        for document in load_documents(knowledge_directory)
    }
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    topic_truths: dict[str, tuple[str, str]] = {}

    with Path(cases_path).open("r", encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}"
                ) from error

            required_fields = {
                "id",
                "topic_id",
                "query",
                "expected_source",
                "evidence_substring",
                "type",
            }
            if not isinstance(payload, dict) or not required_fields.issubset(payload):
                raise ValueError(f"missing case fields at line {line_number}")
            if any(
                not isinstance(payload[field], str) or not payload[field].strip()
                for field in required_fields
            ):
                raise ValueError(f"case fields must be non-empty strings at line {line_number}")

            case = RetrievalCase(**{field: payload[field] for field in required_fields})
            if case.id in seen_ids:
                raise ValueError(f"duplicate case id: {case.id}")
            if case.type not in CASE_TYPES:
                raise ValueError(f"unsupported case type for {case.id}: {case.type}")
            if case.expected_source not in sources:
                raise ValueError(
                    f"unknown expected source for {case.id}: {case.expected_source}"
                )
            if case.evidence_substring not in sources[case.expected_source]:
                raise ValueError(
                    f"evidence substring not found for {case.id} "
                    f"in {case.expected_source}"
                )

            truth = (case.expected_source, case.evidence_substring)
            previous_truth = topic_truths.setdefault(case.topic_id, truth)
            if previous_truth != truth:
                raise ValueError(f"inconsistent labels for topic: {case.topic_id}")

            seen_ids.add(case.id)
            cases.append(case)

    return cases


def evaluate_cases(
    service: BM25RagService,
    cases: Sequence[RetrievalCase],
    top_k: int,
) -> list[CaseResult]:
    """Run BM25 and record the first rank of each expected source."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results: list[CaseResult] = []
    for case in cases:
        hits = service.search(case.query, top_k=top_k)
        actual_sources = [hit.source for hit in hits]
        rank = next(
            (
                hit.rank
                for hit in hits
                if hit.source == case.expected_source
            ),
            None,
        )
        results.append(
            CaseResult(
                case=case,
                rank=rank,
                actual_sources=actual_sources,
            )
        )
    return results


def calculate_metrics(results: Sequence[CaseResult]) -> Metrics:
    """Calculate Recall@K and MRR from already ranked case results."""
    total = len(results)
    hits = sum(result.rank is not None for result in results)
    reciprocal_rank_sum = sum(
        1.0 / result.rank
        for result in results
        if result.rank is not None
    )
    return Metrics(
        total=total,
        hits=hits,
        recall_at_k=hits / total if total else 0.0,
        mrr=reciprocal_rank_sum / total if total else 0.0,
    )


def summarize(results: Sequence[CaseResult]) -> EvaluationSummary:
    """Calculate overall metrics and the same metrics grouped by case type."""
    return EvaluationSummary(
        overall=calculate_metrics(results),
        by_type={
            case_type: calculate_metrics(
                [result for result in results if result.case.type == case_type]
            )
            for case_type in CASE_TYPES
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic offline retrieval evaluation command."""
    parser = argparse.ArgumentParser(description="Evaluate local BM25 source recall.")
    parser.add_argument("--knowledge-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.knowledge_dir.is_dir():
        print(
            f"Error: knowledge directory is not a directory: {args.knowledge_dir}",
            file=sys.stderr,
        )
        return 2

    try:
        cases = load_cases(args.cases, args.knowledge_dir)
        service = BM25RagService.from_directory(args.knowledge_dir)
        results = evaluate_cases(service, cases, top_k=args.top_k)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    for result in results:
        if result.rank is not None:
            print(
                f"PASS {result.case.id} "
                f"expected={result.case.expected_source} rank={result.rank}"
            )
        else:
            print(
                f"FAIL {result.case.id} "
                f"expected={result.case.expected_source} "
                f"actual={json.dumps(result.actual_sources, ensure_ascii=False)}"
            )

    summary = summarize(results)
    print(
        f"Overall Recall@{args.top_k}: {summary.overall.recall_at_k:.4f} "
        f"({summary.overall.hits}/{summary.overall.total})"
    )
    print(f"Overall MRR: {summary.overall.mrr:.4f}")
    for case_type in CASE_TYPES:
        metrics = summary.by_type[case_type]
        print(
            f"{case_type} Recall@{args.top_k}: {metrics.recall_at_k:.4f} "
            f"({metrics.hits}/{metrics.total})"
        )
        print(f"{case_type} MRR: {metrics.mrr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
