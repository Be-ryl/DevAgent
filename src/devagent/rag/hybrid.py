"""Minimal reciprocal-rank fusion over BM25 and Dense Retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from devagent.rag.service import RetrievalHit


DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60


class RetrievalService(Protocol):
    """Search interface shared by the existing retrieval services."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return ranked hits for one query."""


@dataclass
class _FusedCandidate:
    hit: RetrievalHit
    bm25_rank: int | None = None
    dense_rank: int | None = None

    def rrf_score(self, rrf_k: int) -> float:
        return sum(
            1.0 / (rrf_k + rank)
            for rank in (self.bm25_rank, self.dense_rank)
            if rank is not None
        )


class HybridRagService:
    """Fuse BM25 and Dense candidates by reciprocal rank, keyed by chunk ID."""

    def __init__(
        self,
        bm25_service: RetrievalService,
        dense_service: RetrievalService,
        *,
        candidate_top_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.bm25_service = bm25_service
        self.dense_service = dense_service
        self.candidate_top_k = candidate_top_k
        self.rrf_k = rrf_k

    @staticmethod
    def _merge_route(
        candidates: dict[str, _FusedCandidate],
        hits: Sequence[RetrievalHit],
        route: str,
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            candidate = candidates.setdefault(
                hit.chunk_id,
                _FusedCandidate(hit=hit),
            )
            if route == "bm25" and candidate.bm25_rank is None:
                candidate.bm25_rank = rank
            elif route == "dense" and candidate.dense_rank is None:
                candidate.dense_rank = rank

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Retrieve from both routes and return deterministic RRF-ranked hits."""
        if not query.strip() or top_k <= 0:
            return []

        bm25_hits = self.bm25_service.search(
            query,
            top_k=self.candidate_top_k,
        )
        dense_hits = self.dense_service.search(
            query,
            top_k=self.candidate_top_k,
        )
        candidates: dict[str, _FusedCandidate] = {}
        self._merge_route(candidates, bm25_hits, "bm25")
        self._merge_route(candidates, dense_hits, "dense")

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.rrf_score(self.rrf_k),
                min(
                    rank
                    for rank in (candidate.bm25_rank, candidate.dense_rank)
                    if rank is not None
                ),
                candidate.hit.chunk_id,
            ),
        )[:top_k]

        results: list[RetrievalHit] = []
        for final_rank, candidate in enumerate(ranked, start=1):
            hit = candidate.hit
            rrf_score = candidate.rrf_score(self.rrf_k)
            retrieved_by = []
            if candidate.bm25_rank is not None:
                retrieved_by.append("bm25")
            if candidate.dense_rank is not None:
                retrieved_by.append("dense")
            results.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    source=hit.source,
                    content=hit.content,
                    score=rrf_score,
                    rank=final_rank,
                    retrieved_by=retrieved_by,
                    source_url=hit.source_url,
                    version=hit.version,
                    retrieved_at=hit.retrieved_at,
                    bm25_rank=candidate.bm25_rank,
                    dense_rank=candidate.dense_rank,
                    rrf_score=rrf_score,
                )
            )
        return results
