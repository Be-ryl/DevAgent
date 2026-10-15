"""Minimal reciprocal-rank fusion over BM25 and Dense Retrieval."""

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from devagent.rag.service import RetrievalHit


DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60


class HybridRetrievalError(RuntimeError):
    """Raised when one of the Hybrid retrieval routes fails."""


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
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="hybrid-retrieval",
        )
        self._state_lock = Lock()
        self._closed = False

    def close(self) -> None:
        """Release the reusable retrieval worker threads."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "HybridRagService":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _submit_routes(
        self,
        query: str,
    ) -> tuple[
        Future[list[RetrievalHit]],
        Future[list[RetrievalHit]],
    ]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("HybridRagService is closed")
            bm25_future = self._executor.submit(
                self.bm25_service.search,
                query,
                top_k=self.candidate_top_k,
            )
            dense_future = self._executor.submit(
                self.dense_service.search,
                query,
                top_k=self.candidate_top_k,
            )
        return bm25_future, dense_future

    @staticmethod
    def _route_result(
        route: str,
        future: Future[list[RetrievalHit]],
    ) -> list[RetrievalHit]:
        try:
            return future.result()
        except Exception as error:
            raise HybridRetrievalError(f"{route} retrieval failed") from error

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

        bm25_future, dense_future = self._submit_routes(query)
        wait((bm25_future, dense_future))
        bm25_hits = self._route_result("BM25", bm25_future)
        dense_hits = self._route_result("Dense", dense_future)
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
