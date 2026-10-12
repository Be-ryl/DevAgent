"""Tests for minimal BM25 + Dense reciprocal-rank fusion."""

import json
from pathlib import Path
import tempfile
import unittest

from devagent.agent import run_agent
from devagent.core.state import RuntimeState
from devagent.rag.hybrid import HybridRagService
from devagent.rag.service import RetrievalHit
from devagent.tools.registry import build_tools


def hit(
    chunk_id: str,
    route: str,
    rank: int,
    *,
    source: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source=source or f"{chunk_id}.md",
        content=f"{chunk_id} 的政策正文",
        score=float(10 - rank),
        rank=rank,
        retrieved_by=[route],
        source_url=f"https://example.test/{chunk_id}",
        version="test-version",
        retrieved_at="2026-10-13",
    )


class StaticService:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class HybridRagServiceTest(unittest.TestCase):
    def make_service(self) -> tuple[HybridRagService, StaticService, StaticService]:
        bm25 = StaticService(
            [
                hit("shared", "bm25", 1),
                hit("bm25-only", "bm25", 2),
                hit("tail", "bm25", 3),
            ]
        )
        dense = StaticService(
            [
                hit("shared", "dense", 1),
                hit("dense-only", "dense", 2),
                hit("tail", "dense", 3),
            ]
        )
        return HybridRagService(bm25, dense), bm25, dense

    def test_fusion_keeps_single_route_hits_and_deduplicates_shared_chunks(self) -> None:
        service, bm25, dense = self.make_service()

        hits = service.search("退票规则", top_k=10)

        self.assertEqual(len(hits), 4)
        self.assertEqual(len({item.chunk_id for item in hits}), 4)
        by_id = {item.chunk_id: item for item in hits}
        self.assertEqual(by_id["bm25-only"].retrieved_by, ["bm25"])
        self.assertEqual(by_id["bm25-only"].bm25_rank, 2)
        self.assertIsNone(by_id["bm25-only"].dense_rank)
        self.assertEqual(by_id["dense-only"].retrieved_by, ["dense"])
        self.assertIsNone(by_id["dense-only"].bm25_rank)
        self.assertEqual(by_id["dense-only"].dense_rank, 2)
        self.assertEqual(by_id["shared"].retrieved_by, ["bm25", "dense"])
        self.assertEqual(by_id["shared"].bm25_rank, 1)
        self.assertEqual(by_id["shared"].dense_rank, 1)
        self.assertEqual(bm25.calls, [("退票规则", 20)])
        self.assertEqual(dense.calls, [("退票规则", 20)])

    def test_double_high_rank_has_higher_rrf_score(self) -> None:
        service, _, _ = self.make_service()

        hits = service.search("退票规则", top_k=10)
        by_id = {item.chunk_id: item for item in hits}

        self.assertEqual(hits[0].chunk_id, "shared")
        self.assertAlmostEqual(by_id["shared"].rrf_score, 2 / 61)
        self.assertGreater(
            by_id["shared"].rrf_score,
            by_id["bm25-only"].rrf_score,
        )
        self.assertEqual(by_id["shared"].score, by_id["shared"].rrf_score)

    def test_top_k_and_sorting_are_deterministic(self) -> None:
        service, _, _ = self.make_service()

        first = service.search("退票规则", top_k=2)
        second = service.search("退票规则", top_k=2)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual([item.rank for item in first], [1, 2])

    def test_hybrid_results_enter_tool_message_and_trace(self) -> None:
        service, _, _ = self.make_service()
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = build_tools(
                RuntimeState(Path(temporary_directory) / "workspace"),
                service,
            )
            requests: list[list[dict]] = []

            def fake_chat(*, messages, tools):
                requests.append(json.loads(json.dumps(messages, ensure_ascii=False)))
                if len(requests) == 1:
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_hybrid",
                                "type": "function",
                                "function": {
                                    "name": "knowledge_search",
                                    "arguments": json.dumps(
                                        {"query": "退票规则", "top_k": 2},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    }
                return {"role": "assistant", "content": "融合检索完成。"}

            trace_events: list[str] = []
            answer = run_agent(
                "请查询退票规则",
                registry,
                chat_fn=fake_chat,
                trace_fn=trace_events.append,
            )

        self.assertEqual(answer, "融合检索完成。")
        result = json.loads(requests[1][-1]["content"])
        first_hit = result["results"][0]
        self.assertEqual(first_hit["chunk_id"], "shared")
        self.assertEqual(first_hit["bm25_rank"], 1)
        self.assertEqual(first_hit["dense_rank"], 1)
        self.assertEqual(first_hit["retrieved_by"], ["bm25", "dense"])
        self.assertIn("rrf_score", first_hit)
        self.assertRegex(
            trace_events[1],
            r"^\[retrieval_hit\] rank=1 chunk_id=shared "
            r"bm25_rank=1 dense_rank=1 rrf_score=0\.032787 "
            r"source=shared\.md content_preview=",
        )


if __name__ == "__main__":
    unittest.main()
