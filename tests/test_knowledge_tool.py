"""Tests for the BM25-backed knowledge_search tool contract."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from devagent.core.state import RuntimeState
from devagent.rag.service import RetrievalHit
from devagent.tools.registry import InvalidToolArgumentsError, build_tools


class KnowledgeSearchToolTest(unittest.TestCase):
    """Tests for conditional registration, schema, validation, and execution."""

    def build_registry(self, rag_service: Mock):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        state = RuntimeState(Path(temporary_directory.name) / "workspace")
        return build_tools(state, rag_service)

    def test_registered_tool_exports_schema_and_executes_search(self) -> None:
        rag_service = Mock()
        rag_service.search.return_value = [
            RetrievalHit(
                chunk_id="baggage-policy-001",
                source="baggage_policy.md",
                content="超额行李费为每公斤 60 元。",
                score=3.25,
                rank=1,
                retrieved_by=["bm25"],
            )
        ]
        registry = self.build_registry(rag_service)

        schema = next(
            item
            for item in registry.openai_schemas()
            if item["function"]["name"] == "knowledge_search"
        )
        result = registry.invoke(
            "knowledge_search",
            '{"query":"超重行李如何收费","top_k":2}',
        )

        parameters = schema["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(set(parameters["properties"]), {"query", "top_k"})
        self.assertEqual(set(parameters["required"]), {"query"})
        self.assertEqual(parameters["properties"]["top_k"]["default"], 5)
        self.assertEqual(parameters["properties"]["top_k"]["minimum"], 1)
        self.assertEqual(parameters["properties"]["top_k"]["maximum"], 10)
        rag_service.search.assert_called_once_with(
            query="超重行李如何收费",
            top_k=2,
        )
        self.assertEqual(
            result,
            {
                "ok": True,
                "query": "超重行李如何收费",
                "results": [
                    {
                        "chunk_id": "baggage-policy-001",
                        "source": "baggage_policy.md",
                        "content": "超额行李费为每公斤 60 元。",
                        "score": 3.25,
                        "rank": 1,
                        "retrieved_by": ["bm25"],
                    }
                ],
            },
        )
        json.dumps(result, ensure_ascii=False)

    def test_top_k_outside_bounds_is_rejected_before_search(self) -> None:
        rag_service = Mock()
        registry = self.build_registry(rag_service)

        for top_k in (0, 11):
            with self.subTest(top_k=top_k):
                with self.assertRaises(InvalidToolArgumentsError):
                    registry.invoke(
                        "knowledge_search",
                        {"query": "酒店取消", "top_k": top_k},
                    )

        rag_service.search.assert_not_called()

    def test_no_hits_returns_empty_results(self) -> None:
        rag_service = Mock()
        rag_service.search.return_value = []
        registry = self.build_registry(rag_service)

        result = registry.invoke(
            "knowledge_search",
            {"query": "知识库外问题"},
        )

        self.assertEqual(
            result,
            {"ok": True, "query": "知识库外问题", "results": []},
        )
        rag_service.search.assert_called_once_with(
            query="知识库外问题",
            top_k=5,
        )
