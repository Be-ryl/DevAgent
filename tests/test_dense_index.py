"""Tests for local Dense indexing and retrieval without downloading a model."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from devagent.agent import run_agent
from devagent.core.state import RuntimeState
from devagent.rag.dense_index import (
    QUERY_INSTRUCTION,
    DenseIndexError,
    DenseRagService,
    build_dense_index,
)
from devagent.tools.registry import build_tools


class FakeEmbedder:
    """Deterministic two-dimensional encoder with observable calls."""

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        texts = list(sentences)
        self.calls.append(texts)
        vectors = []
        for text in texts:
            if self.dimension != 2:
                vectors.append([1.0] + [0.0] * (self.dimension - 1))
            elif "手术" in text or "术后" in text:
                vectors.append([3.0, 0.0])
            else:
                vectors.append([0.0, 2.0])
        return np.asarray(vectors, dtype=np.float32)


def write_knowledge(directory: Path) -> None:
    directory.mkdir()
    (directory / "baggage.md").write_text(
        "# 行李政策\n\n经济舱旅客可以免费托运行李。\n",
        encoding="utf-8",
    )
    (directory / "medical.md").write_text(
        """---
title: "伤病旅客规定"
source_url: "https://www.airchina.com.cn/medical/"
version: "2026年4月版"
retrieved_at: "2026-10-12"
---

# 伤病旅客规定

术后旅客乘机前应当咨询医生，并按规定提供医疗证明。
""",
        encoding="utf-8",
    )


def build_fixture(root: Path):
    knowledge = root / "knowledge"
    index = root / "index"
    write_knowledge(knowledge)
    build_embedder = FakeEmbedder()
    manifest = build_dense_index(
        knowledge,
        index,
        model_name="fake-model",
        embedder=build_embedder,
    )
    return knowledge, index, manifest, build_embedder


class DenseIndexTest(unittest.TestCase):
    def test_index_files_save_normalized_vectors_and_complete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, index, manifest, build_embedder = build_fixture(root)

            embeddings = np.load(index / "embeddings.npy", allow_pickle=False)
            chunks = [
                json.loads(line)
                for line in (index / "chunks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            saved_manifest = json.loads(
                (index / "manifest.json").read_text(encoding="utf-8")
            )
            index_files = {path.name for path in index.iterdir()}

        self.assertEqual(
            index_files,
            {"embeddings.npy", "chunks.jsonl", "manifest.json"},
        )
        self.assertEqual(manifest["chunk_count"], 2)
        self.assertEqual(saved_manifest["embedding_dimension"], 2)
        self.assertEqual(embeddings.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)
        self.assertEqual(len(build_embedder.calls), 1)
        self.assertEqual(len(build_embedder.calls[0]), 2)
        required = {
            "chunk_id",
            "document_id",
            "content",
            "source",
            "source_url",
            "version",
            "retrieved_at",
            "position",
        }
        self.assertTrue(all(required.issubset(chunk) for chunk in chunks))
        medical = next(chunk for chunk in chunks if chunk["source"] == "medical.md")
        self.assertEqual(
            medical["source_url"],
            "https://www.airchina.com.cn/medical/",
        )

    def test_dense_search_ranks_by_cosine_and_respects_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, index, _, _ = build_fixture(root)
            query_embedder = FakeEmbedder()
            service = DenseRagService.from_index(
                index,
                model_name="fake-model",
                embedder=query_embedder,
            )

            hits = service.search("刚做完手术能坐飞机吗", top_k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "medical.md")
        self.assertEqual(hits[0].retrieved_by, ["dense"])
        self.assertAlmostEqual(hits[0].score, 1.0)
        self.assertEqual(hits[0].version, "2026年4月版")

    def test_online_load_and_query_never_reencode_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, index, _, _ = build_fixture(root)
            query_embedder = FakeEmbedder()

            service = DenseRagService.from_index(
                index,
                model_name="fake-model",
                embedder=query_embedder,
            )
            self.assertEqual(query_embedder.calls, [])
            service.search("术后能否乘机", top_k=2)

        self.assertEqual(
            query_embedder.calls,
            [[f"{QUERY_INSTRUCTION}术后能否乘机"]],
        )

    def test_model_name_and_dimension_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, index, _, _ = build_fixture(root)

            with self.assertRaisesRegex(DenseIndexError, "model mismatch"):
                DenseRagService.from_index(
                    index,
                    model_name="different-model",
                    embedder=FakeEmbedder(),
                )
            with self.assertRaisesRegex(DenseIndexError, "dimension"):
                DenseRagService.from_index(
                    index,
                    model_name="fake-model",
                    embedder=FakeEmbedder(dimension=3),
                )

    def test_changed_source_documents_require_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            knowledge, index, _, _ = build_fixture(root)
            (knowledge / "medical.md").write_text(
                "# 已经变化\n\n新的政策正文。\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DenseIndexError, "source documents changed"):
                DenseRagService.from_index(
                    index,
                    model_name="fake-model",
                    embedder=FakeEmbedder(),
                )

    def test_dense_results_enter_agent_tool_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, index, _, _ = build_fixture(root)
            service = DenseRagService.from_index(
                index,
                model_name="fake-model",
                embedder=FakeEmbedder(),
            )
            registry = build_tools(RuntimeState(root / "workspace"), service)
            requests = []

            def fake_chat(*, messages, tools):
                requests.append(json.loads(json.dumps(messages, ensure_ascii=False)))
                if len(requests) == 1:
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_dense",
                                "type": "function",
                                "function": {
                                    "name": "knowledge_search",
                                    "arguments": json.dumps(
                                        {"query": "术后能否乘机", "top_k": 1},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    }
                return {"role": "assistant", "content": "请先咨询医生。"}

            answer = run_agent(
                "请查询术后乘机政策",
                registry,
                chat_fn=fake_chat,
            )

        self.assertEqual(answer, "请先咨询医生。")
        tool_message = requests[1][-1]
        result = json.loads(tool_message["content"])
        self.assertEqual(result["results"][0]["retrieved_by"], ["dense"])
        self.assertIn("医疗证明", result["results"][0]["content"])


if __name__ == "__main__":
    unittest.main()
