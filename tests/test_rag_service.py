"""Tests for Markdown chunking and Chinese BM25 retrieval."""

from pathlib import Path
import tempfile
import unittest

from devagent.rag.service import (
    BM25RagService,
    Document,
    chunk_document,
    load_documents,
)


POLICY_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "policies"


class DocumentAndChunkTest(unittest.TestCase):
    """Tests for deterministic Markdown loading and chunking."""

    def test_markdown_documents_load_in_stable_order_with_metadata(self) -> None:
        documents = load_documents(POLICY_DIRECTORY)

        self.assertEqual(
            [document.source for document in documents],
            [
                "baggage_policy.md",
                "coupon_policy.md",
                "customer_service.md",
                "flight_change.md",
                "flight_refund.md",
                "hotel_cancellation.md",
                "hotel_checkin.md",
                "invoice_policy.md",
            ],
        )
        self.assertEqual(
            [document.document_id for document in documents],
            [Path(document.source).stem for document in documents],
        )
        self.assertTrue(all(document.title for document in documents))

    def test_empty_markdown_file_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "empty.md").write_text("", encoding="utf-8")

            service = BM25RagService.from_directory(directory)

        self.assertEqual(len(service.documents), 1)
        self.assertEqual(service.documents[0].source, "empty.md")
        self.assertEqual(service.chunks, [])
        self.assertEqual(service.search("anything"), [])

    def test_chunk_ids_are_stable_and_metadata_is_preserved(self) -> None:
        document = Document(
            document_id="hotel-test",
            title="酒店测试",
            source="hotel_test.md",
            content="# 酒店测试\n\n入住时间为下午两点。",
        )

        first = chunk_document(document, max_chunk_chars=100)
        second = chunk_document(document, max_chunk_chars=100)

        self.assertEqual(first, second)
        self.assertEqual(
            [chunk.position for chunk in first],
            list(range(len(first))),
        )
        self.assertTrue(all(chunk.document_id == "hotel-test" for chunk in first))
        self.assertTrue(all(chunk.source == "hotel_test.md" for chunk in first))

    def test_oversized_paragraph_is_split_by_character_count(self) -> None:
        document = Document(
            document_id="long-policy",
            title="超长政策",
            source="long_policy.md",
            content="# 超长政策\n\n" + "行李规定" * 70,
        )

        chunks = chunk_document(document, max_chunk_chars=80)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk.content) <= 80 for chunk in chunks))


class BM25RagServiceTest(unittest.TestCase):
    """Tests for local Chinese BM25 ranking behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = BM25RagService.from_directory(POLICY_DIRECTORY)

    def test_chinese_query_hits_hotel_cancellation_policy(self) -> None:
        hits = self.service.search("酒店入住前一天取消是否收取手续费？")

        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "hotel_cancellation.md")
        self.assertEqual(hits[0].retrieved_by, ["bm25"])
        self.assertEqual(hits[0].rank, 1)

    def test_exact_fee_and_weight_terms_recall_baggage_policy(self) -> None:
        hits = self.service.search("托运行李超过20公斤每公斤收费60元")

        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "baggage_policy.md")
        self.assertIn("60", hits[0].content)

    def test_top_k_limits_result_count_and_assigns_ranks(self) -> None:
        hits = self.service.search("取消 退款 费用 政策", top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertEqual([hit.rank for hit in hits], [1, 2])

    def test_empty_directory_returns_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = BM25RagService.from_directory(temporary_directory)

        self.assertEqual(service.documents, [])
        self.assertEqual(service.chunks, [])
        self.assertEqual(service.search("酒店取消"), [])

    def test_empty_query_returns_no_results(self) -> None:
        self.assertEqual(self.service.search("   "), [])

    def test_unrelated_query_does_not_return_zero_score_chunks(self) -> None:
        hits = self.service.search("量子芯片光刻机超导实验")

        self.assertEqual(hits, [])

    def test_ranking_is_deterministic(self) -> None:
        other_service = BM25RagService.from_directory(POLICY_DIRECTORY)
        first = self.service.search("机票退票手续费", top_k=5)
        second = other_service.search("机票退票手续费", top_k=5)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
