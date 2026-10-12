"""Tests for deterministic Air China HTML extraction and provenance."""

from pathlib import Path
import json
import tempfile
import unittest

from devagent.ingestion.airchina import (
    AirChinaIngestionError,
    AirChinaSource,
    build_markdown,
    extract_policy_body,
    load_html,
    load_manifest,
)
from devagent.rag.service import BM25RagService, load_documents
from devagent.tools.knowledge_tool import knowledge_search


FIXTURE = Path(__file__).parent / "fixtures" / "airchina_page.html"
AIRCHINA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "airchina" / "processed"


def source(version: str = "2026年4月版", minimum_characters: int = 30):
    return AirChinaSource(
        id="sample_policy",
        title="示例运输政策",
        url="https://www.airchina.com.cn/zh-CN/content/example/",
        version=version,
        content_attribute="content-data",
        minimum_characters=minimum_characters,
    )


class AirChinaExtractionTest(unittest.TestCase):
    """Exercises the extractor against a local publisher-shaped fixture."""

    def test_body_cleaning_preserves_headings_lists_and_tables(self) -> None:
        body = extract_policy_body(FIXTURE.read_text(encoding="utf-8"), source())

        self.assertIn("## 示例运输政策（2026年4月版）", body)
        self.assertIn("### 第一章 适用范围", body)
        self.assertIn("- 第一项服务规则", body)
        self.assertIn("| 等待时间 |", body)
        self.assertIn("五百元", body)
        self.assertNotIn("导航菜单", body)
        self.assertNotIn("Copyright", body)
        self.assertNotIn("display: none", body)

    def test_markdown_metadata_reaches_documents_chunks_and_hits(self) -> None:
        body = extract_policy_body(FIXTURE.read_text(encoding="utf-8"), source())
        markdown = build_markdown(source(), body, "2026-10-12")

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "sample_policy.md").write_text(markdown, encoding="utf-8")
            documents = load_documents(directory)
            service = BM25RagService(documents)
            hits = service.search("补偿金额五百元", top_k=1)

        self.assertEqual(documents[0].source_url, source().url)
        self.assertEqual(documents[0].version, "2026年4月版")
        self.assertEqual(documents[0].retrieved_at, "2026-10-12")
        self.assertTrue(all(chunk.source_url == source().url for chunk in service.chunks))
        self.assertTrue(hits)
        self.assertEqual(hits[0].source_url, source().url)
        self.assertEqual(hits[0].version, "2026年4月版")
        self.assertEqual(hits[0].retrieved_at, "2026-10-12")
        tool_result = knowledge_search(service, "补偿金额五百元", top_k=1)
        self.assertEqual(tool_result["results"][0]["source_url"], source().url)
        self.assertEqual(tool_result["results"][0]["version"], "2026年4月版")

    def test_empty_or_navigation_shell_body_is_rejected(self) -> None:
        shell = (
            '<html><div content-data="&lt;p&gt;2026年4月版&lt;/p&gt;">'
            "</div></html>"
        )

        with self.assertRaisesRegex(AirChinaIngestionError, "too short"):
            extract_policy_body(shell, source(minimum_characters=50))

    def test_historical_candidate_is_not_selected_from_later_tab(self) -> None:
        html = (
            '<div content-data="&lt;p&gt;最新版（2026年4月版）正文足够长，'
            '并且是页面第一个候选。&lt;/p&gt;"></div>'
            '<div content-data="&lt;p&gt;历史版（2024年9月版）正文足够长，'
            '但不能被选中。&lt;/p&gt;"></div>'
        )

        with self.assertRaisesRegex(AirChinaIngestionError, "declared version"):
            extract_policy_body(
                html,
                source(version="2024年9月版", minimum_characters=10),
            )

    def test_cached_html_is_used_without_a_network_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            expected = FIXTURE.read_text(encoding="utf-8")
            (cache_directory / "sample_policy.html").write_text(
                expected,
                encoding="utf-8",
            )

            html, used_cache = load_html(source(), cache_directory)

        self.assertTrue(used_cache)
        self.assertEqual(html, expected)

    def test_manifest_rejects_non_airchina_domains(self) -> None:
        payload = {
            "documents": [
                {
                    "id": "unsafe",
                    "title": "不安全来源",
                    "url": "https://example.com/policy",
                    "version": "2026年4月版",
                    "content_attribute": "content-data",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "sources.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(AirChinaIngestionError, "not an allowed"):
                load_manifest(manifest)

    def test_overbooking_query_top_hit_contains_compensation_evidence(self) -> None:
        service = BM25RagService.from_directory(AIRCHINA_DIRECTORY)

        hits = service.search("航班超售后如何补偿", top_k=3)

        self.assertTrue(hits)
        top_hit = hits[0]
        self.assertEqual(top_hit.source, "overbooking_compensation.md")
        self.assertIn("等待后续航班时间", top_hit.content)
        self.assertRegex(top_hit.content, r"代金券|可消费里程|货币")
        self.assertNotIn("source_url:", top_hit.content)
        self.assertNotIn("retrieved_at:", top_hit.content)


if __name__ == "__main__":
    unittest.main()
