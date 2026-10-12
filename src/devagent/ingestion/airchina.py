"""Import the latest declared Air China policies into Markdown."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


OFFICIAL_HOSTS = frozenset({"www.airchina.com.cn"})
USER_AGENT = "DevAgentPolicyImporter/0.1 (+local educational project)"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_RETRIES = 2


class AirChinaIngestionError(RuntimeError):
    """Raised when a policy cannot be safely downloaded or extracted."""


@dataclass(frozen=True)
class AirChinaSource:
    """One explicitly allowed policy page and its expected latest version."""

    id: str
    title: str
    url: str
    version: str
    content_attribute: str
    minimum_characters: int = 500
    search_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportedPolicy:
    """A cleaned policy ready to be persisted as Markdown."""

    source: AirChinaSource
    body: str
    markdown: str
    retrieved_at: str
    used_cache: bool


def _validate_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        raise AirChinaIngestionError(
            f"URL is not an allowed Air China HTTPS source: {url}"
        )


class _OfficialRedirectHandler(HTTPRedirectHandler):
    """Reject redirects that leave the explicitly allowed official host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load_manifest(path: Path | str) -> list[AirChinaSource]:
    """Load and validate the explicit official-source manifest."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AirChinaIngestionError(f"cannot read source manifest: {error}") from error

    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or not documents:
        raise AirChinaIngestionError("source manifest must contain documents")

    sources: list[AirChinaSource] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(documents, start=1):
        if not isinstance(item, dict):
            raise AirChinaIngestionError(f"manifest document {index} must be an object")
        raw_aliases = item.get("search_aliases", [])
        if not isinstance(raw_aliases, list):
            raise AirChinaIngestionError(
                f"search_aliases must be a list for manifest document {index}"
            )
        try:
            source = AirChinaSource(
                id=item["id"],
                title=item["title"],
                url=item["url"],
                version=item["version"],
                content_attribute=item["content_attribute"],
                minimum_characters=item.get("minimum_characters", 500),
                search_aliases=tuple(raw_aliases),
            )
        except KeyError as error:
            raise AirChinaIngestionError(
                f"manifest document {index} is missing {error.args[0]}"
            ) from error

        string_values = (
            source.id,
            source.title,
            source.url,
            source.version,
            source.content_attribute,
        )
        if any(not isinstance(value, str) or not value.strip() for value in string_values):
            raise AirChinaIngestionError(
                f"manifest document {index} has an empty or invalid field"
            )
        if not re.fullmatch(r"[a-z0-9_-]+", source.id):
            raise AirChinaIngestionError(f"unsafe source id: {source.id}")
        if source.id in seen_ids:
            raise AirChinaIngestionError(f"duplicate source id: {source.id}")
        if (
            not isinstance(source.minimum_characters, int)
            or source.minimum_characters <= 0
        ):
            raise AirChinaIngestionError(
                f"minimum_characters must be positive for {source.id}"
            )
        if any(
            not isinstance(alias, str) or not alias.strip()
            for alias in source.search_aliases
        ):
            raise AirChinaIngestionError(
                f"search_aliases must contain non-empty strings for {source.id}"
            )
        _validate_official_url(source.url)
        seen_ids.add(source.id)
        sources.append(source)
    return sources


def _download(url: str, timeout: float, retries: int) -> bytes:
    """Download one official page with a bounded retry count."""
    _validate_official_url(url)
    opener = build_opener(_OfficialRedirectHandler())
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                _validate_official_url(final_url)
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise AirChinaIngestionError(f"download failed for {url}: {last_error}")


def load_html(
    source: AirChinaSource,
    cache_directory: Path | str,
    *,
    refresh: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> tuple[str, bool]:
    """Return decoded HTML, using a persistent raw-page cache when available."""
    cache_root = Path(cache_directory)
    cache_path = cache_root / f"{source.id}.html"
    if cache_path.is_file() and not refresh:
        return cache_path.read_text(encoding="utf-8"), True

    raw = _download(source.url, timeout=timeout, retries=retries)
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AirChinaIngestionError(
            f"official page is not valid UTF-8 for {source.id}"
        ) from error

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".html.tmp")
    temporary_path.write_text(html, encoding="utf-8")
    temporary_path.replace(cache_path)
    return html, False


class _ContentCandidateParser(HTMLParser):
    def __init__(self, attribute: str) -> None:
        super().__init__(convert_charrefs=True)
        self.attribute = attribute
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == self.attribute and value:
                self.candidates.append(value)


class _HtmlToMarkdownParser(HTMLParser):
    """Small, deterministic converter for the publisher's embedded policy HTML."""

    def __init__(
        self,
        table_context: str,
        search_aliases: tuple[str, ...] = (),
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0
        self.table_depth = 0
        self.table_context = table_context
        self.search_aliases = search_aliases

    def _write(self, value: str) -> None:
        if self.ignored_depth == 0:
            self.parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return
        if tag == "table":
            recent_text = "".join(self.parts)[-500:]
            context = self.table_context
            is_overbooking_table = recent_text.rfind(
                "航班超售"
            ) > recent_text.rfind("非自愿降低舱位")
            if self.search_aliases and is_overbooking_table:
                context += (
                    "；检索别名（非政策正文）："
                    + "、".join(self.search_aliases)
                )
            self._write("\n\n")
            self._write(f"**表格来源：{context}**")
            self.table_depth += 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            if not self.table_depth:
                level = min(int(tag[1]) + 1, 6)
                self._write(f"\n\n{'#' * level} ")
        elif tag in {"p", "section", "article"}:
            if not self.table_depth:
                self._write("\n\n")
        elif tag == "br":
            self._write("\n")
        elif tag == "li":
            self._write("\n- ")
        elif tag == "tr":
            self._write("\n")
        elif tag in {"th", "td"}:
            self._write("| ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            if self.ignored_depth:
                self.ignored_depth -= 1
            return
        if self.ignored_depth:
            return
        if tag == "table":
            if self.table_depth:
                self.table_depth -= 1
            self._write("\n\n")
        elif tag in {"p", "section", "article"} or re.fullmatch(r"h[1-6]", tag):
            if not self.table_depth:
                self._write("\n\n")
        elif tag == "li":
            self._write("\n")
        elif tag in {"th", "td"}:
            self._write(" | ")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        normalized = re.sub(r"\s+", " ", data)
        if normalized.strip():
            self._write(normalized)

    def markdown(self) -> str:
        lines = []
        previous_blank = True
        for raw_line in "".join(self.parts).splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if line:
                lines.append(line)
                previous_blank = False
            elif not previous_blank:
                lines.append("")
                previous_blank = True
        return "\n".join(lines).strip()


def extract_policy_body(html: str, source: AirChinaSource) -> str:
    """Extract and clean only the declared latest embedded policy body."""
    collector = _ContentCandidateParser(source.content_attribute)
    collector.feed(html)
    if not collector.candidates:
        raise AirChinaIngestionError(
            f"no {source.content_attribute!r} policy body found for {source.id}"
        )

    # The publisher orders multiple versions newest-first. We deliberately inspect
    # only the first candidate so an outdated manifest cannot select history.
    latest_candidate = collector.candidates[0]
    table_context = f"中国国际航空股份有限公司（国航）《{source.title}》"
    converter = _HtmlToMarkdownParser(table_context, source.search_aliases)
    converter.feed(latest_candidate)
    body = converter.markdown()
    visible_version_text = re.sub(r"\s+", "", body)
    expected_version = re.sub(r"\s+", "", source.version)
    if expected_version not in visible_version_text:
        raise AirChinaIngestionError(
            f"latest page body does not match declared version {source.version} "
            f"for {source.id}"
        )
    plain_length = len(re.sub(r"[\s#>|*`_-]+", "", body))
    if plain_length < source.minimum_characters:
        raise AirChinaIngestionError(
            f"policy body is too short for {source.id}: "
            f"{plain_length} < {source.minimum_characters} characters"
        )
    return body


def build_markdown(
    source: AirChinaSource,
    body: str,
    retrieved_at: str,
) -> str:
    """Store provenance in front matter without duplicating it into the body."""
    metadata = {
        "title": source.title,
        "publisher": "中国国际航空股份有限公司（国航）",
        "source_url": source.url,
        "version": source.version,
        "retrieved_at": retrieved_at,
    }
    if source.search_aliases:
        metadata["search_aliases"] = list(source.search_aliases)
    front_matter = "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in metadata.items()
    )
    return (
        f"---\n{front_matter}\n---\n\n"
        f"# {source.title}\n\n{body}\n"
    )


def prepare_policy(
    source: AirChinaSource,
    cache_directory: Path | str,
    *,
    retrieved_at: str,
    refresh: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> ImportedPolicy:
    """Download/cache, validate, clean, and package one policy."""
    html, used_cache = load_html(
        source,
        cache_directory,
        refresh=refresh,
        timeout=timeout,
        retries=retries,
    )
    body = extract_policy_body(html, source)
    return ImportedPolicy(
        source=source,
        body=body,
        markdown=build_markdown(source, body, retrieved_at),
        retrieved_at=retrieved_at,
        used_cache=used_cache,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Import all declared official policies, failing on invalid page bodies."""
    parser = argparse.ArgumentParser(
        description="Import declared Air China policies into Markdown."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args(argv)

    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries must be non-negative")

    cache_directory = args.cache_dir or args.manifest.parent / "raw"
    retrieved_at = date.today().isoformat()
    try:
        sources = load_manifest(args.manifest)
        policies = [
            prepare_policy(
                source,
                cache_directory,
                retrieved_at=retrieved_at,
                refresh=args.refresh,
                timeout=args.timeout,
                retries=args.retries,
            )
            for source in sources
        ]
    except AirChinaIngestionError as error:
        parser.exit(2, f"Error: {error}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    for policy in policies:
        output_path = args.output / f"{policy.source.id}.md"
        output_path.write_text(policy.markdown, encoding="utf-8")
        print(
            f"IMPORTED {policy.source.id} version={policy.source.version} "
            f"body_chars={len(policy.body)} cache={str(policy.used_cache).lower()} "
            f"output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
