"""Deterministic Markdown loading, chunking, and Chinese BM25 retrieval."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import jieba
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class Document:
    """One source policy document."""

    document_id: str
    title: str
    source: str
    content: str
    source_url: str | None = None
    version: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One deterministic, source-aware document chunk."""

    chunk_id: str
    document_id: str
    source: str
    content: str
    position: int
    source_url: str | None = None
    version: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked BM25 result."""

    chunk_id: str
    source: str
    content: str
    score: float
    rank: int
    retrieved_by: list[str]
    source_url: str | None = None
    version: str | None = None
    retrieved_at: str | None = None


def _front_matter(content: str) -> tuple[dict[str, str], str]:
    """Separate importer metadata from the searchable Markdown body."""
    match = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", content, re.DOTALL)
    if match is None:
        return {}, content

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if isinstance(value, str):
            metadata[key] = value
    return metadata, content[match.end() :].lstrip("\n")


def _document_title(content: str, fallback: str) -> str:
    """Return the first Markdown heading or a filename-based fallback."""
    for line in content.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if match:
            return match.group(1)
    return fallback


def load_documents(directory: Path | str) -> list[Document]:
    """Load top-level Markdown files in a stable order."""
    root = Path(directory).expanduser().resolve()
    documents: list[Document] = []

    for path in sorted(root.glob("*.md"), key=lambda item: item.name):
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue

        raw_content = path.read_text(encoding="utf-8")
        metadata, content = _front_matter(raw_content)
        source = path.relative_to(root).as_posix()
        documents.append(
            Document(
                document_id=path.stem,
                title=metadata.get("title") or _document_title(content, path.stem),
                source=source,
                content=content,
                source_url=metadata.get("source_url"),
                version=metadata.get("version"),
                retrieved_at=metadata.get("retrieved_at"),
            )
        )

    return documents


def chunk_document(document: Document, max_chunk_chars: int = 600) -> list[Chunk]:
    """Merge headings into following text, then split oversized content."""
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")

    raw_blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", document.content)
        if block.strip()
    ]
    blocks: list[str] = []
    pending_headings: list[str] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        is_heading = bool(lines) and all(
            re.match(r"^#{1,6}\s+\S", line) for line in lines
        )
        if is_heading:
            pending_headings.append(block)
            continue
        if pending_headings:
            block = "\n\n".join([*pending_headings, block])
            pending_headings.clear()
        blocks.append(block)

    # A heading with no following body is navigation, not retrievable evidence.
    contents: list[str] = []
    for block in blocks:
        contents.extend(
            block[start : start + max_chunk_chars]
            for start in range(0, len(block), max_chunk_chars)
        )

    chunks: list[Chunk] = []
    for position, content in enumerate(contents):
        identity = f"{document.document_id}\0{position}\0{content}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}-{digest}",
                document_id=document.document_id,
                source=document.source,
                content=content,
                position=position,
                source_url=document.source_url,
                version=document.version,
                retrieved_at=document.retrieved_at,
            )
        )
    return chunks


def chunk_documents(
    documents: Sequence[Document],
    max_chunk_chars: int = 600,
) -> list[Chunk]:
    """Chunk documents while preserving their deterministic input order."""
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_chunk_chars=max_chunk_chars)
    ]


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text consistently and discard punctuation-only tokens."""
    return [
        token
        for raw_token in jieba.lcut(text, cut_all=False)
        if (token := raw_token.strip().lower())
        and any(character.isalnum() for character in token)
    ]


class BM25RagService:
    """In-memory BM25 index over deterministic policy chunks."""

    def __init__(
        self,
        documents: Sequence[Document],
        max_chunk_chars: int = 600,
    ) -> None:
        self.documents = list(documents)
        candidate_chunks = chunk_documents(
            self.documents,
            max_chunk_chars=max_chunk_chars,
        )
        tokenized_pairs = [
            (chunk, tokens)
            for chunk in candidate_chunks
            if (tokens := tokenize(chunk.content))
        ]
        self.chunks = [chunk for chunk, _ in tokenized_pairs]
        self._tokenized_chunks = [tokens for _, tokens in tokenized_pairs]
        self._index = (
            BM25Okapi(self._tokenized_chunks)
            if self._tokenized_chunks
            else None
        )

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        max_chunk_chars: int = 600,
    ) -> "BM25RagService":
        """Load a policy directory and build its BM25 index."""
        return cls(
            load_documents(directory),
            max_chunk_chars=max_chunk_chars,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return positive-score chunks ordered by BM25 relevance."""
        query_tokens = tokenize(query)
        if not query_tokens or top_k <= 0 or self._index is None:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked = sorted(
            (
                (float(score), chunk)
                for score, chunk in zip(scores, self.chunks, strict=True)
                if score > 0
            ),
            key=lambda item: (-item[0], item[1].chunk_id),
        )[:top_k]

        return [
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                content=chunk.content,
                score=score,
                rank=rank,
                retrieved_by=["bm25"],
                source_url=chunk.source_url,
                version=chunk.version,
                retrieved_at=chunk.retrieved_at,
            )
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run a small local BM25 retrieval demonstration."""
    parser = argparse.ArgumentParser(description="Search local policy Markdown with BM25.")
    parser.add_argument("directory", type=Path, help="policy Markdown directory")
    parser.add_argument("query", help="query text")
    parser.add_argument("--top-k", type=int, default=5, help="maximum results")
    args = parser.parse_args(argv)

    service = BM25RagService.from_directory(args.directory)
    hits = service.search(args.query, top_k=args.top_k)
    print(f"Query: {args.query}")
    if not hits:
        print("No positive-score BM25 matches.")
        return 0

    for hit in hits:
        metadata = ""
        if hit.version:
            metadata += f" version={hit.version}"
        if hit.source_url:
            metadata += f" source_url={hit.source_url}"
        print(
            f"\n#{hit.rank} source={hit.source} score={hit.score:.6f}{metadata}"
        )
        print(hit.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
