"""Local NumPy-backed Dense Retrieval index and search service."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Protocol
import uuid

import numpy as np

from devagent.rag.service import (
    Chunk,
    RetrievalHit,
    chunk_documents,
    load_documents,
)


DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_MAX_CHUNK_CHARS = 600


class DenseIndexError(RuntimeError):
    """Raised when a Dense index is invalid, stale, or incompatible."""


class DenseDependencyError(DenseIndexError):
    """Raised when the optional sentence-transformers dependency is absent."""


class Embedder(Protocol):
    """Small subset of SentenceTransformer used by the index."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any:
        """Encode text into one vector per sentence."""

    def get_sentence_embedding_dimension(self) -> int | None:
        """Return the model output dimension when known."""


def _load_embedder(model_name: str) -> Embedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise DenseDependencyError(
            "Dense Retrieval requires optional dependencies; "
            "install the project with `pip install -e '.[dense]'`"
        ) from error
    return SentenceTransformer(model_name)


def _encode(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    vectors = embedder.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise DenseIndexError(
            "embedder returned an invalid shape: "
            f"expected ({len(texts)}, dimension), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise DenseIndexError("embedder returned non-finite vector values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise DenseIndexError("embedder returned a zero-length vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _source_files_hash(directory: Path | str) -> str:
    root = Path(directory).expanduser().resolve()
    digest = hashlib.sha256()
    paths = sorted(root.glob("*.md"), key=lambda path: path.name)
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _replace_directory(staged: Path, destination: Path) -> None:
    """Publish a complete staged index with rollback for an existing index."""
    backup: Path | None = None
    if destination.exists():
        if not destination.is_dir():
            raise DenseIndexError(f"index path is not a directory: {destination}")
        backup = destination.with_name(
            f".{destination.name}.backup-{uuid.uuid4().hex}"
        )
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_dense_index(
    knowledge_directory: Path | str,
    index_directory: Path | str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Encode the same cleaned chunks used by BM25 and atomically persist them."""
    if max_chunk_chars <= 0:
        raise DenseIndexError("max_chunk_chars must be positive")
    knowledge_root = Path(knowledge_directory).expanduser().resolve()
    if not knowledge_root.is_dir():
        raise DenseIndexError(
            f"knowledge directory is not a directory: {knowledge_root}"
        )

    documents = load_documents(knowledge_root)
    chunks = chunk_documents(documents, max_chunk_chars=max_chunk_chars)
    if not chunks:
        raise DenseIndexError("knowledge directory produced no searchable chunks")

    model = embedder if embedder is not None else _load_embedder(model_name)
    embeddings = _encode(model, [chunk.content for chunk in chunks])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "chunking": {"max_chunk_chars": max_chunk_chars},
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(knowledge_root),
        "source_documents_hash": _source_files_hash(knowledge_root),
    }

    destination = Path(index_directory).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        with (staged / "embeddings.npy").open("wb") as embedding_file:
            np.save(embedding_file, embeddings, allow_pickle=False)
        with (staged / "chunks.jsonl").open("w", encoding="utf-8") as chunk_file:
            for chunk in chunks:
                chunk_file.write(
                    json.dumps(asdict(chunk), ensure_ascii=False) + "\n"
                )
        _write_json(staged / "manifest.json", manifest)
        _replace_directory(staged, destination)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return manifest


def _load_manifest(index_directory: Path) -> dict[str, Any]:
    manifest_path = index_directory / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenseIndexError(f"cannot read Dense index manifest: {error}") from error
    if not isinstance(payload, dict):
        raise DenseIndexError("Dense index manifest must be a JSON object")
    return payload


def _load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    try:
        with path.open("r", encoding="utf-8") as chunk_file:
            for line_number, line in enumerate(chunk_file, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunks.append(Chunk(**payload))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise DenseIndexError(f"cannot read Dense chunks: {error}") from error
    return chunks


class DenseRagService:
    """In-memory cosine search over a validated local NumPy index."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        model_name: str,
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.embedder = embedder
        self.model_name = model_name

    @classmethod
    def from_index(
        cls,
        index_directory: Path | str,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        embedder: Embedder | None = None,
        knowledge_directory: Path | str | None = None,
    ) -> "DenseRagService":
        """Validate an index, source hash, model, and dimensions before use."""
        root = Path(index_directory).expanduser().resolve()
        if not root.is_dir():
            raise DenseIndexError(f"Dense index directory does not exist: {root}")
        manifest = _load_manifest(root)
        if manifest.get("schema_version") != 1:
            raise DenseIndexError(
                "unsupported Dense index schema; rebuild the index"
            )
        indexed_model = manifest.get("model_name")
        if indexed_model != model_name:
            raise DenseIndexError(
                f"Dense index model mismatch: index uses {indexed_model!r}, "
                f"requested {model_name!r}; rebuild the index"
            )

        required_integer_fields = ("embedding_dimension", "chunk_count")
        if any(
            not isinstance(manifest.get(field), int) or manifest[field] <= 0
            for field in required_integer_fields
        ):
            raise DenseIndexError("Dense index manifest has invalid dimensions or count")

        chunks = _load_chunks(root / "chunks.jsonl")
        try:
            embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise DenseIndexError(f"cannot read Dense embeddings: {error}") from error
        expected_shape = (
            manifest["chunk_count"],
            manifest["embedding_dimension"],
        )
        if embeddings.shape != expected_shape or embeddings.dtype.kind != "f":
            raise DenseIndexError(
                f"Dense embedding shape mismatch: expected {expected_shape}, "
                f"got {embeddings.shape}"
            )
        if len(chunks) != manifest["chunk_count"]:
            raise DenseIndexError(
                "Dense chunk count does not match manifest; rebuild the index"
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(embeddings).all() or not np.allclose(
            norms, 1.0, atol=1e-4
        ):
            raise DenseIndexError("Dense embeddings are not finite normalized vectors")

        source_directory = Path(
            knowledge_directory or manifest.get("source_directory", "")
        ).expanduser()
        if not source_directory.is_dir():
            raise DenseIndexError(
                "source knowledge directory is unavailable; provide --knowledge-dir "
                "or rebuild the index"
            )
        current_hash = _source_files_hash(source_directory)
        if current_hash != manifest.get("source_documents_hash"):
            raise DenseIndexError(
                "source documents changed after Dense index creation; rebuild the index"
            )

        model = embedder if embedder is not None else _load_embedder(model_name)
        dimension_method = getattr(model, "get_sentence_embedding_dimension", None)
        model_dimension = dimension_method() if callable(dimension_method) else None
        if (
            model_dimension is not None
            and model_dimension != manifest["embedding_dimension"]
        ):
            raise DenseIndexError(
                "embedding model dimension does not match the Dense index; "
                "rebuild the index"
            )
        return cls(chunks, embeddings, model, model_name)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Encode only the instructed query and rank normalized chunk vectors."""
        if not query.strip() or top_k <= 0:
            return []
        query_vector = _encode(
            self.embedder,
            [f"{QUERY_INSTRUCTION}{query}"],
        )[0]
        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise DenseIndexError(
                "query vector dimension does not match the Dense index; rebuild it"
            )
        scores = self.embeddings @ query_vector
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id),
        )[:top_k]
        return [
            RetrievalHit(
                chunk_id=self.chunks[index].chunk_id,
                source=self.chunks[index].source,
                content=self.chunks[index].content,
                score=float(scores[index]),
                rank=rank,
                retrieved_by=["dense"],
                source_url=self.chunks[index].source_url,
                version=self.chunks[index].version,
                retrieved_at=self.chunks[index].retrieved_at,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]


def main(argv: Sequence[str] | None = None) -> int:
    """Build a local Dense Retrieval index."""
    parser = argparse.ArgumentParser(description="Manage local Dense indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build a Dense index")
    build_parser.add_argument("--knowledge-dir", type=Path, required=True)
    build_parser.add_argument("--index-dir", type=Path, required=True)
    build_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    build_parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        manifest = build_dense_index(
            args.knowledge_dir,
            args.index_dir,
            model_name=args.model,
            max_chunk_chars=args.max_chunk_chars,
        )
    except DenseIndexError as error:
        parser.exit(2, f"Error: {error}\n")
    elapsed = time.perf_counter() - started
    index_root = args.index_dir.expanduser().resolve()
    total_bytes = sum(
        path.stat().st_size for path in index_root.iterdir() if path.is_file()
    )
    print(
        f"BUILT chunks={manifest['chunk_count']} "
        f"dimension={manifest['embedding_dimension']} "
        f"bytes={total_bytes} elapsed_seconds={elapsed:.3f} "
        f"index={index_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
