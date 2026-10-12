"""Adapter that exposes local retrieval as a JSON-serializable tool."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from devagent.rag.dense_index import DenseRagService
    from devagent.rag.hybrid import HybridRagService
    from devagent.rag.service import BM25RagService


def knowledge_search(
    rag_service: "BM25RagService | DenseRagService | HybridRagService",
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Search local policies and return evidence without generating an answer."""
    hits = rag_service.search(query=query, top_k=top_k)
    return {
        "ok": True,
        "query": query,
        "results": [
            {
                key: value
                for key, value in asdict(hit).items()
                if value is not None
            }
            for hit in hits
        ],
    }
