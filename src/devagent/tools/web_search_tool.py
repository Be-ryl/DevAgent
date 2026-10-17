"""Tavily-backed web search shared by the single and search agents."""

import os
from typing import Any

from dotenv import load_dotenv


def web_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = True,
) -> dict[str, Any]:
    """Search Tavily and return a stable, compact result shape."""
    load_dotenv(override=False)
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "query": query,
            "error": "TAVILY_API_KEY is not set",
        }

    from tavily import TavilyClient

    bounded_results = max(1, min(int(max_results), 10))
    try:
        response = TavilyClient(api_key=api_key).search(
            query=query,
            search_depth="basic",
            max_results=bounded_results,
            include_answer=include_answer,
        )
    except Exception as error:
        return {
            "ok": False,
            "query": query,
            "error": f"{type(error).__name__}: {error}",
        }

    results = [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", ""))[:1200],
            "score": item.get("score"),
        }
        for item in response.get("results", []) or []
    ]
    return {
        "ok": True,
        "query": query,
        "answer": str(response.get("answer") or ""),
        "results": results,
    }
