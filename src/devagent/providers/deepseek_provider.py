"""Minimal DeepSeek chat provider."""

import os
from typing import Any

from openai import OpenAI


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return one complete assistant message from DeepSeek."""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
    )
    request: dict[str, Any] = {
        "model": model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": messages,
    }
    if tools is not None:
        request["tools"] = tools

    response = client.chat.completions.create(**request)
    assistant_message = response.choices[0].message.model_dump(exclude_none=True)
    assistant_message.setdefault("role", "assistant")
    assistant_message.setdefault("content", None)
    return assistant_message


def generate_text(prompt: str, model: str | None = None) -> str:
    """Generate plain text through ``chat`` for backward compatibility."""
    assistant_message = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return assistant_message.get("content") or ""

