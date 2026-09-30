"""Minimal DeepSeek text generation provider."""

import os

from openai import OpenAI


def generate_text(prompt: str, model: str = "deepseek-chat") -> str:
    """Generate a text response with DeepSeek's OpenAI-compatible API."""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""

