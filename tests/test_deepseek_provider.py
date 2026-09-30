"""Tests for the DeepSeek provider without network access."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from devagent.providers.deepseek_provider import generate_text


class DeepSeekProviderTest(unittest.TestCase):
    """Tests for minimal text generation."""

    @patch("devagent.providers.deepseek_provider.OpenAI")
    def test_generate_text_uses_chat_completions(self, mock_openai) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mock response")
                )
            ]
        )
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ):
            result = generate_text("hello")

        self.assertEqual(result, "mock response")
        mock_openai.assert_called_once_with(
            api_key="test-key-not-a-real-secret",
            base_url="https://api.deepseek.com",
        )
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )


if __name__ == "__main__":
    unittest.main()
