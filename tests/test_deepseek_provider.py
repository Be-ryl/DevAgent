"""Tests for the DeepSeek provider without network access."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from devagent.providers.deepseek_provider import chat, generate_text


class DeepSeekProviderTest(unittest.TestCase):
    """Tests for assistant messages returned without network access."""

    @patch("devagent.providers.deepseek_provider.OpenAI")
    def test_chat_returns_text_message_without_sending_tools(self, mock_openai) -> None:
        message = Mock()
        message.model_dump.return_value = {
            "role": "assistant",
            "content": "mock response",
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        mock_openai.return_value.chat.completions.create.return_value = response
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ):
            result = chat(messages)

        self.assertEqual(
            result,
            {"role": "assistant", "content": "mock response"},
        )
        mock_openai.assert_called_once_with(
            api_key="test-key-not-a-real-secret",
            base_url="https://api.deepseek.com",
        )
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=messages,
        )
        request = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertIs(request["messages"], messages)
        self.assertNotIn("tools", request)
        message.model_dump.assert_called_once_with(exclude_none=True)

    @patch("devagent.providers.deepseek_provider.OpenAI")
    def test_chat_passes_tools_and_preserves_tool_calls(self, mock_openai) -> None:
        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": (
                        '{"file_path":"notes/a.txt","content":"hi"}'
                    ),
                },
            }
        ]
        message = Mock()
        message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        mock_openai.return_value.chat.completions.create.return_value = response
        messages = [{"role": "user", "content": "create a note"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            }
        ]

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ):
            result = chat(messages, tools=tools)

        self.assertEqual(
            result,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
        )
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        request = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertIs(request["messages"], messages)
        self.assertIs(request["tools"], tools)

    @patch("devagent.providers.deepseek_provider.chat")
    def test_generate_text_is_a_thin_chat_wrapper(self, mock_chat) -> None:
        mock_chat.return_value = {
            "role": "assistant",
            "content": "mock response",
        }

        result = generate_text("hello", model="test-model")

        self.assertEqual(result, "mock response")
        mock_chat.assert_called_once_with(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )


if __name__ == "__main__":
    unittest.main()
