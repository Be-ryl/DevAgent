"""Tests for shared DeepSeek model configuration."""

import os
import unittest
from unittest.mock import patch

from devagent.providers.model_provider import (
    DeepSeekModelProvider,
    DeepSeekSettings,
)


class DeepSeekModelProviderTest(unittest.TestCase):
    @patch("devagent.providers.model_provider.ChatOpenAI")
    def test_all_roles_receive_identical_model_parameters(
        self,
        mock_chat_openai,
    ) -> None:
        provider = DeepSeekModelProvider(
            DeepSeekSettings(
                api_key="fake-key",
                base_url="https://deepseek.example/v1",
                model="deepseek-test",
            )
        )

        provider.create_model("single", 12.0)
        provider.create_model("planner", 12.0)

        expected = {
            "api_key": "fake-key",
            "base_url": "https://deepseek.example/v1",
            "model": "deepseek-test",
            "temperature": 0,
            "timeout": 12.0,
        }
        self.assertEqual(mock_chat_openai.call_count, 2)
        self.assertEqual(mock_chat_openai.call_args_list[0].kwargs, expected)
        self.assertEqual(mock_chat_openai.call_args_list[1].kwargs, expected)

    def test_settings_read_all_supported_environment_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "fake-key",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
                "DEEPSEEK_MODEL": "deepseek-test",
            },
            clear=True,
        ), patch("devagent.providers.model_provider.load_dotenv"):
            settings = DeepSeekSettings.from_environment()

        self.assertEqual(settings.api_key, "fake-key")
        self.assertEqual(settings.base_url, "https://deepseek.example/v1")
        self.assertEqual(settings.model, "deepseek-test")


if __name__ == "__main__":
    unittest.main()
