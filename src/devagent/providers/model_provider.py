"""Shared DeepSeek model provider for all execution modes and roles."""

from dataclasses import dataclass
import os
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class ModelProvider(Protocol):
    """Factory contract used by runners and easily replaced by fake models."""

    def create_model(self, role: str, timeout_seconds: float) -> Any:
        """Create one chat model configured for a role."""


@dataclass(frozen=True)
class DeepSeekSettings:
    """OpenAI-compatible DeepSeek connection settings."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_environment(cls) -> "DeepSeekSettings":
        load_dotenv(override=False)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )


class DeepSeekModelProvider:
    """Create role-specific clients with identical model parameters."""

    def __init__(self, settings: DeepSeekSettings | None = None) -> None:
        self._settings = settings

    @property
    def settings(self) -> DeepSeekSettings:
        if self._settings is None:
            self._settings = DeepSeekSettings.from_environment()
        return self._settings

    def create_model(self, role: str, timeout_seconds: float) -> ChatOpenAI:
        del role
        settings = self.settings
        return ChatOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            temperature=0,
            timeout=timeout_seconds,
        )
