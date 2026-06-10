"""LLM provider abstraction.

Supports Anthropic, OpenAI, and a deterministic Mock provider.
The active provider is selected by the LLM_PROVIDER environment variable.
All providers share the same interface so the rest of the codebase is
provider-agnostic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base protocol / ABC
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Minimal interface every LLM provider must satisfy."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous text completion."""

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async text completion — default delegates to synchronous version."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.complete, system_prompt, user_prompt
        )

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic  # type: ignore

        async_client = anthropic.AsyncAnthropic(api_key=self._client.api_key)
        msg = await async_client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
            self._async_client = AsyncOpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        resp = await self._async_client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Mock (offline / CI)
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    """Deterministic mock — returns a structured stub answer that reflects context.

    Safe for CI pipelines, local demos, and tests.
    No API key or internet connection required.

    The mock inspects the user_prompt to extract:
    - Number of context passages provided
    - A short excerpt from the first passage (so demo output is non-trivial)
    - The question text (extracted after "Question:")
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Extract question
        question = ""
        if "Question:" in user_prompt:
            question = user_prompt.split("Question:")[-1].strip()[:120]

        # Count context passages by counting [p.N] markers
        import re as _re
        pages = _re.findall(r"\[p\.(\w+)\]", user_prompt)
        n_passages = len(pages)
        page_refs = ", ".join(f"p.{p}" for p in pages[:3])
        if len(pages) > 3:
            page_refs += f" (and {len(pages) - 3} more)"

        # Lift first ~150 chars of first passage as excerpt
        excerpt = ""
        ctx_match = _re.search(r"\[p\.\w+\]\s*(.{20,150})", user_prompt)
        if ctx_match:
            excerpt = f' Key excerpt: "{ctx_match.group(1).strip()[:120]}…"'

        if n_passages == 0:
            return (
                f"[MOCK ANSWER] No relevant passages were retrieved for the question: "
                f"'{question}'. The corpus may not contain information on this topic. "
                f"(Mock provider — set LLM_PROVIDER=anthropic or openai for real answers.)"
            )

        return (
            f"[MOCK ANSWER] Based on {n_passages} retrieved passage(s) "
            f"({page_refs}), the policy corpus contains relevant information "
            f"about: '{question}'.{excerpt} "
            f"For a complete, synthesised answer, configure a real LLM provider "
            f"via LLM_PROVIDER=anthropic or LLM_PROVIDER=openai."
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock-model"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llm_provider(
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    anthropic_api_key: str = "",
    openai_api_key: str = "",
) -> BaseLLMProvider:
    if provider == "anthropic":
        return AnthropicProvider(anthropic_api_key, model, temperature, max_tokens)
    if provider == "openai":
        return OpenAIProvider(openai_api_key, model, temperature, max_tokens)
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r}. Choose anthropic|openai|mock.")
