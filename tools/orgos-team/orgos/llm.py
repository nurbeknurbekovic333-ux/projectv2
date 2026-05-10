"""Thin async wrapper around Canopy Wave (OpenAI-compatible) chat completions.

Agents always request structured JSON output. The wrapper:
  * loads the prompt template for a role from prompts/<role>.md
  * calls the LLM with JSON response_format
  * retries on transient errors with exponential backoff (configurable
    via ``ORGOS_MAX_RETRIES`` and ``ORGOS_RETRY_BACKOFF_MAX``)
  * validates the JSON against the requested Pydantic schema
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import TypeVar

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from .config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised after we've exhausted retries or got unrecoverable output."""


class LLMClient:
    """One client per OrgOS run.

    Holds **one underlying AsyncOpenAI client per role** so that each
    sub-agent (product, architect, backend, frontend, devops, qa, reviewer,
    security) talks to Canopy Wave under its own API key / base URL — as
    configured via ``CANOPYWAVE_API_KEY_<ROLE>`` and
    ``CANOPYWAVE_BASE_URL_<ROLE>`` (with the shared
    ``CANOPYWAVE_API_KEY`` / ``CANOPYWAVE_BASE_URL`` as fallback). All
    calls share a single semaphore so the global concurrency cap
    (``ORGOS_MAX_CONCURRENCY``) still applies across roles.

    Retry behaviour is driven by ``Config.max_retries`` (default 5) and
    ``Config.retry_backoff_max`` (default 30s) — set via the
    ``ORGOS_MAX_RETRIES`` and ``ORGOS_RETRY_BACKOFF_MAX`` env vars.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._prompt_cache: dict[str, str] = {}
        self._clients: dict[str, AsyncOpenAI] = {}
        self._client_by_endpoint: dict[tuple[str, str], AsyncOpenAI] = {}

    def _client_for(self, role: str) -> AsyncOpenAI:
        cached = self._clients.get(role)
        if cached is not None:
            return cached
        api_key = self.config.api_key_for(role)
        base_url = self.config.base_url_for(role)
        # Reuse one underlying httpx client across roles that share the same
        # (key, base_url) pair so we don't open redundant connection pools
        # when the user keeps the shared CANOPYWAVE_API_KEY.
        endpoint = (api_key, base_url)
        existing = self._client_by_endpoint.get(endpoint)
        if existing is None:
            existing = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self._client_by_endpoint[endpoint] = existing
        self._clients[role] = existing
        return existing

    @property
    def client(self) -> AsyncOpenAI:
        """Back-compat: returns the underlying client for an arbitrary role.

        Prefer :meth:`_client_for` (or just call ``call_structured``) so the
        correct per-role key/base URL is used.
        """
        any_role = next(iter(self.config.role_api_keys))
        return self._client_for(any_role)

    def load_prompt(self, role: str) -> str:
        if role in self._prompt_cache:
            return self._prompt_cache[role]
        path: Path = self.config.prompts_dir / f"{role}.md"
        if not path.exists():
            raise LLMError(f"Prompt file not found for role '{role}': {path}")
        text = path.read_text(encoding="utf-8")
        self._prompt_cache[role] = text
        return text

    async def call_structured(
        self,
        *,
        role: str,
        user_message: str,
        schema: type[T],
        max_retries: int | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Call the model, parse JSON, validate against `schema`, return instance.

        Retries up to ``max_retries`` (default: ``config.max_retries``) on:
          * transient transport / API errors (rate-limit, connection, 5xx)
            — with exponential backoff + jitter, capped at
            ``config.retry_backoff_max`` seconds;
          * malformed JSON in the response body (no sleep);
          * JSON that doesn't validate against ``schema`` (no sleep).

        ``max_tokens`` defaults to ``Config.max_response_tokens`` (8192)
        so big implementer outputs don't get silently truncated by a
        small server-side default — a truncated response shows up as a
        ``json.JSONDecodeError("Unterminated string")`` and we log a
        loud warning telling the operator how to fix it.
        """

        if max_retries is None:
            max_retries = self.config.max_retries
        max_retries = max(1, max_retries)

        if max_tokens is None:
            max_tokens = self.config.max_response_tokens

        system_prompt = self.load_prompt(role)
        model = self.config.model_for(role)

        # Append a short JSON-shape hint so even models without strict
        # response_format support emit valid JSON.
        json_hint = (
            "\n\nRespond ONLY with a single JSON object that matches this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            "Do not wrap it in markdown fences. Do not add commentary."
        )
        full_system = system_prompt + json_hint

        client = self._client_for(role)

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            response = None
            async with self._semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": full_system},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"},
                        max_tokens=max_tokens,
                    )
                except (RateLimitError, APIConnectionError, APIError) as e:
                    last_err = e
                    if attempt == max_retries:
                        break
                    delay = self._backoff(attempt)
                    logger.warning(
                        "LLM call %s failed (%s). Attempt %d/%d. Retrying in %.1fs",
                        role,
                        type(e).__name__,
                        attempt,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            content = response.choices[0].message.content or ""
            finish_reason = _finish_reason(response)
            try:
                parsed = json.loads(_strip_fences(content))
            except json.JSONDecodeError as e:
                last_err = e
                if _looks_truncated(e, finish_reason):
                    logger.warning(
                        "LLM call %s was TRUNCATED at ~%d chars (finish_reason=%s, "
                        "max_tokens=%d). Bump ORGOS_MAX_RESPONSE_TOKENS in .env, "
                        "or simplify the project idea so the architect plans "
                        "fewer files per domain. Attempt %d/%d.",
                        role,
                        len(content),
                        finish_reason,
                        max_tokens,
                        attempt,
                        max_retries,
                    )
                else:
                    logger.warning(
                        "LLM call %s returned non-JSON (attempt %d/%d): %s",
                        role,
                        attempt,
                        max_retries,
                        content[:200],
                    )
                continue

            try:
                return schema.model_validate(parsed)
            except ValidationError as e:
                last_err = e
                logger.warning(
                    "LLM call %s failed schema validation (attempt %d/%d): %s",
                    role,
                    attempt,
                    max_retries,
                    e,
                )
                continue

        raise LLMError(
            f"Role '{role}' exhausted retries (last error: {last_err})"
        )

    def _backoff(self, attempt: int) -> float:
        base = 2.0 ** (attempt - 1)
        jitter = random.uniform(0, 0.5)
        return min(base + jitter, self.config.retry_backoff_max)

    async def close(self) -> None:
        # _client_by_endpoint holds the unique underlying clients; closing
        # via _clients would close the same client multiple times.
        for c in self._client_by_endpoint.values():
            await c.close()
        self._clients.clear()
        self._client_by_endpoint.clear()


def _strip_fences(text: str) -> str:
    """Some models still wrap JSON in ``` fences; strip them defensively."""
    s = text.strip()
    if s.startswith("```"):
        # find the first newline after the opening fence
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _finish_reason(response: object) -> str:
    """Best-effort extraction of OpenAI's ``finish_reason`` field.

    Returns ``"unknown"`` if the response object doesn't expose one
    (e.g. our test fakes don't bother).
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "unknown"
    first = choices[0]
    return getattr(first, "finish_reason", None) or "unknown"


def _looks_truncated(err: json.JSONDecodeError, finish_reason: str) -> bool:
    """Heuristic: did this JSON fail because the model hit max_tokens?

    The two strong signals:
      * ``finish_reason == "length"`` (OpenAI-compatible servers set
        this when generation was cut by max_tokens);
      * the JSONDecodeError message starts with ``Unterminated string``
        — i.e. the model ran out of room mid-quote.
    """
    if finish_reason == "length":
        return True
    return "Unterminated string" in str(err)
