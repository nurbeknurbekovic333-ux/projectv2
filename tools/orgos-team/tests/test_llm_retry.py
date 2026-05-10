"""Tests for the configurable retry-with-backoff in LLMClient.

We don't hit the real Canopy Wave API — instead we plug a fake
AsyncOpenAI client into ``LLMClient._clients`` and observe how
``call_structured`` behaves under transient errors.

The key invariant: a transient 429 / connection error / malformed
response should NOT bring down the whole 5-minute generation. The
client retries up to ``Config.max_retries`` (default 5) with
exponential backoff capped at ``Config.retry_backoff_max``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, RateLimitError
from pydantic import BaseModel

from orgos.config import ROLES, Config
from orgos.llm import LLMClient, LLMError


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_config(
    *,
    max_retries: int = 5,
    max_response_tokens: int = 8192,
) -> Config:
    """Construct a minimal real Config without touching env vars."""
    role_api_keys = {role: "fake-key" for role in ROLES}
    role_models = {role: "fake-model" for role in ROLES}
    role_base_urls = {role: "https://fake.example/v1" for role in ROLES}
    return Config(
        base_url="https://fake.example/v1",
        default_model="fake-model",
        role_models=role_models,
        role_api_keys=role_api_keys,
        role_base_urls=role_base_urls,
        max_concurrency=2,
        temperature=0.0,
        output_dir=Path("/tmp/orgos-test-out"),
        github_token=None,
        github_owner=None,
        prompts_dir=REPO_ROOT / "prompts",
        max_retries=max_retries,
        retry_backoff_max=0.05,  # tiny so retry tests don't actually wait
        max_response_tokens=max_response_tokens,
    )


class _Toy(BaseModel):
    name: str
    n: int


# ---------- fake AsyncOpenAI ----------


@dataclasses.dataclass
class _FakeMessage:
    content: str


@dataclasses.dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclasses.dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]


def _completion(content: str, *, finish_reason: str = "stop") -> _FakeCompletion:
    return _FakeCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=content), finish_reason=finish_reason)]
    )


def _rate_limit_error() -> RateLimitError:
    """Build a real RateLimitError — its constructor takes a response."""
    request = httpx.Request("POST", "https://fake.example/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={},
        content=b'{"error":{"message":"rate limited"}}',
        request=request,
    )
    return RateLimitError(message="rate limited", response=response, body=None)


def _api_connection_error() -> APIConnectionError:
    request = httpx.Request("POST", "https://fake.example/v1/chat/completions")
    return APIConnectionError(request=request)


class _Completions:
    def __init__(self, responses: Iterable[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _FakeAsyncOpenAI:
    def __init__(self, responses: Iterable[Any]) -> None:
        self.chat = _Chat(_Completions(responses))

    async def close(self) -> None:  # mirror openai.AsyncOpenAI.close
        return None


def _install_fake(client: LLMClient, role: str, fake: _FakeAsyncOpenAI) -> None:
    """Wire a fake AsyncOpenAI into the LLMClient so it skips network."""
    client._clients[role] = fake  # type: ignore[assignment]
    # Also stash in _client_by_endpoint so close() can clean up.
    client._client_by_endpoint[("fake-key", "https://fake.example/v1")] = fake  # type: ignore[assignment]


# ---------- tests ----------


def test_retries_on_rate_limit_then_succeeds() -> None:
    """A 429 followed by a success should NOT raise."""
    async def run() -> None:
        cfg = _make_config(max_retries=4)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI(
            [
                _rate_limit_error(),
                _rate_limit_error(),
                _completion('{"name": "x", "n": 42}'),
            ]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert isinstance(out, _Toy)
        assert out.n == 42
        # All three responses consumed (2 errors + 1 success).
        assert fake.chat.completions.responses == []
        await client.close()

    asyncio.run(run())


def test_retries_on_connection_error() -> None:
    """APIConnectionError is also treated as transient."""
    async def run() -> None:
        cfg = _make_config(max_retries=3)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI(
            [
                _api_connection_error(),
                _completion('{"name": "y", "n": 1}'),
            ]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert out.n == 1
        await client.close()

    asyncio.run(run())


def test_exhausts_retries_and_raises_llm_error() -> None:
    """If every attempt fails transiently, we surface an LLMError."""
    async def run() -> None:
        cfg = _make_config(max_retries=2)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI(
            [_rate_limit_error(), _rate_limit_error()]  # never succeeds
        )
        _install_fake(client, "product", fake)

        with pytest.raises(LLMError):
            await client.call_structured(
                role="product", user_message="hi", schema=_Toy
            )
        # We attempted exactly max_retries times.
        assert fake.chat.completions.responses == []
        assert len(fake.chat.completions.calls) == 2
        await client.close()

    asyncio.run(run())


def test_invalid_json_response_retries_without_sleep() -> None:
    """A non-JSON response triggers a retry (and tokens that consumed don't matter)."""
    async def run() -> None:
        cfg = _make_config(max_retries=3)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI(
            [
                _completion("not actually json {{{"),
                _completion('{"name": "ok", "n": 7}'),
            ]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert out.n == 7
        await client.close()

    asyncio.run(run())


def test_schema_validation_failure_then_success() -> None:
    """JSON that parses but doesn't fit schema is also retried."""
    async def run() -> None:
        cfg = _make_config(max_retries=3)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI(
            [
                # Parses as JSON but is missing the required 'n' field.
                _completion('{"name": "x"}'),
                _completion('{"name": "x", "n": 9}'),
            ]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert out.n == 9
        await client.close()

    asyncio.run(run())


def test_max_retries_is_configurable_via_config() -> None:
    """Per-call default falls through to Config.max_retries."""
    async def run() -> None:
        cfg = _make_config(max_retries=7)
        assert cfg.max_retries == 7
        client = LLMClient(cfg)
        # 6 transient failures then success — should succeed because
        # config allows up to 7 attempts.
        fake = _FakeAsyncOpenAI(
            [_rate_limit_error() for _ in range(6)]
            + [_completion('{"name": "z", "n": 99}')]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert out.n == 99
        await client.close()

    asyncio.run(run())


def test_backoff_is_capped_by_config() -> None:
    """retry_backoff_max caps the per-attempt sleep delay."""
    cfg = _make_config(max_retries=5)
    client = LLMClient(cfg)
    # Attempt 10 would normally back off 2^9 = 512s; cap is 0.05s.
    delay = client._backoff(10)
    assert 0 <= delay <= cfg.retry_backoff_max + 1e-6


def test_explicit_max_retries_overrides_config() -> None:
    """call_structured(max_retries=N) wins over Config.max_retries."""
    async def run() -> None:
        cfg = _make_config(max_retries=10)
        client = LLMClient(cfg)
        # Provide 3 failures; override max_retries=2 means we'll only
        # try twice and bail.
        fake = _FakeAsyncOpenAI(
            [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()]
        )
        _install_fake(client, "product", fake)

        with pytest.raises(LLMError):
            await client.call_structured(
                role="product",
                user_message="hi",
                schema=_Toy,
                max_retries=2,
            )
        assert len(fake.chat.completions.calls) == 2
        await client.close()

    asyncio.run(run())


# ---------- max_tokens / truncation tests ----------


def test_max_tokens_defaults_to_config_value() -> None:
    """Without an explicit max_tokens, we send config.max_response_tokens."""
    async def run() -> None:
        cfg = _make_config(max_response_tokens=4321)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI([_completion('{"name": "x", "n": 1}')])
        _install_fake(client, "product", fake)

        await client.call_structured(role="product", user_message="hi", schema=_Toy)
        assert fake.chat.completions.calls[0]["max_tokens"] == 4321
        await client.close()

    asyncio.run(run())


def test_explicit_max_tokens_overrides_config() -> None:
    """A per-call max_tokens still wins."""
    async def run() -> None:
        cfg = _make_config(max_response_tokens=4321)
        client = LLMClient(cfg)
        fake = _FakeAsyncOpenAI([_completion('{"name": "x", "n": 1}')])
        _install_fake(client, "product", fake)

        await client.call_structured(
            role="product", user_message="hi", schema=_Toy, max_tokens=99
        )
        assert fake.chat.completions.calls[0]["max_tokens"] == 99
        await client.close()

    asyncio.run(run())


def test_truncated_response_logs_helpful_warning(caplog: pytest.LogCaptureFixture) -> None:
    """When the model is cut off mid-string, our warning explicitly says so."""
    import logging as _logging

    async def run() -> None:
        cfg = _make_config(max_retries=2)
        client = LLMClient(cfg)
        # First response is a truncated JSON (no closing quote/brace) and
        # finish_reason="length" — exactly what an OpenAI-compatible
        # server emits when max_tokens is hit.
        truncated = '{"name": "x", "long": "aaaaaa'
        fake = _FakeAsyncOpenAI(
            [
                _completion(truncated, finish_reason="length"),
                _completion('{"name": "x", "n": 1}'),
            ]
        )
        _install_fake(client, "product", fake)

        out = await client.call_structured(
            role="product", user_message="hi", schema=_Toy
        )
        assert out.n == 1
        await client.close()

    with caplog.at_level(_logging.WARNING, logger="orgos.llm"):
        asyncio.run(run())

    truncation_logs = [r for r in caplog.records if "TRUNCATED" in r.getMessage()]
    assert truncation_logs, f"expected a truncation warning, got: {[r.getMessage() for r in caplog.records]}"
    msg = truncation_logs[0].getMessage()
    assert "ORGOS_MAX_RESPONSE_TOKENS" in msg
    assert "finish_reason=length" in msg


def test_truncated_response_via_unterminated_string_pattern() -> None:
    """Even if finish_reason is missing, JSONDecodeError text triggers detection."""
    from orgos.llm import _looks_truncated  # type: ignore[attr-defined]

    try:
        json_module_decode_error()
    except Exception as e:  # noqa: BLE001
        assert isinstance(e, ValueError)
        # Generic Python JSONDecodeError with "Unterminated string"
        # message should be flagged even with finish_reason="unknown".
        from json import JSONDecodeError
        assert isinstance(e, JSONDecodeError)
        assert _looks_truncated(e, "unknown") is True
        # Sanity: a well-formed but invalid-schema error shouldn't false-positive.
        try:
            import json as _json
            _json.loads('{"name": "x", "n":}')
        except _json.JSONDecodeError as e2:
            # "Expecting value" — this is bad JSON but NOT truncation.
            assert _looks_truncated(e2, "stop") is False


def json_module_decode_error() -> None:
    """Helper: trigger an Unterminated-string JSONDecodeError."""
    import json as _json

    _json.loads('{"name": "x", "long": "aaaaaa')
