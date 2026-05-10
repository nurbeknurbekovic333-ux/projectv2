"""Tests for per-role API key / base URL configuration.

Verifies the contract that:
  * each sub-agent role gets its own API key when CANOPYWAVE_API_KEY_<ROLE>
    is set,
  * any role that has no per-role key falls back to the shared
    CANOPYWAVE_API_KEY,
  * if neither is set for a role, Config.load() refuses to start,
  * LLMClient routes each call to the correct underlying AsyncOpenAI
    client (one per unique key/base-url pair).
"""

from __future__ import annotations

import os

import pytest

# Ensure imports below don't trip on a missing shared key — individual
# tests will set the env they actually care about via monkeypatch.
os.environ.setdefault("CANOPYWAVE_API_KEY", "test-fake-key")

from orgos.config import ROLES, Config
from orgos.llm import LLMClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _clear_canopywave_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CANOPYWAVE_API_KEY",
        "CANOPYWAVE_BASE_URL",
        "ORGOS_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    for role in ROLES:
        up = role.upper()
        monkeypatch.delenv(f"CANOPYWAVE_API_KEY_{up}", raising=False)
        monkeypatch.delenv(f"CANOPYWAVE_BASE_URL_{up}", raising=False)
        monkeypatch.delenv(f"ORGOS_MODEL_{up}", raising=False)


def test_shared_key_fans_out_to_every_role(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)
    monkeypatch.setenv("CANOPYWAVE_API_KEY", "SHARED")

    cfg = Config.load()

    for role in ROLES:
        assert cfg.api_key_for(role) == "SHARED"
        assert cfg.base_url_for(role) == "https://inference.canopywave.io/v1"


def test_per_role_key_overrides_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)
    monkeypatch.setenv("CANOPYWAVE_API_KEY", "SHARED")
    monkeypatch.setenv("CANOPYWAVE_API_KEY_PRODUCT", "PROD-KEY")
    monkeypatch.setenv("CANOPYWAVE_API_KEY_REVIEWER", "REV-KEY")

    cfg = Config.load()

    assert cfg.api_key_for("product") == "PROD-KEY"
    assert cfg.api_key_for("reviewer") == "REV-KEY"
    for role in ROLES:
        if role in ("product", "reviewer"):
            continue
        assert cfg.api_key_for(role) == "SHARED", role


def test_per_role_base_url_overrides_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)
    monkeypatch.setenv("CANOPYWAVE_API_KEY", "SHARED")
    monkeypatch.setenv("CANOPYWAVE_BASE_URL", "https://primary.example/v1")
    monkeypatch.setenv("CANOPYWAVE_BASE_URL_SECURITY", "https://other.example/v1")

    cfg = Config.load()

    assert cfg.base_url_for("security") == "https://other.example/v1"
    for role in ROLES:
        if role == "security":
            continue
        assert cfg.base_url_for(role) == "https://primary.example/v1", role


def test_all_per_role_keys_no_shared_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)
    for role in ROLES:
        monkeypatch.setenv(f"CANOPYWAVE_API_KEY_{role.upper()}", f"key-{role}")

    cfg = Config.load()

    for role in ROLES:
        assert cfg.api_key_for(role) == f"key-{role}"


def test_missing_key_for_one_role_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)
    # Set keys for all roles except 'qa'.
    for role in ROLES:
        if role == "qa":
            continue
        monkeypatch.setenv(f"CANOPYWAVE_API_KEY_{role.upper()}", "x")

    with pytest.raises(RuntimeError, match="qa"):
        Config.load()


def test_no_keys_at_all_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_canopywave_env(monkeypatch)

    with pytest.raises(RuntimeError, match="CANOPYWAVE_API_KEY"):
        Config.load()


# ---------------------------------------------------------------------------
# LLMClient: ensures the right AsyncOpenAI is used for each role.
# ---------------------------------------------------------------------------


def test_llm_client_uses_distinct_clients_per_unique_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_canopywave_env(monkeypatch)
    monkeypatch.setenv("CANOPYWAVE_API_KEY", "SHARED")
    monkeypatch.setenv("CANOPYWAVE_API_KEY_PRODUCT", "PROD-KEY")
    monkeypatch.setenv("CANOPYWAVE_BASE_URL_REVIEWER", "https://other.example/v1")

    cfg = Config.load()
    client = LLMClient(cfg)

    # product has its own key -> its own underlying httpx client
    prod_client = client._client_for("product")
    arch_client = client._client_for("architect")
    rev_client = client._client_for("reviewer")
    backend_client = client._client_for("backend")

    # product key differs -> different from shared roles
    assert prod_client is not arch_client
    # reviewer base URL differs -> different from shared roles
    assert rev_client is not arch_client
    # architect and backend share both key & base URL -> same client (pool reuse)
    assert arch_client is backend_client


def test_llm_client_close_closes_each_unique_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    _clear_canopywave_env(monkeypatch)
    monkeypatch.setenv("CANOPYWAVE_API_KEY", "SHARED")
    monkeypatch.setenv("CANOPYWAVE_API_KEY_PRODUCT", "PROD-KEY")

    cfg = Config.load()
    client = LLMClient(cfg)

    # touch two roles -> two unique underlying clients are built
    c1 = client._client_for("product")
    c2 = client._client_for("architect")
    assert c1 is not c2

    closed: list[object] = []

    async def fake_close(self):  # noqa: ARG001
        closed.append(self)

    monkeypatch.setattr(type(c1), "close", fake_close, raising=False)

    asyncio.run(client.close())

    assert len(closed) == 2
    assert set(map(id, closed)) == {id(c1), id(c2)}
