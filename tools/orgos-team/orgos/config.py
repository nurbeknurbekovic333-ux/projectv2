"""Configuration loader.

Reads .env (or environment variables), exposes a typed Config object that the
rest of the system imports. Per-role model overrides fall back to the default.

Per-role API keys and base URLs are also supported:
    CANOPYWAVE_API_KEY_<ROLE>   overrides the shared CANOPYWAVE_API_KEY
    CANOPYWAVE_BASE_URL_<ROLE>  overrides the shared CANOPYWAVE_BASE_URL
where <ROLE> ∈ {PRODUCT, ARCHITECT, BACKEND, FRONTEND, DEVOPS, QA, REVIEWER,
SECURITY}. The shared CANOPYWAVE_API_KEY remains a valid fallback for any
role that doesn't set its own key. A role is only required to have *some*
key (its own or the fallback); the loader raises if any role lacks both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROLES = (
    "product",
    "architect",
    "backend",
    "frontend",
    "devops",
    "qa",
    "reviewer",
    "security",
)

DEFAULT_BASE_URL = "https://inference.canopywave.io/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"


@dataclass(frozen=True)
class Config:
    base_url: str
    default_model: str
    role_models: dict[str, str]
    role_api_keys: dict[str, str]
    role_base_urls: dict[str, str]
    max_concurrency: int
    temperature: float
    output_dir: Path
    github_token: str | None
    github_owner: str | None
    prompts_dir: Path
    # Retry / resilience knobs for LLMClient.
    max_retries: int = 5
    retry_backoff_max: float = 30.0
    # Per-call response cap (in tokens). We pass this as ``max_tokens`` to
    # every chat.completions.create call so the server doesn't fall back
    # to a tiny default that truncates large implementer outputs
    # mid-string. Kimi K2.6 supports up to ~8k completion tokens.
    max_response_tokens: int = 8192

    @property
    def api_key(self) -> str:
        """Back-compat: any one role key (they may all be the same).

        Kept so that older callers that read ``config.api_key`` keep working;
        new code should call :meth:`api_key_for` for the actual per-role key.
        """
        return next(iter(self.role_api_keys.values()))

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = None) -> "Config":
        load_dotenv(env_file, override=False)

        shared_key = os.environ.get("CANOPYWAVE_API_KEY", "").strip()
        shared_base_url = os.environ.get(
            "CANOPYWAVE_BASE_URL", DEFAULT_BASE_URL
        ).strip() or DEFAULT_BASE_URL
        default_model = os.environ.get(
            "ORGOS_DEFAULT_MODEL", DEFAULT_MODEL
        ).strip() or DEFAULT_MODEL

        role_models: dict[str, str] = {}
        role_api_keys: dict[str, str] = {}
        role_base_urls: dict[str, str] = {}
        missing_keys: list[str] = []

        for role in ROLES:
            up = role.upper()

            model_override = os.environ.get(f"ORGOS_MODEL_{up}", "").strip()
            role_models[role] = model_override or default_model

            key_override = os.environ.get(f"CANOPYWAVE_API_KEY_{up}", "").strip()
            key = key_override or shared_key
            if not key:
                missing_keys.append(role)
            else:
                role_api_keys[role] = key

            url_override = os.environ.get(f"CANOPYWAVE_BASE_URL_{up}", "").strip()
            role_base_urls[role] = url_override or shared_base_url

        if missing_keys:
            if len(missing_keys) == len(ROLES):
                raise RuntimeError(
                    "CANOPYWAVE_API_KEY is not set, and no per-role keys "
                    "(CANOPYWAVE_API_KEY_<ROLE>) are set either. Copy "
                    ".env.example to .env and fill in at least the shared "
                    "CANOPYWAVE_API_KEY, or set a key for every role."
                )
            joined = ", ".join(missing_keys)
            raise RuntimeError(
                "Missing API key for role(s): "
                f"{joined}. Set CANOPYWAVE_API_KEY_<ROLE> for each listed "
                "role, or set the shared CANOPYWAVE_API_KEY as a fallback."
            )

        max_concurrency = int(os.environ.get("ORGOS_MAX_CONCURRENCY", "4"))
        temperature = float(os.environ.get("ORGOS_TEMPERATURE", "0.2"))

        max_retries = max(1, int(os.environ.get("ORGOS_MAX_RETRIES", "5")))
        retry_backoff_max = max(
            1.0, float(os.environ.get("ORGOS_RETRY_BACKOFF_MAX", "30.0"))
        )
        max_response_tokens = max(
            512, int(os.environ.get("ORGOS_MAX_RESPONSE_TOKENS", "8192"))
        )

        output_dir = Path(os.environ.get("ORGOS_OUTPUT_DIR", "output")).expanduser()

        github_token = os.environ.get("GITHUB_TOKEN", "").strip() or None
        github_owner = os.environ.get("GITHUB_OWNER", "").strip() or None

        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"

        return cls(
            base_url=shared_base_url,
            default_model=default_model,
            role_models=role_models,
            role_api_keys=role_api_keys,
            role_base_urls=role_base_urls,
            max_concurrency=max_concurrency,
            temperature=temperature,
            output_dir=output_dir,
            github_token=github_token,
            github_owner=github_owner,
            prompts_dir=prompts_dir,
            max_retries=max_retries,
            retry_backoff_max=retry_backoff_max,
            max_response_tokens=max_response_tokens,
        )

    def model_for(self, role: str) -> str:
        return self.role_models.get(role, self.default_model)

    def api_key_for(self, role: str) -> str:
        try:
            return self.role_api_keys[role]
        except KeyError as e:
            raise KeyError(f"No API key configured for role '{role}'") from e

    def base_url_for(self, role: str) -> str:
        return self.role_base_urls.get(role, self.base_url)
