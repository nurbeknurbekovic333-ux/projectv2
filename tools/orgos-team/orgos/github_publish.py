"""GitHub auto-publish for finished orgos projects.

Flow:
  1. Caller already ran ``git_init_and_commit`` inside ``project_root`` —
     we don't re-init.
  2. We POST to the GitHub REST API to create a private repo under
     ``owner``. If it already exists (HTTP 422 with `name already exists`),
     we treat it as "use existing" and continue.
  3. We add a token-auth'd ``origin`` remote of the form
     ``https://x-access-token:<TOKEN>@github.com/<owner>/<repo>.git`` and
     ``git push -u origin main``.

Auth: a personal-access token (classic, ``repo`` scope) OR a fine-grained
token with **Contents: read & write** on the target repo (or
**Administration: read & write** if we are also creating the repo).
Set it as ``GITHUB_TOKEN`` in ``.env``. ``GITHUB_OWNER`` is either a user
(use ``/user/repos``) or an org (use ``/orgs/<owner>/repos``); we detect
which by hitting ``/users/<owner>`` first.

Path safety: we only touch ``project_root`` (existing dir) and never
recurse outside it. The token only appears in the ``origin`` URL and in
the Authorization header; it never gets written to a tracked file (we
rewrite the remote to a token-free URL after the push completes
successfully).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_BRANCH = "main"

# Subprocess runner injected for tests.
SubprocessRunner = Callable[..., subprocess.CompletedProcess]


class GitHubPublishError(RuntimeError):
    """Raised when we can't create / push to a repository."""


@dataclass(frozen=True)
class PublishResult:
    owner: str
    repo: str
    html_url: str
    clone_url: str
    created: bool  # True if we created the repo, False if it already existed


def _is_org(client: httpx.Client, owner: str) -> bool:
    """Return True if ``owner`` is an org (not a user)."""
    r = client.get(f"{GITHUB_API}/users/{owner}", timeout=15)
    if r.status_code == 404:
        # Could be an org accessible only via /orgs; try that.
        r2 = client.get(f"{GITHUB_API}/orgs/{owner}", timeout=15)
        if r2.status_code == 200:
            return True
        raise GitHubPublishError(
            f"GitHub owner '{owner}' not found (neither user nor org). "
            f"Check GITHUB_OWNER in your .env."
        )
    r.raise_for_status()
    return (r.json().get("type") or "").lower() == "organization"


def _create_repo(
    client: httpx.Client,
    *,
    owner: str,
    repo: str,
    private: bool,
    description: str,
) -> tuple[dict, bool]:
    """Create a repo. Returns (json, created). Reuses existing repo on 422."""
    is_org = _is_org(client, owner)
    endpoint = (
        f"{GITHUB_API}/orgs/{owner}/repos"
        if is_org
        else f"{GITHUB_API}/user/repos"
    )
    payload = {
        "name": repo,
        "private": private,
        "auto_init": False,
        "description": description[:350],
        "has_issues": True,
        "has_wiki": False,
    }
    r = client.post(endpoint, json=payload, timeout=30)
    if r.status_code in (200, 201):
        return r.json(), True

    if r.status_code == 422:
        # Already exists — reuse it. Fetch the metadata.
        r2 = client.get(f"{GITHUB_API}/repos/{owner}/{repo}", timeout=15)
        if r2.status_code == 200:
            return r2.json(), False
    raise GitHubPublishError(
        f"GitHub repo create failed ({r.status_code}): "
        f"{_safe_error_body(r)}"
    )


def _safe_error_body(r: httpx.Response) -> str:
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        return r.text[:300]
    if isinstance(body, dict) and "message" in body:
        msg = body["message"]
        errors = body.get("errors") or []
        if errors:
            msg = f"{msg}: {errors}"
        return str(msg)[:300]
    return str(body)[:300]


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    runner: SubprocessRunner | None = None,
) -> subprocess.CompletedProcess:
    """Run git in ``cwd``, capturing output. Never raise on non-zero — caller decides."""
    r = (runner or subprocess.run)(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return r


def publish_to_github(
    *,
    project_root: Path,
    token: str,
    owner: str,
    repo_name: str,
    description: str = "",
    private: bool = True,
    http_client: httpx.Client | None = None,
    git_runner: SubprocessRunner | None = None,
) -> PublishResult:
    """Create the repo (if needed) and push ``project_root`` to it.

    The local repo at ``project_root`` MUST already exist and have at least
    one commit (orgos' ``git_init_and_commit`` does that).
    """
    if not (project_root / ".git").exists():
        raise GitHubPublishError(
            f"{project_root} is not a git repository — call git_init_and_commit first"
        )
    if not token:
        raise GitHubPublishError("GITHUB_TOKEN is empty")
    if not owner:
        raise GitHubPublishError("GITHUB_OWNER is empty")
    if not repo_name:
        raise GitHubPublishError("repo_name is empty")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "orgos-team-publisher",
    }
    owns_client = http_client is None
    client = http_client or httpx.Client()
    # Always ensure the auth + accept headers are present. ``update`` (vs
    # ``setdefault``) is deliberate: even if the caller pre-populated a
    # client, our token has to be on every request.
    for k, v in headers.items():
        client.headers[k] = v

    try:
        repo_json, created = _create_repo(
            client,
            owner=owner,
            repo=repo_name,
            private=private,
            description=description,
        )
    finally:
        if owns_client:
            client.close()

    html_url = repo_json["html_url"]
    clone_url = repo_json["clone_url"]
    # Embed token only in the remote URL we're about to push to; never log it.
    push_url = clone_url.replace(
        "https://", f"https://x-access-token:{token}@", 1
    )

    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "orgos-team")
    env.setdefault("GIT_AUTHOR_EMAIL", "orgos-team@local")
    env.setdefault("GIT_COMMITTER_NAME", "orgos-team")
    env.setdefault("GIT_COMMITTER_EMAIL", "orgos-team@local")
    # Don't let the user's askpass / credential helper interfere.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/echo"

    # Make sure default branch is `main`.
    _run_git(["branch", "-M", DEFAULT_BRANCH], cwd=project_root, env=env, runner=git_runner)

    # Drop any pre-existing origin (e.g. re-runs) then add the auth'd one.
    _run_git(["remote", "remove", "origin"], cwd=project_root, env=env, runner=git_runner)
    add = _run_git(
        ["remote", "add", "origin", push_url],
        cwd=project_root,
        env=env,
        runner=git_runner,
    )
    if add.returncode != 0:
        raise GitHubPublishError(
            f"git remote add origin failed: {add.stderr.strip() or add.stdout.strip()}"
        )

    push = _run_git(
        ["push", "-u", "origin", DEFAULT_BRANCH],
        cwd=project_root,
        env=env,
        runner=git_runner,
    )
    if push.returncode != 0:
        # Scrub the token out of any error message we surface.
        scrubbed = (push.stderr or push.stdout).replace(token, "***")
        raise GitHubPublishError(f"git push failed: {scrubbed.strip()}")

    # Rewrite remote to the token-free URL so the token never ends up
    # serialized into the user's git config on disk.
    _run_git(
        ["remote", "set-url", "origin", clone_url],
        cwd=project_root,
        env=env,
        runner=git_runner,
    )

    logger.info("Published %s to %s", project_root.name, html_url)
    return PublishResult(
        owner=owner,
        repo=repo_name,
        html_url=html_url,
        clone_url=clone_url,
        created=created,
    )
