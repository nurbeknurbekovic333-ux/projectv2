"""Tests for orgos.github_publish.

We don't hit GitHub for real — both httpx and ``git`` subprocesses are
mocked. We verify:
  * the right REST endpoint is hit depending on user-vs-org owner,
  * existing repos (HTTP 422) are reused instead of failing,
  * the token is embedded in the push URL but stripped from the
    persisted remote (and from error messages),
  * git commands run in the right cwd in the right order,
  * push failure raises GitHubPublishError with a scrubbed message,
  * non-git directory raises a clear error.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

os.environ.setdefault("CANOPYWAVE_API_KEY", "test-fake-key")

from orgos.github_publish import (
    GITHUB_API,
    GitHubPublishError,
    publish_to_github,
)


TOKEN = "ghp_fake_token_123"
OWNER_USER = "alice"
OWNER_ORG = "acme-inc"
REPO_NAME = "hello-world"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_git_repo(tmp_path: Path) -> Path:
    project = tmp_path / "demo"
    project.mkdir()
    (project / ".git").mkdir()  # publisher only checks .git exists
    return project


def _user_response(t: str = "User") -> httpx.Response:
    return httpx.Response(200, json={"type": t, "login": OWNER_USER})


def _org_response() -> httpx.Response:
    return httpx.Response(200, json={"type": "Organization", "login": OWNER_ORG})


def _repo_json(owner: str, repo: str) -> dict[str, Any]:
    return {
        "name": repo,
        "html_url": f"https://github.com/{owner}/{repo}",
        "clone_url": f"https://github.com/{owner}/{repo}.git",
        "default_branch": "main",
    }


class _FakeGit:
    """Records git invocations; configurable per-call return codes."""

    def __init__(self, push_returncode: int = 0, push_stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.cwds: list[str] = []
        self.push_returncode = push_returncode
        self.push_stderr = push_stderr

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        # argv is ["git", ...]
        self.calls.append(list(argv))
        self.cwds.append(cwd)
        if argv[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(
                argv,
                returncode=self.push_returncode,
                stdout="",
                stderr=self.push_stderr,
            )
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")


def _client(
    handler,  # type: ignore[no-untyped-def]
) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url=GITHUB_API)


# ---------------------------------------------------------------------------
# happy path: user owner, fresh repo
# ---------------------------------------------------------------------------


def test_publish_user_owner_creates_repo_and_pushes(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/users/{OWNER_USER}":
            return _user_response("User")
        if request.url.path == "/user/repos" and request.method == "POST":
            captured["create_body"] = request.read()
            captured["create_auth"] = request.headers.get("Authorization")
            return httpx.Response(201, json=_repo_json(OWNER_USER, REPO_NAME))
        return httpx.Response(404)

    git = _FakeGit()
    with _client(handler) as client:
        result = publish_to_github(
            project_root=project,
            token=TOKEN,
            owner=OWNER_USER,
            repo_name=REPO_NAME,
            description="hello",
            private=True,
            http_client=client,
            git_runner=git,
        )

    assert result.created is True
    assert result.owner == OWNER_USER
    assert result.html_url == f"https://github.com/{OWNER_USER}/{REPO_NAME}"
    assert captured["create_auth"] == f"Bearer {TOKEN}"

    # git workflow: branch -M main, remote remove origin, remote add origin <token-url>, push, remote set-url <token-free>
    cmds = [tuple(c) for c in git.calls]
    assert cmds[0] == ("git", "branch", "-M", "main")
    assert cmds[1] == ("git", "remote", "remove", "origin")
    add = cmds[2]
    assert add[:3] == ("git", "remote", "add")
    assert add[3] == "origin"
    assert f"x-access-token:{TOKEN}@github.com" in add[4], (
        "token must be embedded in push URL"
    )
    push = cmds[3]
    assert push == ("git", "push", "-u", "origin", "main")
    # Final step: token-free URL written back
    set_url = cmds[4]
    assert set_url == (
        "git", "remote", "set-url", "origin",
        f"https://github.com/{OWNER_USER}/{REPO_NAME}.git",
    )
    # ALL commands run inside the project directory
    assert all(c == str(project) for c in git.cwds)


# ---------------------------------------------------------------------------
# org owner -> /orgs/<owner>/repos
# ---------------------------------------------------------------------------


def test_publish_org_owner_uses_org_endpoint(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)
    seen_endpoints: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_endpoints.append(f"{request.method} {request.url.path}")
        if request.url.path == f"/users/{OWNER_ORG}":
            return _org_response()
        if request.url.path == f"/orgs/{OWNER_ORG}/repos" and request.method == "POST":
            return httpx.Response(201, json=_repo_json(OWNER_ORG, REPO_NAME))
        return httpx.Response(404)

    with _client(handler) as client:
        result = publish_to_github(
            project_root=project,
            token=TOKEN,
            owner=OWNER_ORG,
            repo_name=REPO_NAME,
            http_client=client,
            git_runner=_FakeGit(),
        )

    assert result.owner == OWNER_ORG
    assert any(e == f"POST /orgs/{OWNER_ORG}/repos" for e in seen_endpoints)
    assert not any("/user/repos" == e.split()[-1] for e in seen_endpoints), (
        f"must not hit /user/repos for an org: {seen_endpoints}"
    )


# ---------------------------------------------------------------------------
# existing repo (422) -> reuse
# ---------------------------------------------------------------------------


def test_publish_reuses_existing_repo_on_422(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/users/{OWNER_USER}":
            return _user_response("User")
        if request.url.path == "/user/repos" and request.method == "POST":
            return httpx.Response(
                422,
                json={
                    "message": "Repository creation failed.",
                    "errors": [
                        {
                            "resource": "Repository",
                            "code": "custom",
                            "field": "name",
                            "message": "name already exists on this account",
                        }
                    ],
                },
            )
        if request.url.path == f"/repos/{OWNER_USER}/{REPO_NAME}":
            return httpx.Response(200, json=_repo_json(OWNER_USER, REPO_NAME))
        return httpx.Response(404)

    with _client(handler) as client:
        result = publish_to_github(
            project_root=project,
            token=TOKEN,
            owner=OWNER_USER,
            repo_name=REPO_NAME,
            http_client=client,
            git_runner=_FakeGit(),
        )

    assert result.created is False
    assert result.owner == OWNER_USER


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_publish_rejects_non_git_directory(tmp_path: Path) -> None:
    project = tmp_path / "not-a-repo"
    project.mkdir()  # no .git
    with pytest.raises(GitHubPublishError, match="not a git repository"):
        publish_to_github(
            project_root=project,
            token=TOKEN,
            owner=OWNER_USER,
            repo_name=REPO_NAME,
        )


def test_publish_rejects_empty_token(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)
    with pytest.raises(GitHubPublishError, match="GITHUB_TOKEN"):
        publish_to_github(
            project_root=project,
            token="",
            owner=OWNER_USER,
            repo_name=REPO_NAME,
        )


def test_publish_rejects_unknown_owner(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        with pytest.raises(GitHubPublishError, match="not found"):
            publish_to_github(
                project_root=project,
                token=TOKEN,
                owner="ghost-user-12345",
                repo_name=REPO_NAME,
                http_client=client,
                git_runner=_FakeGit(),
            )


def test_publish_push_failure_scrubs_token_from_error(tmp_path: Path) -> None:
    project = _make_git_repo(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/users/{OWNER_USER}":
            return _user_response("User")
        if request.url.path == "/user/repos":
            return httpx.Response(201, json=_repo_json(OWNER_USER, REPO_NAME))
        return httpx.Response(404)

    # Simulate `git push` failing with a stderr that contains the token —
    # publisher must scrub it before raising.
    stderr_with_token = (
        f"fatal: Authentication failed for "
        f"'https://x-access-token:{TOKEN}@github.com/{OWNER_USER}/{REPO_NAME}.git/'\n"
    )
    git = _FakeGit(push_returncode=128, push_stderr=stderr_with_token)

    with _client(handler) as client:
        with pytest.raises(GitHubPublishError) as exc:
            publish_to_github(
                project_root=project,
                token=TOKEN,
                owner=OWNER_USER,
                repo_name=REPO_NAME,
                http_client=client,
                git_runner=git,
            )

    msg = str(exc.value)
    assert TOKEN not in msg, f"token leaked into error message: {msg!r}"
    assert "***" in msg, f"token wasn't scrubbed in error: {msg!r}"
