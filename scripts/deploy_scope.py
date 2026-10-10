"""Decide whether a deploy rebuilds the HF Space, and record each deploy that passed its gate.

Usage (from ``.github/workflows/deploy-backend.yml``):
    python3 -m scripts.deploy_scope decide --head <sha> --space <owner/name> [--force]
    python3 -m scripts.deploy_scope record --sha <sha> --space-commit <sha> --space <owner/name>

The Space image is built from a fixed set of repository paths, ``BUILD_CONTEXT``: the
Dockerfile's COPY sources, plus the files that shape the build and the deployed README.
``IMAGE_INPUTS`` adds the deploy procedure (``DEPLOY_PROCEDURE``), which is never pushed
to the Space but whose changes must still be proven by a deploy.
``decide`` compares those paths between the commit being deployed and the last commit
that deployed successfully, and deploys only if one of them differs. ``record`` writes
that baseline as a GitHub deployment in the ``huggingface-space`` environment with a
``success`` status. The workflow runs it only after the Space reached RUNNING and
``/healthz`` reported ``model_loaded: true`` (ADR-027).

The baseline is the last successful deploy, never the parent commit. A commit whose
deploy was skipped, superseded, cancelled or failed has no record, so its changes stay
in the range the next run compares. The record also names the commit pushed to the
Space. ``decide`` skips only while the Space is still on that commit, because a deploy
that pushed but then failed its gate has left the Space on something else. When any of
this can't be established (no record yet, an API error, a commit missing from the
checkout), the run deploys. The script uses only the standard library, so the deploy
job runs it on the runner's interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENVIRONMENT = "huggingface-space"
# Only records this workflow wrote count as a baseline (GITHUB_TOKEN acts as this bot).
RECORDER = "github-actions[bot]"

# A trailing slash marks a directory prefix; anything else is an exact path.
#
# What the Space builds from. ``scripts/space_snapshot.py`` deploys exactly these paths of
# the tested commit, so a path missing here is missing from the Space (ADR-030).
BUILD_CONTEXT: tuple[str, ...] = (
    # The Dockerfile's COPY sources: the files the image is built from.
    "requirements.txt",
    "pyproject.toml",
    "README.md",  # copied for `readme = "README.md"`; the deployed copy is the Space card
    "src/",
    "backend/",
    "configs/",
    "models/",
    # The build recipe, and the filter that prunes its context.
    "Dockerfile",
    ".dockerignore",
    # The Space builds from a git checkout of the deploy commit; this file decides how
    # that checkout materialises files (LFS filters, line endings).
    ".gitattributes",
)

# The deploy procedure: the workflow runs the deploy, the snapshot script builds what is
# pushed (and writes the Space's README config header), this script decides and records
# each deploy, and the smoke test gates it. None of them is pushed to the Space, but a
# change to any of them is proven by the run that introduces it.
DEPLOY_PROCEDURE: tuple[str, ...] = (
    ".github/workflows/deploy-backend.yml",
    "scripts/deploy_scope.py",
    "scripts/space_snapshot.py",
    "scripts/smoke_caption.py",
)

IMAGE_INPUTS: tuple[str, ...] = BUILD_CONTEXT + DEPLOY_PROCEDURE

_SHA = re.compile(r"[0-9a-f]{40}")
_SPACE = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
# Read failures that mean "unknown", so the run deploys instead of guessing.
_UNREADABLE = (
    urllib.error.URLError,
    TimeoutError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
)

Fetch = Callable[[str], Any]
Post = Callable[[str, dict[str, Any]], Any]


class BaselineUnavailable(Exception):
    """The deploy record, or the Space's head, couldn't be read or didn't look right."""


@dataclass(frozen=True)
class Baseline:
    """The last successful deploy: the GitHub commit, and the commit pushed to the Space."""

    sha: str
    space_commit: str


@dataclass(frozen=True)
class Decision:
    deploy: bool
    reason: str
    base: str | None = None
    image_changes: tuple[str, ...] = ()


def _matches(path: str, entries: tuple[str, ...]) -> bool:
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry for entry in entries
    )


def is_image_input(path: str) -> bool:
    """True if a repository path is, or is inside, one of ``IMAGE_INPUTS``."""
    return _matches(path, IMAGE_INPUTS)


def is_build_context(path: str) -> bool:
    """True if a repository path is, or is inside, one of ``BUILD_CONTEXT``."""
    return _matches(path, BUILD_CONTEXT)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)


def commit_exists(sha: str, repo: Path) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def changed_paths(base: str, head: str, repo: Path) -> list[str]:
    """Paths whose content differs between the trees of two commits.

    A tree diff, not a walk of the commits in between: a change that was later reverted
    doesn't count, because the image would be the same. ``--no-renames`` lists both
    sides of a move, so a file moved out of an image directory still counts.
    """
    result = _git(repo, "diff", "--name-only", "--no-renames", "-z", base, head, "--")
    if result.returncode != 0:
        raise RuntimeError(f"git diff {base} {head} failed: {result.stderr.decode().strip()}")
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def decide(head: str, baseline: Baseline | None, space_head: str | None, repo: Path) -> Decision:
    """Deploy unless the Space is on the last successful deploy and no image input changed.

    ``space_head`` is the commit the Space's repository is on now.
    """
    if baseline is None:
        return Decision(
            True,
            f"Deploying: no successful deploy is recorded in the {ENVIRONMENT} environment, "
            "so there is nothing to compare against.",
        )
    base = baseline.sha
    if space_head != baseline.space_commit:
        return Decision(
            True,
            f"Deploying: the Space is on {space_head}, not {baseline.space_commit}, the commit "
            f"the last successful deploy ({base}) pushed. A later deploy pushed without "
            "passing the health gate, or the Space changed outside this workflow.",
            base,
        )
    if not commit_exists(base, repo):
        return Decision(
            True, f"Deploying: the last deployed commit {base} isn't in this checkout.", base
        )
    try:
        paths = changed_paths(base, head, repo)
    except RuntimeError as exc:
        return Decision(True, f"Deploying: comparing with {base} failed ({exc}).", base)
    changes = tuple(path for path in paths if is_image_input(path))
    if changes:
        return Decision(
            True,
            f"Deploying: {len(changes)} image input(s) changed since {base}, "
            "the last successfully deployed commit.",
            base,
            changes,
        )
    return Decision(
        False,
        f"No image input changed since {base}, the last successfully deployed commit, "
        "and the Space is still on it. Run the workflow manually to force a rebuild.",
        base,
    )


def find_baseline(fetch: Fetch, repository: str) -> Baseline | None:
    """The newest deployment this workflow recorded, if it succeeded.

    Only this workflow writes to the environment, and it writes a record only after a
    deploy passed its gate. So the newest record is the baseline, and one without a
    ``success`` status (its status write failed) isn't trusted: ``None`` means deploy.
    Raises ``BaselineUnavailable`` when the answer doesn't look like a deploy record.
    """
    deployments = fetch(f"/repos/{repository}/deployments?environment={ENVIRONMENT}&per_page=100")
    if not isinstance(deployments, list):
        raise BaselineUnavailable("the deployments response is not a list")
    ours = [d for d in deployments if (d.get("creator") or {}).get("login") == RECORDER]
    if not ours:
        return None
    newest = max(ours, key=lambda d: int(d["id"]))
    statuses = fetch(f"/repos/{repository}/deployments/{newest['id']}/statuses?per_page=100")
    if not isinstance(statuses, list):
        raise BaselineUnavailable(f"the statuses of deployment {newest['id']} are not a list")
    if not any(status.get("state") == "success" for status in statuses):
        return None
    return Baseline(
        _valid_sha(newest.get("sha")),
        _valid_sha((newest.get("payload") or {}).get("space_commit")),
    )


def _valid_sha(value: object) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise BaselineUnavailable(f"not a full commit SHA: {value!r}")
    return value


def record_deploy(
    post: Post, repository: str, sha: str, space_commit: str, space: str, log_url: str
) -> int:
    """Write the baseline: a deployment of ``sha`` with a ``success`` status."""
    deployment = post(
        f"/repos/{repository}/deployments",
        {
            "ref": sha,
            "environment": ENVIRONMENT,
            "auto_merge": False,
            "required_contexts": [],  # CI already gated this commit
            "production_environment": True,
            "description": f"Space commit {space_commit[:12]}",
            "payload": {"space_commit": space_commit},
        },
    )
    deployment_id = int(deployment["id"])
    post(
        f"/repos/{repository}/deployments/{deployment_id}/statuses",
        {
            "state": "success",
            "description": "Space RUNNING and /healthz reported model_loaded: true",
            "environment_url": f"https://huggingface.co/spaces/{space}",
            "log_url": log_url,
        },
    )
    return deployment_id


def _github(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    request = urllib.request.Request(
        os.environ.get("GITHUB_API_URL", "https://api.github.com") + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _space_head(space: str) -> str:
    """The commit the Space's repository is on, from the public HF API (no token)."""
    url = f"https://huggingface.co/api/spaces/{space}"
    with urllib.request.urlopen(url, timeout=30) as response:
        sha = json.load(response).get("sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise BaselineUnavailable(f"the HF API returned no valid head commit for {space}")
    return sha


def _append(env_var: str, lines: Iterable[str]) -> None:
    with Path(os.environ[env_var]).open("a", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)


def report(decision: Decision, head: str) -> None:
    """Print the decision, and write it to the step outputs and the job summary."""
    if decision.deploy:
        print(decision.reason)
    else:
        print(f"::notice title=Space deploy skipped::{decision.reason}")
    for path in decision.image_changes:
        print(f"  image input changed: {path}")
    _append(
        "GITHUB_OUTPUT",
        [f"deploy={'true' if decision.deploy else 'false'}", f"base={decision.base or ''}"],
    )
    summary = [
        f"### Space deploy: {'deploy' if decision.deploy else 'skipped'}",
        "",
        decision.reason,
        "",
        f"Compared `{decision.base or 'none'}` (last successful deploy) with `{head}`.",
    ]
    if decision.image_changes:
        summary += ["", "Image inputs changed:", ""]
        summary += [f"- `{path}`" for path in decision.image_changes[:50]]
        if len(decision.image_changes) > 50:
            summary.append(f"- … and {len(decision.image_changes) - 50} more")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        _append("GITHUB_STEP_SUMMARY", summary)


def _run_decide(head: str, space: str, force: bool) -> int:
    if not os.environ.get("GITHUB_OUTPUT"):
        # Without the output every later step would read deploy='' and skip silently.
        print("::error::GITHUB_OUTPUT is not set, so the decision can't reach the deploy steps.")
        return 1
    if force:
        report(Decision(True, "Manual run: deploys regardless of what changed."), head)
        return 0
    repository = os.environ["GITHUB_REPOSITORY"]
    try:
        baseline = find_baseline(lambda path: _github("GET", path), repository)
        space_head = _space_head(space) if baseline else None
    except (*_UNREADABLE, BaselineUnavailable) as exc:
        reason = (
            f"the deploy record or the Space's head couldn't be read ({type(exc).__name__}: {exc})"
        )
        print(f"::warning::{reason}; deploying rather than risk skipping a needed rebuild.")
        report(Decision(True, f"Deploying: {reason}."), head)
        return 0
    report(decide(head, baseline, space_head, Path.cwd()), head)
    return 0


def _run_record(sha: str, space_commit: str, space: str) -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    log_url = f"{server}/{repository}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    try:
        deployment_id = record_deploy(
            lambda path, body: _github("POST", path, body),
            repository,
            sha,
            space_commit,
            space,
            log_url,
        )
    except _UNREADABLE as exc:
        print(
            f"::error::The Space is healthy on {sha}, but recording the deploy failed "
            f"({type(exc).__name__}: {exc}). The next run finds no baseline, so it redeploys."
        )
        return 1
    print(f"Recorded {sha} (Space commit {space_commit}) as deployment {deployment_id}")
    return 0


def _sha(value: str) -> str:
    if not _SHA.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a full lowercase commit SHA: {value!r}")
    return value


def _space(value: str) -> str:
    if not _SPACE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not an owner/name Space id: {value!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether a deploy rebuilds the HF Space, and record each deploy that passed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    decide_parser = commands.add_parser("decide", help="deploy, or skip: no image input changed")
    decide_parser.add_argument("--head", type=_sha, required=True, help="the commit to deploy")
    decide_parser.add_argument("--space", type=_space, required=True, help="owner/name")
    decide_parser.add_argument("--force", action="store_true", help="deploy whatever changed")
    record_parser = commands.add_parser("record", help="record a deploy that passed the gate")
    record_parser.add_argument("--sha", type=_sha, required=True, help="the GitHub commit")
    record_parser.add_argument("--space-commit", type=_sha, required=True, help="Space commit")
    record_parser.add_argument("--space", type=_space, required=True, help="owner/name")
    args = parser.parse_args(argv)
    if args.command == "decide":
        return _run_decide(args.head, args.space, args.force)
    return _run_record(args.sha, args.space_commit, args.space)


if __name__ == "__main__":
    sys.exit(main())
