"""Tests for the deploy-scope decision and the deploy record (TASK-022, ADR-027).

The scenarios run against throwaway git repositories, so the real ``git diff`` decides.
The GitHub API is a hand-built fake, so nothing touches the network. The last tests read
the committed Dockerfile and deploy workflow, so the image-input list and the step order
can't drift from what actually builds and deploys the Space.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import scripts.deploy_scope as scope
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-backend.yml"
STATUS_FUNCTIONS = re.compile(r"\b(always|failure|cancelled|success)\(\)")


class Repo:
    """A throwaway git repository whose commits stand in for pushes to main."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                "user.name=deploy-scope-test",
                "-c",
                "user.email=deploy-scope-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.autocrlf=false",
                *args,
            ],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def commit(self, files: dict[str, str | None], message: str = "change") -> str:
        """Write (or, for ``None``, delete) files and commit them. Returns the SHA."""
        for name, content in files.items():
            target = self.path / name
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)


@pytest.fixture
def deployed(repo: Repo) -> str:
    """The commit the Space last deployed successfully: the baseline."""
    return repo.commit(
        {
            "README.md": "# Image captioning\n",
            "requirements.txt": "fastapi==0.133.0\n",
            "pyproject.toml": "[project]\nname = 'captioning'\n",
            "Dockerfile": "FROM python:3.11-slim-bookworm\n",
            "backend/app/main.py": "app = None\n",
            "src/captioning/__init__.py": "",
            "configs/base.yaml": "seed: 42\n",
            "docs/TASKS.md": "# Tasks\n",
            "frontend/src/App.jsx": "export default 1\n",
            "tests/unit/test_x.py": "",
        },
        "deployed",
    )


# ---------------------------------------------------------------- image inputs


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "Dockerfile",
        ".dockerignore",
        ".gitattributes",
        "backend/app/main.py",
        "backend/app/tests/test_routes.py",  # backend/ is copied whole
        "src/captioning/inference/beam.py",
        "configs/base.yaml",
        "models/v1.0.0/vocab.json",
        ".github/workflows/deploy-backend.yml",
        "scripts/deploy_scope.py",
        "scripts/space_snapshot.py",
        "scripts/smoke_caption.py",
    ],
)
def test_image_inputs_are_recognised(path: str) -> None:
    assert scope.is_image_input(path)


def test_image_inputs_are_the_build_context_plus_the_deploy_procedure() -> None:
    assert scope.IMAGE_INPUTS == scope.BUILD_CONTEXT + scope.DEPLOY_PROCEDURE
    # The deploy procedure triggers a deploy but is never pushed to the Space (ADR-030).
    for path in scope.DEPLOY_PROCEDURE:
        assert scope.is_image_input(path)
        assert not scope.is_build_context(path)


@pytest.mark.parametrize(
    "path",
    [
        "docs/TASKS.md",
        "docs/README.md",  # only the root README is copied
        "frontend/src/App.jsx",
        "tests/unit/test_deploy_scope.py",
        "results/stabilized-greedy/metrics.json",
        "notebooks/01_ieee_inceptionv3_transformer.ipynb",
        ".github/workflows/ci.yml",
        "requirements-dev.txt",
        "scripts/train.py",
        "CLAUDE.md",
        "backend-notes.md",  # a directory prefix needs its slash
        "README.md.orig",
    ],
)
def test_other_paths_are_not_image_inputs(path: str) -> None:
    assert not scope.is_image_input(path)


def _dockerfile_copy_sources() -> list[str]:
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    sources: list[str] = []
    for line in text.splitlines():
        words = line.split()
        if not words or words[0].upper() not in {"COPY", "ADD"}:
            continue
        assert not any(w.startswith("--from") for w in words), f"multi-stage COPY: {line}"
        args = [w for w in words[1:] if not w.startswith("--")]
        sources += args[:-1]  # the last argument is the destination
    return sources


def test_every_dockerfile_copy_source_is_in_the_build_context() -> None:
    """The snapshot pushed to the Space holds only BUILD_CONTEXT, so a COPY source
    missing from it would be missing from the Space's build (ADR-030)."""
    sources = _dockerfile_copy_sources()
    assert sources, "no COPY lines found in the Dockerfile"
    for source in sources:
        assert "*" not in source and "?" not in source, f"glob COPY source {source!r}"
        is_dir = (REPO_ROOT / source).is_dir()
        entry = source.rstrip("/") + "/" if is_dir else source
        assert entry in scope.BUILD_CONTEXT, f"Dockerfile copies {source!r}; add {entry!r}"
    for build_file in ("Dockerfile", ".dockerignore", "README.md", "pyproject.toml"):
        assert build_file in scope.BUILD_CONTEXT


def test_every_image_input_exists() -> None:
    for entry in scope.IMAGE_INPUTS:
        path = REPO_ROOT / entry
        assert path.is_dir() if entry.endswith("/") else path.is_file(), f"{entry} is missing"


# ------------------------------------------------------------- the decision

SPACE_A = "5" * 40  # the Space commit that the recorded deploy pushed


def _decide(repo: Repo, head: str, base: str, space_head: str = SPACE_A) -> scope.Decision:
    return scope.decide(head, scope.Baseline(base, SPACE_A), space_head, repo.path)


def test_image_change_deploys(repo: Repo, deployed: str) -> None:
    head = repo.commit({"backend/app/main.py": "app = 'v2'\n"})
    decision = _decide(repo, head, deployed)
    assert decision.deploy
    assert decision.image_changes == ("backend/app/main.py",)
    assert decision.base == deployed


def test_docs_only_change_skips(repo: Repo, deployed: str) -> None:
    head = repo.commit(
        {
            "docs/TASKS.md": "# Tasks\n- done\n",
            "frontend/src/App.jsx": "export default 2\n",
            "tests/unit/test_x.py": "# more\n",
        }
    )
    decision = _decide(repo, head, deployed)
    assert not decision.deploy
    assert decision.image_changes == ()
    assert deployed in decision.reason


def test_readme_change_deploys(repo: Repo, deployed: str) -> None:
    head = repo.commit({"README.md": "# Image captioning\n\nNew status line.\n"})
    decision = _decide(repo, head, deployed)
    assert decision.deploy
    assert decision.image_changes == ("README.md",)


@pytest.mark.parametrize(
    "path",
    ["requirements.txt", "pyproject.toml", "configs/base.yaml", "Dockerfile", ".dockerignore"],
)
def test_dependency_config_and_build_changes_deploy(repo: Repo, deployed: str, path: str) -> None:
    head = repo.commit({path: "changed\n"})
    decision = _decide(repo, head, deployed)
    assert decision.deploy
    assert decision.image_changes == (path,)


def test_one_image_change_anywhere_in_the_range_deploys(repo: Repo, deployed: str) -> None:
    repo.commit({"docs/TASKS.md": "a\n"})
    repo.commit({"src/captioning/__init__.py": "VERSION = 2\n"})
    head = repo.commit({"frontend/src/App.jsx": "b\n"})
    decision = _decide(repo, head, deployed)
    assert decision.deploy
    assert decision.image_changes == ("src/captioning/__init__.py",)


def test_undeployed_image_change_is_still_deployed_by_a_later_docs_commit(
    repo: Repo, deployed: str
) -> None:
    """A, then B (image change; skipped, superseded, or failed before its push), then C (docs)."""
    b = repo.commit({"backend/app/main.py": "app = 'B'\n"}, "B")
    c = repo.commit({"docs/TASKS.md": "C\n"}, "C")
    # B has no record, so the baseline is still A and C carries B's change.
    decision = _decide(repo, c, deployed)
    assert decision.deploy
    assert decision.image_changes == ("backend/app/main.py",)
    # Comparing with the parent instead would have skipped it.
    assert not _decide(repo, c, b).deploy


def test_failed_deploy_that_pushed_is_redeployed_even_after_a_revert(
    repo: Repo, deployed: str
) -> None:
    """B pushed its image to the Space, failed the health gate, and C reverted B.

    C's tree equals A's, but the Space is on B's deploy commit, not A's.
    """
    repo.commit({"backend/app/main.py": "app = 'broken'\n"}, "B")
    c = repo.commit({"backend/app/main.py": "app = None\n"}, "C reverts B")
    decision = _decide(repo, c, deployed, space_head="6" * 40)
    assert decision.deploy
    assert "6" * 40 in decision.reason
    assert SPACE_A in decision.reason


def test_space_changed_outside_the_workflow_deploys(repo: Repo, deployed: str) -> None:
    head = repo.commit({"docs/TASKS.md": "docs\n"})
    assert _decide(repo, head, deployed, space_head="7" * 40).deploy


def test_file_moved_out_of_an_image_directory_deploys(repo: Repo, deployed: str) -> None:
    repo.git("mv", "backend/app/main.py", "docs/main.py")
    head = repo.commit({})
    decision = _decide(repo, head, deployed)
    assert decision.deploy
    assert decision.image_changes == ("backend/app/main.py",)


def test_change_reverted_before_it_was_pushed_skips(repo: Repo, deployed: str) -> None:
    repo.commit({"backend/app/main.py": "app = 'broken'\n"})
    head = repo.commit({"backend/app/main.py": "app = None\n", "docs/TASKS.md": "reverted\n"})
    assert not _decide(repo, head, deployed).deploy


def test_redeploying_the_deployed_commit_skips(repo: Repo, deployed: str) -> None:
    assert not _decide(repo, deployed, deployed).deploy


def test_no_recorded_deploy_deploys(repo: Repo, deployed: str) -> None:
    decision = scope.decide(deployed, None, None, repo.path)
    assert decision.deploy
    assert "no successful deploy is recorded" in decision.reason


def test_baseline_missing_from_history_deploys(repo: Repo, deployed: str) -> None:
    decision = _decide(repo, deployed, "f" * 40)
    assert decision.deploy
    assert "isn't in this checkout" in decision.reason


def test_failed_comparison_deploys(
    repo: Repo, deployed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(base: str, head: str, repo_path: Path) -> list[str]:
        raise RuntimeError("git diff failed: bad object")

    monkeypatch.setattr(scope, "changed_paths", broken)
    decision = _decide(repo, deployed, deployed)
    assert decision.deploy
    assert "bad object" in decision.reason


# ------------------------------------------------------- the deploy record


def _deployment(
    deployment_id: int,
    sha: str,
    creator: str = scope.RECORDER,
    space_commit: str | None = SPACE_A,
) -> dict[str, Any]:
    payload = {"space_commit": space_commit} if space_commit else {}
    return {"id": deployment_id, "sha": sha, "creator": {"login": creator}, "payload": payload}


def _fetch(deployments: list[dict[str, Any]], statuses: dict[int, list[str]]) -> scope.Fetch:
    def fetch(path: str) -> Any:
        if "/statuses" in path:
            deployment_id = int(path.split("/deployments/")[1].split("/")[0])
            return [{"state": state} for state in statuses.get(deployment_id, [])]
        assert f"environment={scope.ENVIRONMENT}" in path
        return deployments

    return fetch


def test_baseline_is_the_newest_record_with_its_space_commit() -> None:
    a, b = "a" * 40, "b" * 40
    fetch = _fetch([_deployment(1, a), _deployment(2, b)], {1: ["success"], 2: ["success"]})
    assert scope.find_baseline(fetch, "owner/repo") == scope.Baseline(b, SPACE_A)


def test_a_record_without_success_is_not_trusted() -> None:
    """Its status write failed, so whether the Space is healthy on it is unknown."""
    a, b = "a" * 40, "b" * 40
    fetch = _fetch([_deployment(2, b), _deployment(1, a)], {2: ["in_progress"], 1: ["success"]})
    assert scope.find_baseline(fetch, "owner/repo") is None


def test_records_written_by_anyone_else_are_ignored() -> None:
    a, v = "a" * 40, "9" * 40
    fetch = _fetch(
        [_deployment(2, v, creator="vercel[bot]"), _deployment(1, a)],
        {2: ["success"], 1: ["success"]},
    )
    assert scope.find_baseline(fetch, "owner/repo") == scope.Baseline(a, SPACE_A)


def test_no_records_means_no_baseline() -> None:
    assert scope.find_baseline(_fetch([], {}), "owner/repo") is None


def test_malformed_records_are_unavailable_not_trusted() -> None:
    with pytest.raises(scope.BaselineUnavailable):
        scope.find_baseline(lambda path: {"message": "Not Found"}, "owner/repo")
    bad_sha = _fetch([_deployment(1, "--upload-pack=x")], {1: ["success"]})
    with pytest.raises(scope.BaselineUnavailable):
        scope.find_baseline(bad_sha, "owner/repo")
    no_space_commit = _fetch([_deployment(1, "a" * 40, space_commit=None)], {1: ["success"]})
    with pytest.raises(scope.BaselineUnavailable):
        scope.find_baseline(no_space_commit, "owner/repo")


def test_record_creates_a_deployment_then_its_success_status() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def post(path: str, body: dict[str, Any]) -> Any:
        calls.append((path, body))
        return {"id": 77}

    sha, space = "a" * 40, "b" * 40
    deployment_id = scope.record_deploy(post, "owner/repo", sha, space, "o/s", "https://log")
    assert deployment_id == 77
    (create_path, create), (status_path, status) = calls
    assert create_path == "/repos/owner/repo/deployments"
    assert create["ref"] == sha
    assert create["environment"] == scope.ENVIRONMENT
    assert create["auto_merge"] is False
    assert create["required_contexts"] == []
    assert create["payload"] == {"space_commit": space}
    assert status_path == "/repos/owner/repo/deployments/77/statuses"
    assert status["state"] == "success"
    assert status["environment_url"] == "https://huggingface.co/spaces/o/s"


# ------------------------------------------------------------------- the CLI


@pytest.fixture
def actions_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    files = {"GITHUB_OUTPUT": tmp_path / "output", "GITHUB_STEP_SUMMARY": tmp_path / "summary"}
    for name, path in files.items():
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    return files


def _fake_github(deployments: list[dict[str, Any]], statuses: dict[int, list[str]]) -> Any:
    fetch = _fetch(deployments, statuses)

    def github(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        assert method == "GET"
        return fetch(path)

    return github


def _unreachable(*args: Any, **kwargs: Any) -> Any:
    raise urllib.error.URLError("unreachable")


def _decide_cli(head: str, *extra: str) -> int:
    return scope.main(["decide", "--head", head, "--space", "owner/space", *extra])


def test_cli_skip_writes_outputs_summary_and_notice(
    repo: Repo,
    deployed: str,
    actions_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    head = repo.commit({"docs/TASKS.md": "docs\n"})
    monkeypatch.chdir(repo.path)
    monkeypatch.setattr(
        scope, "_github", _fake_github([_deployment(1, deployed)], {1: ["success"]})
    )
    monkeypatch.setattr(scope, "_space_head", lambda space: SPACE_A)
    assert _decide_cli(head) == 0
    output = actions_env["GITHUB_OUTPUT"].read_text(encoding="utf-8").splitlines()
    assert output == ["deploy=false", f"base={deployed}"]
    summary = actions_env["GITHUB_STEP_SUMMARY"].read_text(encoding="utf-8")
    assert "### Space deploy: skipped" in summary
    assert "::notice title=Space deploy skipped::No image input changed" in capsys.readouterr().out


def test_cli_deploys_when_the_record_or_the_space_cannot_be_read(
    repo: Repo,
    deployed: str,
    actions_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def odd_shape(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if "/statuses" in path:
            return ["success"]  # strings, not status objects
        return [_deployment(1, deployed)]

    def space_ok(space: str) -> str:
        return SPACE_A

    head = repo.commit({"docs/TASKS.md": "docs\n"})
    monkeypatch.chdir(repo.path)
    good = _fake_github([_deployment(1, deployed)], {1: ["success"]})
    for github, space_head in (
        (_unreachable, space_ok),
        (odd_shape, space_ok),
        (good, _unreachable),
    ):
        actions_env["GITHUB_OUTPUT"].write_text("", encoding="utf-8")
        monkeypatch.setattr(scope, "_github", github)
        monkeypatch.setattr(scope, "_space_head", space_head)
        assert _decide_cli(head) == 0
        assert "deploy=true" in actions_env["GITHUB_OUTPUT"].read_text(encoding="utf-8")
        out = capsys.readouterr().out
        assert "::warning::the deploy record or the Space's head couldn't be read" in out


def test_cli_manual_run_deploys_without_reading_anything(
    actions_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scope, "_github", _unreachable)
    monkeypatch.setattr(scope, "_space_head", _unreachable)
    monkeypatch.setattr(scope, "changed_paths", _unreachable)
    assert _decide_cli("a" * 40, "--force") == 0
    output = actions_env["GITHUB_OUTPUT"].read_text(encoding="utf-8").splitlines()
    assert output == ["deploy=true", "base="]


def test_cli_refuses_to_decide_without_a_step_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert _decide_cli("a" * 40, "--force") == 1
    assert "::error::GITHUB_OUTPUT is not set" in capsys.readouterr().out


def test_cli_record_failure_fails_the_step(
    actions_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(scope, "_github", _unreachable)
    argv = ["record", "--sha", "a" * 40, "--space-commit", "b" * 40, "--space", "o/s"]
    assert scope.main(argv) == 1
    assert "::error::The Space is healthy" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["decide", "--head", "--upload-pack=touch x", "--space", "o/s"],
        ["decide", "--head", "HEAD", "--space", "o/s"],
        ["decide", "--head", "A" * 40, "--space", "o/s"],
        ["decide", "--head", "a" * 39, "--space", "o/s"],
        ["decide", "--head", "a" * 40, "--space", "../../etc"],
        ["record", "--sha", "a" * 40, "--space-commit", "main", "--space", "o/s"],
    ],
)
def test_cli_accepts_only_full_commit_shas_and_space_ids(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        scope.main(argv)
    assert exc.value.code == 2


# -------------------------------------------------------- the deploy workflow


def _workflow() -> dict[Any, Any]:
    workflow: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow


def _steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _workflow()["jobs"]["push-to-space"]["steps"]
    return steps


def _index(steps: list[dict[str, Any]], *, step_id: str = "", run: str = "") -> int:
    for i, step in enumerate(steps):
        if (step_id and step.get("id") == step_id) or (run and run in step.get("run", "")):
            return i
    raise AssertionError(f"no step matching id={step_id!r} run={run!r}")


def test_decision_runs_only_for_commits_the_superseded_guard_lets_through() -> None:
    steps = _steps()
    guard, decision = _index(steps, step_id="guard"), _index(steps, step_id="scope")
    assert guard < decision
    assert steps[decision]["if"] == "steps.guard.outputs.skip != 'true'"
    assert "python3 -m scripts.deploy_scope decide" in steps[decision]["run"]
    assert '--space "$HF_USERNAME/$HF_SPACE"' in steps[decision]["run"]


def test_every_step_after_the_decision_needs_it_and_stops_on_failure() -> None:
    steps = _steps()
    after = steps[_index(steps, step_id="scope") + 1 :]
    assert after
    for step in after:
        condition = step.get("if", "")
        assert condition == "steps.scope.outputs.deploy == 'true'", step["name"]
        assert not STATUS_FUNCTIONS.search(condition), step["name"]


def test_baseline_is_recorded_last_after_the_health_gate() -> None:
    steps = _steps()
    snapshot = _index(steps, run="python3 -m scripts.space_snapshot")
    push = _index(steps, run="push --force")
    health = _index(steps, run="model_loaded")
    record = _index(steps, run="scripts.deploy_scope record")
    assert snapshot < push < health < record == len(steps) - 1
    for argument in ('--sha "$DEPLOY_SHA"', '--space-commit "$DEPLOY_COMMIT"', "--space "):
        assert argument in steps[record]["run"]
    # The health gate waits for the Space to be on the snapshot that was pushed.
    assert 'EXPECTED = os.environ["DEPLOY_COMMIT"]' in steps[health]["run"]


def test_the_space_receives_the_snapshot_of_the_tested_commit_only() -> None:
    steps = _steps()
    build = steps[_index(steps, run="python3 -m scripts.space_snapshot")]
    assert '--sha "$DEPLOY_SHA"' in build["run"]
    assert '--dest "$RUNNER_TEMP/space-snapshot"' in build["run"]
    assert "secrets." not in json.dumps(build)
    push = steps[_index(steps, run="push --force")]["run"]
    # Exactly the verified snapshot commit, from the snapshot's repository, never HEAD.
    assert 'git -C "$SNAPSHOT_DIR" push --force' in push
    assert '"${DEPLOY_COMMIT}:refs/heads/main"' in push
    assert "HEAD:main" not in push
    # An empty DEPLOY_COMMIT would make the push delete the Space's main.
    assert "grep -Eqx '[0-9a-f]{40}'" in push
    assert push.index("grep -Eqx") < push.index("push --force")
    # The token never sits in a URL, a git config or the checkout.
    assert "GIT_ASKPASS=" in push and "HF_TOKEN}@" not in push
    for step in steps:
        run = step.get("run", "")
        for command in ("git remote add", "git commit", "git config", "git add "):
            assert command not in run, f"{step['name']} runs {command!r} in the checkout"


def test_manual_runs_stay_ci_verified_and_force_a_deploy() -> None:
    steps = _steps()
    verify = steps[0]
    assert verify["if"] == "github.event_name == 'workflow_dispatch'"
    assert "conclusion" in verify["run"] and "refusing to deploy" in verify["run"]
    decision = steps[_index(steps, step_id="scope")]["run"]
    assert 'if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]' in decision
    assert "--force" in decision


def test_permissions_concurrency_and_token_exposure() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {
        "contents": "read",
        "actions": "read",
        "deployments": "write",
    }
    assert workflow["concurrency"] == {"group": "deploy-backend", "cancel-in-progress": False}
    job = workflow["jobs"]["push-to-space"]
    assert job["env"]["DEPLOY_SHA"] == "${{ github.event.workflow_run.head_sha || github.sha }}"
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["persist-credentials"] is False
    assert json.dumps(job).count("secrets.HF_TOKEN") == 2  # the push and the health gate only
