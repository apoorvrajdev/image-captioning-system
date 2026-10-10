"""Tests for the single-commit Space snapshot (TASK-025, ADR-030).

The snapshot is built by the real ``git`` plumbing from throwaway repositories, so what
these tests inspect is what the deploy would push. One test builds the snapshot of this
repository's own HEAD, so the build-context list can't drift from the real tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import scripts.deploy_scope as scope
import scripts.space_snapshot as snap

from tests.unit.test_deploy_scope import REPO_ROOT, Repo

TOKEN = "hf_SENTINEL_never_logged"  # a stand-in secret that must never be printed

BUILD_FILES = {
    "README.md": "# Image Captioning System\n\nThe GitHub README.\n",
    "requirements.txt": "fastapi==0.133.0\n",
    "pyproject.toml": "[project]\nname = 'captioning'\nreadme = 'README.md'\n",
    "Dockerfile": "FROM python:3.11-slim-bookworm\nCOPY src/ ./src/\n",
    ".dockerignore": "docs\ntests\n",
    ".gitattributes": "*.h5 filter=lfs diff=lfs merge=lfs -text\n",
    "src/captioning/__init__.py": "",
    "src/captioning/py.typed": "",
    "backend/app/main.py": "app = None\n",
    "backend/app/tests/test_routes.py": "",  # backend/ is copied whole
    "configs/base.yaml": "seed: 42\n",
    "models/v1.0.0/vocab.json": '["a", "man"]\n',
}
OTHER_FILES: dict[str, str | bytes] = {
    "docs/demo/image-captioning-demo.mp4": b"\x00\x00\x00\x18ftypmp42\x00binary",
    "docs/demo/image-captioning-demo.jpg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00binary",
    "docs/demo/README.md": "# Product demo video\n",
    "docs/TASKS.md": "# Tasks\n",
    "results/stabilized-greedy/metrics.json": "{}\n",
    "notebooks/01_ieee_inceptionv3_transformer.ipynb": "{}\n",
    "frontend/src/App.jsx": "export default 1\n",
    "frontend/src/assets/hero.png": b"\x89PNG\r\n\x1a\n\x00binary",
    "tests/unit/test_x.py": "",
    "scripts/deploy_scope.py": "",
    ".github/workflows/deploy-backend.yml": "name: deploy\n",
    ".env.example": "HF_TOKEN=\n",
    "requirements-dev.txt": "pytest\n",
}


def _commit(repo: Repo, files: dict[str, str | bytes | None], message: str = "change") -> str:
    """Like ``Repo.commit``, but also writes bytes, for binary files."""
    for name, content in files.items():
        target = repo.path / name
        if content is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_bytes(content.encode("utf-8"))
    repo.git("add", "-A")
    repo.git("commit", "-q", "--no-verify", "--allow-empty", "-m", message)
    return repo.git("rev-parse", "HEAD")


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True, encoding="utf-8"
    ).stdout


@pytest.fixture(autouse=True)
def _identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deploy step's git identity, set the same way: through git's variables."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "apoorvrajdev")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "apoorvrajmgr@gmail.com")


@pytest.fixture
def source(tmp_path: Path) -> tuple[Repo, str]:
    """A repository with history: the build context, plus everything that must stay out."""
    (tmp_path / "repo").mkdir()
    repo = Repo(tmp_path / "repo")
    _commit(repo, {**BUILD_FILES, **OTHER_FILES}, "first")
    tested = _commit(repo, {"docs/TASKS.md": "# Tasks\n\nTASK-025\n"}, "second")
    return repo, tested


def _build(repo: Repo, sha: str, tmp_path: Path, name: str = "snapshot") -> snap.Snapshot:
    return snap.build(repo.path, sha, tmp_path / name, source_url=f"https://x.invalid/{sha}")


# ------------------------------------------------------------------- contents


def test_the_snapshot_holds_exactly_the_build_context(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, tested = source
    snapshot = _build(repo, tested, tmp_path)
    assert set(snapshot.paths) == set(BUILD_FILES)
    assert snapshot.excluded == len(OTHER_FILES)


@pytest.mark.parametrize("path", sorted(OTHER_FILES))
def test_docs_demo_media_results_and_dev_files_are_left_out(
    source: tuple[Repo, str], tmp_path: Path, path: str
) -> None:
    repo, tested = source
    assert path not in _build(repo, tested, tmp_path).paths


def test_untracked_local_files_never_enter_the_snapshot(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, tested = source
    for name in (
        ".env",
        ".venv/lib/site.py",
        "src/captioning/__pycache__/x.cpython-310.pyc",
        "models/v1.0.0/model.h5",
        "outputs/runs/latest/history.json",
        "backend/app/local_notes.txt",
    ):
        (repo.path / name).parent.mkdir(parents=True, exist_ok=True)
        (repo.path / name).write_text(TOKEN, encoding="utf-8")
    snapshot = _build(repo, tested, tmp_path)
    assert set(snapshot.paths) == set(BUILD_FILES)
    # git grep exits 1 when nothing in the snapshot's tree contains the local files' text.
    found = subprocess.run(
        ["git", "grep", "-q", "-F", TOKEN, snapshot.commit], cwd=tmp_path / "snapshot"
    )
    assert found.returncode == 1


def test_file_contents_and_executable_bits_are_preserved(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, _ = source
    (repo.path / "backend/start.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    repo.git("add", "backend/start.sh")
    repo.git("update-index", "--chmod=+x", "backend/start.sh")
    repo.git("commit", "-q", "--no-verify", "-m", "script")
    tested = repo.git("rev-parse", "HEAD")
    snapshot = _build(repo, tested, tmp_path)
    original = repo.git("ls-tree", "-r", tested).splitlines()
    deployed = _git(tmp_path / "snapshot", "ls-tree", "-r", snapshot.commit).splitlines()
    readme = [line for line in deployed if line.endswith("\tREADME.md")]
    # Every file but README.md is the same blob, with the same mode, as in the tested commit.
    assert set(deployed) - set(readme) <= set(original)
    assert any(
        line.startswith("100755 ") and line.endswith("\tbackend/start.sh") for line in deployed
    )


def test_readme_gets_the_space_config_header(source: tuple[Repo, str], tmp_path: Path) -> None:
    repo, tested = source
    snapshot = _build(repo, tested, tmp_path)
    assert snapshot.header_added
    readme = _git(tmp_path / "snapshot", "show", f"{snapshot.commit}:README.md")
    assert readme == snap.SPACE_HEADER + BUILD_FILES["README.md"]
    assert "sdk: docker\napp_port: 7860\n" in snap.SPACE_HEADER


def test_readme_with_front_matter_is_deployed_unchanged(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, _ = source
    front = "---\ntitle: custom\n---\n# Readme\n"
    tested = _commit(repo, {"README.md": front})
    snapshot = _build(repo, tested, tmp_path)
    assert not snapshot.header_added
    assert _git(tmp_path / "snapshot", "show", f"{snapshot.commit}:README.md") == front


# ---------------------------------------------------------------- the commit


def test_the_snapshot_is_one_commit_with_no_parents(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, tested = source
    assert int(repo.git("rev-list", "--count", tested)) == 2  # the source has history
    snapshot = _build(repo, tested, tmp_path)
    dest = tmp_path / "snapshot"
    assert _git(dest, "rev-list", "--count", snapshot.commit).strip() == "1"
    assert "\nparent " not in _git(dest, "cat-file", "-p", snapshot.commit)
    assert _git(dest, "rev-parse", "refs/heads/main").strip() == snapshot.commit


def test_provenance_names_the_exact_tested_commit_not_the_branch_tip(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, tested = source
    newer = _commit(repo, {"backend/app/main.py": "app = 'newer'\n"}, "newer, untested")
    snapshot = _build(repo, tested, tmp_path)
    dest = tmp_path / "snapshot"
    message = _git(dest, "log", "-1", "--format=%B", snapshot.commit)
    assert f"Source-Commit: {tested}" in message
    assert newer not in message and snapshot.source == tested
    assert _git(dest, "show", f"{snapshot.commit}:backend/app/main.py") == "app = None\n"
    author = _git(dest, "log", "-1", "--format=%an <%ae>", snapshot.commit).strip()
    assert author == "apoorvrajdev <apoorvrajmgr@gmail.com>"


def test_the_checkout_is_not_modified(source: tuple[Repo, str], tmp_path: Path) -> None:
    repo, tested = source

    def state() -> tuple[str, ...]:
        return (
            repo.git("status", "--porcelain", "--ignored"),
            repo.git("show-ref"),
            repo.git("count-objects", "-v"),
            (repo.path / ".git" / "index").read_bytes().hex(),
            (repo.path / ".git" / "config").read_text(encoding="utf-8"),
        )

    before = state()
    _build(repo, tested, tmp_path)
    assert state() == before


# ----------------------------------------------------------------- refusals


@pytest.mark.parametrize("missing", ["Dockerfile", ".dockerignore", "configs/base.yaml"])
def test_a_missing_required_input_refuses_the_build(
    source: tuple[Repo, str], tmp_path: Path, missing: str
) -> None:
    repo, _ = source
    tested = _commit(repo, {missing: None})
    expected = "configs/" if missing.startswith("configs/") else missing
    with pytest.raises(snap.SnapshotError, match=f"lacks required build inputs.*{expected}"):
        _build(repo, tested, tmp_path)
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("path", "content", "reason"),
    [
        ("backend/app/static/logo.png", b"\x89PNG\r\n\x1a\n\x00", "is binary"),
        (
            "models/v2.0.0/model.h5",
            "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n",
            "Git LFS pointer",
        ),
        ("backend/.env", f"HF_TOKEN={TOKEN}\n", "secret, key or cache"),
        ("configs/server.pem", "-----BEGIN-----\n", "secret, key or cache"),
        ("src/captioning/__pycache__/x.cpython-310.pyc", "cache", "secret, key or cache"),
    ],
)
def test_undeployable_files_in_the_build_context_refuse_the_build(
    source: tuple[Repo, str], tmp_path: Path, path: str, content: str | bytes, reason: str
) -> None:
    repo, _ = source
    tested = _commit(repo, {path: content})
    with pytest.raises(snap.SnapshotError, match=reason) as excinfo:
        _build(repo, tested, tmp_path)
    assert TOKEN not in str(excinfo.value)


def test_the_snapshot_must_be_built_outside_the_checkout_in_an_empty_directory(
    source: tuple[Repo, str], tmp_path: Path
) -> None:
    repo, tested = source
    with pytest.raises(snap.SnapshotError, match="outside the checkout"):
        snap.build(repo.path, tested, repo.path / "snapshot")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "x").write_text("", encoding="utf-8")
    with pytest.raises(snap.SnapshotError, match="not empty"):
        snap.build(repo.path, tested, occupied)
    with pytest.raises(snap.SnapshotError, match="not a full"):
        snap.build(repo.path, tested[:12], tmp_path / "short")


# ---------------------------------------------------------- this repository


def test_this_repositorys_head_snapshots_without_its_demo_media(tmp_path: Path) -> None:
    head = _git(REPO_ROOT, "rev-parse", "HEAD").strip()
    snapshot = snap.build(REPO_ROOT, head, tmp_path / "snapshot")
    assert not snap.missing_inputs(snapshot.paths)
    assert not any(path.startswith("docs/") for path in snapshot.paths)
    assert all(scope.is_build_context(path) for path in snapshot.paths)
    assert "Dockerfile" in snapshot.paths and "README.md" in snapshot.paths
    assert _git(tmp_path / "snapshot", "rev-list", "--count", snapshot.commit).strip() == "1"


# ------------------------------------------------------------------- the CLI


def test_cli_hands_the_commit_to_the_push_step_and_prints_no_secret(
    source: tuple[Repo, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, tested = source
    env_file, summary = tmp_path / "github_env", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", "apoorvrajdev/image-captioning-system")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.chdir(repo.path)
    dest = tmp_path / "space-snapshot"
    assert snap.main(["--sha", tested, "--dest", str(dest)]) == 0

    commit = _git(dest, "rev-parse", "refs/heads/main").strip()
    assert env_file.read_text(encoding="utf-8").splitlines() == [
        f"DEPLOY_COMMIT={commit}",
        f"SNAPSHOT_DIR={dest.resolve()}",
    ]
    message = _git(dest, "log", "-1", "--format=%B", commit)
    assert f"github.com/apoorvrajdev/image-captioning-system/commit/{tested}" in message
    assert (
        "Deploy-Run: https://github.com/apoorvrajdev/image-captioning-system/actions/runs/123"
        in message
    )
    out = capsys.readouterr()
    assert f"Snapshot {commit} of {tested}" in out.out and "no parents" in out.out
    for text in (out.out, out.err, summary.read_text(encoding="utf-8"), message):
        assert TOKEN not in text


def test_cli_refusal_fails_the_step_and_writes_no_commit(
    source: tuple[Repo, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = source
    tested = _commit(repo, {"backend/app/static/logo.png": b"\x89PNG\x00"})
    env_file = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.chdir(repo.path)
    assert snap.main(["--sha", tested, "--dest", str(tmp_path / "space-snapshot")]) == 1
    assert "::error::" in capsys.readouterr().out
    assert not env_file.exists()  # the push step finds no DEPLOY_COMMIT and refuses
