"""Build the single-commit snapshot of a tested commit that deploys to the HF Space.

Usage (from ``.github/workflows/deploy-backend.yml``, in the checkout of the tested commit):
    python3 -m scripts.space_snapshot --sha <tested commit> --dest <empty directory>

The Space receives one root commit holding only the build context of the commit CI tested
(``BUILD_CONTEXT`` in ``scripts/deploy_scope.py``: the Dockerfile's COPY sources and the
build files), with the Space's YAML config header prepended to ``README.md``. No GitHub
history goes with it. Hugging Face refuses a push that carries binary files outside
Xet/LFS, and GitHub's history holds some (the README demo video), so pushing history
breaks every deploy (ADR-030).

The snapshot is assembled in a bare repository of its own at ``--dest``. It reads the
tested commit's objects from the checkout through git's alternates mechanism and writes
new objects (the README with the header, the trees, the commit) only under ``--dest``. The
checkout's files, index, refs and object store are left as they were. The commit message
names the tested commit (``Source-Commit:``). Before anything is pushed, the script
refuses an incomplete build context, a non-regular file, a binary or Git LFS file (the
Space can't take either), and a secret- or cache-like path. It uses only the standard
library, so the deploy job runs it on the runner's interpreter.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.deploy_scope import BUILD_CONTEXT, is_build_context

# The Space reads its build config from YAML front matter at the top of README.md. GitHub's
# README has none (removed in befac80, because GitHub renders it as a table), so the
# deployed copy gets this header: the one removed in befac80, unchanged.
SPACE_HEADER = (
    "---\n"
    "title: Image Captioning API\n"
    "emoji: \U0001f5bc\ufe0f\n"
    "colorFrom: blue\n"
    "colorTo: indigo\n"
    "sdk: docker\n"
    "app_port: 7860\n"
    "pinned: false\n"
    "license: mit\n"
    "short_description: InceptionV3 + Transformer image captioning inference API\n"
    "---\n"
    "\n"
)

REGULAR_MODES = frozenset({"100644", "100755"})
# Never deployed, even when tracked under the build context: secrets, keys and caches.
FORBIDDEN_NAMES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "*.pyc")
FORBIDDEN_DIRS = frozenset({"__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"})
ALLOWED_NAMES = frozenset({".env.example"})
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
BINARY_PROBE = 8000  # bytes git itself inspects for a NUL when it classifies a file as binary

_SHA = re.compile(r"[0-9a-f]{40}")


class SnapshotError(Exception):
    """The snapshot can't be built safely, so nothing may be pushed."""


@dataclass(frozen=True)
class Entry:
    """One file of the tested commit's tree."""

    mode: str
    kind: str
    oid: str
    path: str


@dataclass(frozen=True)
class Snapshot:
    commit: str  # the root commit to push to the Space
    source: str  # the tested GitHub commit it was built from
    paths: tuple[str, ...]
    excluded: int  # tracked files of the tested commit left out
    header_added: bool


def _git(cwd: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SnapshotError(f"git {args[0]} failed: {detail}")
    return result.stdout


def tracked_entries(repo: Path, sha: str) -> list[Entry]:
    """Every entry of the commit's tree, recursively, as git stores it."""
    entries = []
    for record in _git(repo, "ls-tree", "-r", "-z", "--full-tree", sha).split(b"\0"):
        if record:
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split()
            entries.append(Entry(mode, kind, oid, path.decode("utf-8")))
    return entries


def missing_inputs(paths: Iterable[str]) -> list[str]:
    """``BUILD_CONTEXT`` entries with nothing behind them in ``paths``."""
    present = set(paths)
    return [
        entry
        for entry in BUILD_CONTEXT
        if not (
            any(path.startswith(entry) for path in present)
            if entry.endswith("/")
            else entry in present
        )
    ]


def is_forbidden(path: str) -> bool:
    """True for secret-, key- or cache-like paths, which never go to the Space."""
    *dirs, name = path.split("/")
    if name in ALLOWED_NAMES:
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in FORBIDDEN_NAMES) or any(
        part in FORBIDDEN_DIRS for part in dirs
    )


def _read_blobs(repo: Path, oids: Sequence[str]) -> dict[str, bytes]:
    """The content of each blob, read in one ``git cat-file --batch`` call."""
    if not oids:
        return {}
    out = _git(repo, "cat-file", "--batch", stdin=("\n".join(oids) + "\n").encode("ascii"))
    blobs: dict[str, bytes] = {}
    position = 0
    for oid in oids:
        end = out.index(b"\n", position)
        header = out[position:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise SnapshotError(f"object {oid} is not a readable blob")
        size = int(header[2])
        blobs[oid] = out[end + 1 : end + 1 + size]
        position = end + 1 + size + 1  # the content is followed by a newline
    return blobs


def _problems(entries: Sequence[Entry], blobs: dict[str, bytes]) -> list[str]:
    problems = []
    for entry in entries:
        if entry.kind != "blob" or entry.mode not in REGULAR_MODES:
            problems.append(f"{entry.path} is not a regular file (mode {entry.mode})")
            continue
        if is_forbidden(entry.path):
            problems.append(f"{entry.path} looks like a secret, key or cache")
        content = blobs[entry.oid]
        if content.startswith(LFS_POINTER):
            problems.append(f"{entry.path} is a Git LFS pointer, which the snapshot can't carry")
        elif b"\0" in content[:BINARY_PROBE]:
            problems.append(f"{entry.path} is binary, which the Space refuses outside Xet/LFS")
    return problems


def _write_tree(dest: Path, entries: Sequence[Entry]) -> str:
    """Write the nested trees for ``entries`` into ``dest``; returns the root tree."""
    root: dict[str, object] = {}
    for entry in entries:
        *dirs, name = entry.path.split("/")
        node = root
        for part in dirs:
            child = node.setdefault(part, {})
            assert isinstance(child, dict)
            node = child
        node[name] = entry

    def write(node: dict[str, object]) -> str:
        records = []
        for name, child in node.items():
            if isinstance(child, dict):
                records.append(f"040000 tree {write(child)}\t{name}")
            else:
                assert isinstance(child, Entry)
                records.append(f"{child.mode} blob {child.oid}\t{name}")
        payload = "".join(f"{record}\0" for record in records).encode("utf-8")
        return _git(dest, "mktree", "-z", stdin=payload).decode("ascii").strip()

    return write(root)


def _message(sha: str, files: int, source_url: str, run_url: str) -> str:
    lines = [
        f"deploy: {sha} with Space config header",
        "",
        f"A single-commit snapshot of the Space build context ({files} files) of",
        f"{source_url or sha}, with no GitHub history (ADR-030).",
        "",
        f"Source-Commit: {sha}",
    ]
    if run_url:
        lines.append(f"Deploy-Run: {run_url}")
    return "\n".join(lines) + "\n"


def build(repo: Path, sha: str, dest: Path, *, source_url: str = "", run_url: str = "") -> Snapshot:
    """Build the snapshot of ``sha`` in a new bare repository at ``dest``.

    The commit author and committer come from git's usual identity variables
    (``GIT_AUTHOR_NAME``, ``GIT_COMMITTER_EMAIL``, ...). Raises ``SnapshotError``, before
    anything could be pushed, when the snapshot would be incomplete or undeployable.
    """
    if not _SHA.fullmatch(sha):
        raise SnapshotError(f"not a full lowercase commit SHA: {sha!r}")
    repo = repo.resolve()
    dest = dest.resolve()
    toplevel = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if dest == toplevel or toplevel in dest.parents:
        raise SnapshotError(f"the snapshot must be built outside the checkout, not in {dest}")
    if dest.exists() and any(dest.iterdir()):
        raise SnapshotError(f"{dest} is not empty")
    _git(repo, "cat-file", "-e", f"{sha}^{{commit}}")

    every = tracked_entries(repo, sha)
    chosen = [entry for entry in every if is_build_context(entry.path)]
    missing = missing_inputs(entry.path for entry in chosen)
    if missing:
        raise SnapshotError(
            f"{sha} lacks required build inputs ({', '.join(missing)}); refusing an "
            "incomplete snapshot"
        )
    blobs = _read_blobs(repo, [entry.oid for entry in chosen if entry.kind == "blob"])
    problems = _problems(chosen, blobs)
    if problems:
        raise SnapshotError("refusing to build the snapshot: " + "; ".join(problems))

    # A bare repository of its own. It reads the checkout's objects through alternates
    # and writes only under dest. write_bytes: write_text would add CRs on Windows.
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").decode().strip())
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "-q", "--bare", ".")
    alternates = dest / "objects" / "info" / "alternates"
    alternates.write_bytes(f"{(git_dir / 'objects').as_posix()}\n".encode())

    readme = next(entry for entry in chosen if entry.path == "README.md")
    text = blobs[readme.oid]
    header_added = text.split(b"\n", 1)[0] != b"---"
    if header_added:
        oid = _git(
            dest,
            "hash-object",
            "-w",
            "--no-filters",
            "--stdin",
            stdin=SPACE_HEADER.encode("utf-8") + text,
        )
        readme = Entry(readme.mode, readme.kind, oid.decode("ascii").strip(), readme.path)
        chosen = [readme if entry.path == "README.md" else entry for entry in chosen]

    tree = _write_tree(dest, chosen)
    message = _message(sha, len(chosen), source_url, run_url)
    commit = (
        _git(
            dest,
            "-c",
            "commit.gpgsign=false",
            "commit-tree",
            tree,
            "-F",
            "-",
            stdin=message.encode(),
        )
        .decode("ascii")
        .strip()
    )
    _git(dest, "update-ref", "refs/heads/main", commit)

    # What will be pushed: one commit, no parents, exactly the build context.
    if _git(dest, "rev-list", "--count", commit).decode().strip() != "1":
        raise SnapshotError(f"snapshot {commit} carries history")
    if _git(dest, "rev-list", "--parents", "-n", "1", commit).decode().split() != [commit]:
        raise SnapshotError(f"snapshot {commit} has a parent")
    listed = _git(dest, "ls-tree", "-r", "-z", "--name-only", commit).decode("utf-8")
    paths = tuple(sorted(path for path in listed.split("\0") if path))
    if paths != tuple(sorted(entry.path for entry in chosen)):
        raise SnapshotError(f"snapshot {commit} doesn't hold exactly the build context")
    return Snapshot(commit, sha, paths, len(every) - len(chosen), header_added)


def _append(env_var: str, lines: Iterable[str]) -> None:
    with Path(os.environ[env_var]).open("a", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)


def _sha(value: str) -> str:
    if not _SHA.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a full lowercase commit SHA: {value!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the single-commit Space snapshot of a tested commit."
    )
    parser.add_argument("--sha", type=_sha, required=True, help="the commit CI tested")
    parser.add_argument("--dest", type=Path, required=True, help="an empty directory")
    args = parser.parse_args(argv)

    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    source_url = f"{server}/{repository}/commit/{args.sha}" if repository else ""
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"{server}/{repository}/actions/runs/{run_id}" if repository and run_id else ""
    try:
        snapshot = build(Path.cwd(), args.sha, args.dest, source_url=source_url, run_url=run_url)
    except SnapshotError as exc:
        print(f"::error::{exc}")
        return 1

    print(
        f"Snapshot {snapshot.commit} of {snapshot.source}: {len(snapshot.paths)} build-context "
        f"files, {snapshot.excluded} other tracked files left out, one commit with no parents."
    )
    if not snapshot.header_added:
        print("::notice::README.md already starts with front matter; deploying it unchanged.")
    if os.environ.get("GITHUB_ENV"):
        _append(
            "GITHUB_ENV",
            [f"DEPLOY_COMMIT={snapshot.commit}", f"SNAPSHOT_DIR={args.dest.resolve()}"],
        )
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        _append(
            "GITHUB_STEP_SUMMARY",
            [
                "### Space snapshot",
                "",
                f"Built `{snapshot.commit}` from `{snapshot.source}`: one commit, no parents.",
                f"{len(snapshot.paths)} build-context files deployed; {snapshot.excluded} other "
                "tracked files (docs, tests, frontend, results, notebooks, ...) left out.",
            ],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
