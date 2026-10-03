"""Rebuild the code index under .claude/context/. Stdlib only; safe to re-run.

Usage (from any directory):
    .venv/Scripts/python.exe .claude/context/build_index.py   # Windows venv
    python .claude/context/build_index.py                     # any Python >= 3.10
Also run automatically at session start by the hook in .claude/settings.json.

Outputs (all regenerated and gitignored, never hand-edit):
    files.txt        tracked files (git ls-files)
    dir-weights.txt  tracked-file count per top-level/second-level directory
    symbols.tsv      name<TAB>kind<TAB>file<TAB>line  (Python via ast, JS/JSX via regex)
    deps.json        internal import graph: module -> {file, imports, imported_by}
    hotspots.txt     most-changed files over the last 100 commits (git log)

repo-map.md is hand-written and is NOT touched by this script.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

INTERNAL_PY_ROOTS = ("captioning", "app", "scripts", "tests")
JS_EXTS = (".js", ".jsx", ".mjs")
JS_SYMBOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)")),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)")),
    ("const", re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=")),
)
JS_IMPORT = re.compile(r"""^\s*import\s+(?:[^'"]+\s+from\s+)?['"](\.[^'"]+)['"]""")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def py_module_name(rel: str) -> str | None:
    """Map a tracked .py path to the dotted name it is imported as."""
    parts = rel[:-3].split("/")
    if parts[:2] in (["src", "captioning"], ["backend", "app"]):
        parts = parts[1:]
    elif parts[0] not in ("scripts", "tests"):
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def py_symbols(rel: str, tree: ast.Module) -> list[tuple[str, str, str, int]]:
    out: list[tuple[str, str, str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append((node.name, "class", rel, node.lineno))
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    out.append((f"{node.name}.{sub.name}", "method", rel, sub.lineno))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.append((node.name, "function", rel, node.lineno))
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.append((t.id, "constant", rel, node.lineno))
    return out


def py_imports(tree: ast.Module, module: str, is_pkg: bool) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> resolve against this module
                base = module.split(".") if is_pkg else module.split(".")[:-1]
                base = base[: len(base) - (node.level - 1)]
                name = ".".join([*base, node.module] if node.module else base)
            else:
                name = node.module or ""
            found.add(name)
    return {n for n in found if n.split(".")[0] in INTERNAL_PY_ROOTS}


def resolve_js(rel: str, spec: str, tracked: set[str]) -> str | None:
    base = (Path(rel).parent / spec).as_posix()
    parts: list[str] = []
    for p in base.split("/"):
        if p == "..":
            parts.pop()
        elif p != ".":
            parts.append(p)
    base = "/".join(parts)
    for cand in (base, *(base + e for e in JS_EXTS), *(f"{base}/index{e}" for e in JS_EXTS)):
        if cand in tracked:
            return cand
    return None


def main() -> int:
    # Resolve the repo from this file's location, not the caller's cwd, so the
    # SessionStart hook (and any other caller) works from any directory.
    root = Path(git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip())
    out = root / ".claude" / "context"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in git(root, "ls-files").splitlines() if f)
    tracked = set(files)
    (out / "files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")

    weights = Counter("/".join(f.split("/")[:2]) if "/" in f else "(root)" for f in files)
    (out / "dir-weights.txt").write_text(
        "".join(f"{n:5d} {d}\n" for d, n in weights.most_common()), encoding="utf-8"
    )

    symbols: list[tuple[str, str, str, int]] = []
    graph: dict[str, dict[str, object]] = {}
    for rel in files:
        path = root / rel
        if rel.endswith(".py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError) as exc:
                print(f"WARN: skipped {rel}: {exc}", file=sys.stderr)
                continue
            symbols.extend(py_symbols(rel, tree))
            mod = py_module_name(rel)
            if mod:
                imports = py_imports(tree, mod, rel.endswith("__init__.py"))
                graph[mod] = {"file": rel, "imports": sorted(imports - {mod})}
        elif rel.startswith("frontend/src/") and rel.endswith(JS_EXTS):
            lines = path.read_text(encoding="utf-8").splitlines()
            imports: set[str] = set()
            for i, line in enumerate(lines, 1):
                for kind, pat in JS_SYMBOL_PATTERNS:
                    if m := pat.match(line):
                        symbols.append((m.group(1), kind, rel, i))
                        break
                if (m := JS_IMPORT.match(line)) and (
                    target := resolve_js(rel, m.group(1), tracked)
                ):
                    imports.add(target)
            graph[rel] = {"file": rel, "imports": sorted(imports)}

    # Reverse edges. An import of a package-level name (``captioning.inference``)
    # counts against that package's __init__ module.
    imported_by: dict[str, set[str]] = defaultdict(set)
    for mod, info in graph.items():
        for dep in info["imports"]:  # type: ignore[attr-defined]
            target = dep
            while target and target not in graph:
                target = target.rpartition(".")[0]
            if target:
                imported_by[target].add(mod)
    for mod, info in graph.items():
        info["imported_by"] = sorted(imported_by.get(mod, set()))

    symbols.sort(key=lambda s: (s[2], s[3]))
    (out / "symbols.tsv").write_text(
        "name\tkind\tfile\tline\n" + "".join(f"{n}\t{k}\t{f}\t{ln}\n" for n, k, f, ln in symbols),
        encoding="utf-8",
    )
    (out / "deps.json").write_text(json.dumps(graph, indent=1, sort_keys=True), encoding="utf-8")

    churn = Counter(
        f
        for f in git(root, "log", "-n", "100", "--name-only", "--pretty=format:").splitlines()
        if f and f in tracked
    )
    (out / "hotspots.txt").write_text(
        "".join(f"{n:4d} {f}\n" for f, n in churn.most_common(40)), encoding="utf-8"
    )

    print(f"index rebuilt: {len(files)} files, {len(symbols)} symbols, {len(graph)} modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
