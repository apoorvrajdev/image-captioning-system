"""Static checks that Make targets match the repository (TASK-005).

The Makefile and the CLI scripts are read as text / AST only: nothing runs
Make, Docker, or an evaluation, and TensorFlow is never imported. ``make -n``
exits 0 even when the printed command is broken, so these checks are the gate.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
COMPOSE_FILES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")


def _parse_makefile() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(variables, recipes)`` with backslash continuations joined."""
    text = MAKEFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    variables: dict[str, str] = {}
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("\t"):
            if current is not None:
                recipes[current].append(line.strip())
            continue
        current = None
        if assignment := re.match(r"^([A-Za-z_]\w*)\s*[?:]?=\s*(.*)$", line):
            variables[assignment.group(1)] = assignment.group(2).strip()
        elif target := re.match(r"^([A-Za-z0-9_-]+):(?!=)", line):
            current = target.group(1)
            recipes.setdefault(current, [])
    return variables, {name: "\n".join(lines) for name, lines in recipes.items()}


VARIABLES, RECIPES = _parse_makefile()


def _expand(text: str) -> str:
    return re.sub(r"\$\((\w+)\)", lambda m: VARIABLES.get(m.group(1), m.group(0)), text)


def _docker_builds() -> list[tuple[str, list[str]]]:
    """``(target, argv)`` for every ``docker build`` line in a recipe."""
    builds = []
    for target, recipe in RECIPES.items():
        for line in recipe.splitlines():
            argv = _expand(line).split()
            if argv[:2] == ["docker", "build"]:
                builds.append((target, argv))
    return builds


def _dockerfile_for(argv: list[str]) -> Path:
    for flag in ("-f", "--file"):
        if flag in argv:
            return REPO_ROOT / argv[argv.index(flag) + 1]
    return REPO_ROOT / argv[-1] / "Dockerfile"


def _build_arg_names(argv: list[str]) -> list[str]:
    names = []
    for i, token in enumerate(argv):
        if token == "--build-arg":
            names.append(argv[i + 1].split("=", 1)[0])
        elif token.startswith("--build-arg="):
            names.append(token.split("=", 2)[1])
    return names


def _required_options(script: Path) -> set[str]:
    """Long flags of ``click.option(...)`` declarations with ``required=True``."""
    flags: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "option"
        ):
            continue
        if any(
            kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        ):
            flags.update(
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            )
    return flags


SCRIPT_TARGETS = sorted(
    (target, match.group(1))
    for target, recipe in RECIPES.items()
    for match in re.finditer(r"-m scripts\.(\w+)", recipe)
)


def test_docker_build_targets_use_existing_dockerfile() -> None:
    builds = _docker_builds()
    assert builds, "expected at least one docker build target"
    for target, argv in builds:
        dockerfile = _dockerfile_for(argv)
        assert dockerfile.is_file(), f"make {target}: {dockerfile} does not exist"


def test_docker_build_args_are_declared_in_dockerfile() -> None:
    for target, argv in _docker_builds():
        dockerfile = _dockerfile_for(argv)
        text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
        declared = set(re.findall(r"^\s*ARG\s+(\w+)", text, flags=re.MULTILINE))
        for name in _build_arg_names(argv):
            assert name in declared, f"make {target}: --build-arg {name} has no ARG {name}"


def test_compose_targets_have_a_compose_file() -> None:
    has_compose_file = any((REPO_ROOT / name).is_file() for name in COMPOSE_FILES)
    compose_targets = sorted(
        target for target, recipe in RECIPES.items() if re.search(r"\bdocker[ -]compose\b", recipe)
    )
    assert (
        has_compose_file or not compose_targets
    ), f"compose targets {compose_targets} but no compose file in the repo root"


@pytest.mark.parametrize(("target", "script"), SCRIPT_TARGETS, ids=[t for t, _ in SCRIPT_TARGETS])
def test_script_targets_pass_required_options(target: str, script: str) -> None:
    passed = set(RECIPES[target].split())
    missing = sorted(_required_options(REPO_ROOT / "scripts" / f"{script}.py") - passed)
    assert not missing, f"make {target}: scripts.{script} requires {missing}"
