---
name: ship-task
description: Take one task from docs/TASKS.md (or a bug/feature request) from requirement to a verified, review-ready change with a proposed commit sequence. Use when the user says "implement TASK-xxx", "continue Phase N", "fix <issue>", or "do the next task".
---

# Ship a task

Input: a task id from `docs/TASKS.md`, or a described change. If the change is phase-sized
("do Phase 3"), don't implement it. Decompose it into tasks in `docs/TASKS.md` first and confirm.

1. **Load state.** Read `docs/MEMORY.md` and the task entry in `docs/TASKS.md`. Quote its goal and
   acceptance criteria. Ask about genuine ambiguity **now**, in one concise question, and only when it's critical.
2. **Locate.** Follow the retrieval protocol in `CLAUDE.md` (repo-map → symbols.tsv → deps.json).
   Read only those files plus their direct dependencies.
3. **Load criteria.** Open every matching `.claude/skills/<area>/SKILL.md`
   (`ml-core`, `inference-api`, `frontend`, `evaluation`, `deployment`). If the task adds behaviour
   the skill doesn't cover, add the behaviour/edge case to that skill.
4. **Plan.** Numbered steps and the files each touches. Multi-lane → ownership map and the
   orchestration protocol in `CLAUDE.md`. Flag any invariant the plan comes near (parity,
   contract, artefact immutability, eval methodology).
5. **Implement** the smallest correct change, with tests that fail without it. Bug reports go
   through the Debugging protocol in `CLAUDE.md` first: reproduce, find the root cause, then fix.
6. **Review.** Re-read the whole diff (`git diff`) against the CLAUDE.md Invariants: parity defaults,
   API contract ↔ `api.js`, artefact immutability, no secrets, no unrelated churn. Run `/code-review`.
   Upload, API, CI or Docker changes also get `/security-review`. Fix findings before verifying.
7. **Verify.** Run the change-type flow below and the DoD of every touched skill on the final tree.
   Quote the results. Red → fix the root cause and re-run. Never weaken assertions.
8. **Record.** Update `docs/MEMORY.md` (state, recent changes), tick the task in `docs/TASKS.md`,
   add a `docs/DECISIONS.md` entry for any permanent decision. Rebuild the index if modules moved:
   `.venv/Scripts/python.exe .claude/context/build_index.py`.
9. **Report.** Files changed (grouped by layer), what and why, the verification table,
   open risks, then the commit sequence (one `git add <files>` + `git commit -m "<conventional>"`
   pair per logical change, ordered schemas → implementation → tests → docs). **Don't run it.**

## Change-type flows (step 7)

| Change touches | Verification flow, in order |
|---|---|
| `frontend/**` | lint → build → browser loop (`frontend` skill; report "not browser-verified" if no browser tooling) → contract check against `backend/app/schemas/caption.py` |
| `backend/app/**`, `src/captioning/inference/**` | backend route tests (no TF) → full pytest → parity audit if inference/preprocessing changed → ruff + mypy → `docker build .` if `Dockerfile`/deps changed and Docker exists |
| `src/captioning/{config,preprocessing,models,data,training}`, `configs/**` | experiment defined as an opt-in flag + YAML → unit tests → parity audit 4/4 + notebook freeze → full pytest → ruff + mypy. Training and full evals are owner-run: prepare the command, never fabricate numbers |
| `src/captioning/evaluation/**`, eval scripts | hand-checkable metric tests → full pytest → new `results/<run_id>/` only via the script, with the eval setup stated (slice, refs, tokenisation) |
| `.github/**`, `Dockerfile`, deps, pre-commit | YAML parses → `SKIP=mypy pre-commit run --all-files` → full pytest → no gate removed or weakened (`deployment` skill) |
| docs only | every claim checked against its source file; no AI/tool attribution in public docs |

Hard stops: no commits, pushes, branches, tags, deploys, HF Hub uploads, retraining, or
notebook edits. Those are the user's. No AI attribution anywhere in commits, code, or docs.
