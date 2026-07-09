# RQuant Rename and README Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project-facing brand to RQuant and replace the README with a concise guide for the current top-level research layout.

**Architecture:** Keep repository paths, Python packages, and CLI entrypoints stable. Update user-facing names and documentation only, plus any tests that assert the old display name.

**Tech Stack:** Markdown, Python unittest, existing CLI modules.

---

### Task 1: Rename User-Facing Project Text

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/architecture.md`
- Modify: `scripts/quant_cli.py`
- Modify: `pipeline/cli.py`
- Modify: `run_all.py`
- Modify: `reports/research_report.py`
- Modify: `tests/test_research_report.py`

- [ ] Replace previous public names with `RQuant`.
- [ ] Preserve existing package names and commands such as `python -m pipeline.cli`.
- [ ] Update the research-report test assertion to `RQuant Research Report`.

### Task 2: Rewrite README

**Files:**
- Modify: `README.md`

- [ ] Replace the old long README with a focused RQuant README.
- [ ] Cover project positioning, top-level package layout, research paths, setup, commands, outputs, validation, and safety boundaries.
- [ ] Keep commands copyable and aligned with the migrated layout.

### Task 3: Verify and Commit

**Files:**
- Modify only files above.

- [ ] Run separate literal-name searches across active files and expect no previous public-name matches.
- [ ] Run `/opt/miniconda3/envs/stocktrade/bin/python -m unittest tests.test_research_report tests.test_cli`.
- [ ] Run `/opt/miniconda3/envs/stocktrade/bin/python -m unittest discover -s tests -p 'test_*.py'`.
- [ ] Run `git diff --check`.
- [ ] Commit with `docs: rename project to RQuant`.
