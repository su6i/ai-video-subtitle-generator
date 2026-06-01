# CLAUDE.md — subtitle

## Project
<!-- TODO: describe what this project does in 1-2 sentences -->

## Tech Stack
Python

## Relevant Skills
Read these from `.agent/skills/` before implementing domain-specific logic:
python-core-standards, python-containerization

## Key Constraints
<!-- TODO: any project-specific rules -->

## Rules & Workflows
- Rules     : `.agent/rules/` — read 000-core.md, global.md, 040-git.md before every task
- Workflows : `.agent/workflows/` — pick the relevant one per task type
- Skills    : `.agent/skills/` — 75 domain knowledge modules

## Global Rules
Git protocol, cost control, and code quality are in ~/.claude/CLAUDE.md (auto-loaded).

## قوانین Agent

### قوانین مشترک (از agent-constitution)
تمام فایل‌های `.agent/constitution/rules/` را بخوان و رعایت کن.
آپدیت: `git submodule update --remote .agent/constitution`

### قوانین اختصاصی این پروژه
تمام فایل‌های `.agent/local-rules/` را بخوان.
در صورت تناقض، **قوانین اختصاصی (local-rules) اولویت دارند.**

