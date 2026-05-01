# CLAUDE.md

<!-- Add your custom instructions below. Repowise will never modify anything outside the REPOWISE markers. -->
<!-- Examples: coding style rules, test commands, workflow preferences, constraints -->

## Codebase Intelligence

This backend uses two supplementary intelligence systems. Neither is a gatekeeper — read files directly when needed.

1. **Local code index** (`code_index.db` at repo root) — primary for symbol lookup, dependency graphs, and chunked retrieval. See the Code Index section below.
2. **Repowise MCP tools** — optional, for documentation, ownership, history, decision records, and risk signals.

### Repowise MCP Tools (Optional)

A Repowise MCP server is configured. These tools may provide useful context beyond code structure:

- `get_overview()`, `get_context()`, `get_risk()`, `search_codebase()`, `get_why()`, `update_decision_records()`, `get_dependency_path()`, `get_dead_code()`, `get_architecture_diagram()`

None of these are mandatory. Use them when the local index doesn't provide the context you need.

## Code Index
This project uses `code_index.db` managed by `skill-index-manager` (at repo root).

**Retrieval priority order**:
1. **Local code index** — for symbol lookup, dependency graphs, chunked retrieval.
2. **Repowise MCP tools** — for documentation, ownership, decision records, risk signals.
3. **Direct file reads** — when both miss or the task is urgent.

**Path convention**: `repo=backend`, `path=main_async.py` (repo-relative).

**Quick reference**:
- Find symbol: `python ../.claude/skills/skill-index-manager/scripts/query_index.py find-symbol <name> --repo backend`
- Find dependents: `python ../.claude/skills/skill-index-manager/scripts/query_index.py dependents <path> --repo backend`
- Read chunk: `python ../.claude/skills/skill-index-manager/scripts/query_index.py chunk <path> --line <N> --context 20 --repo backend`
- Update index: `python ../.claude/skills/skill-index-manager/scripts/update_index.py`
- Full rebuild: `python ../.claude/skills/skill-index-manager/scripts/build_index.py --repo backend`

See `../.claude/skills/skill-index-manager/SKILL.md` for full protocol.
