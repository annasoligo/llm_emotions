---
name: audit
description: Audit and update Claude Code configuration files (CLAUDE.md, skills, MEMORY.md). Searches recent conversations for common failures, checks for duplication and inconsistency, and suggests improvements. Run periodically to keep config current.
---

# Config Audit Skill

## Overview

This skill audits all Claude Code configuration files for this project, finds patterns in recent conversation history, and updates the config to prevent recurring mistakes. Run it periodically (e.g. every week or two) to keep things current.

## Instructions

When the user invokes `/audit`:

### Phase 1: Mine Conversation History for Patterns

Search the conversation history JSONL files for recurring failures and patterns.

**History location**: `/home/annas/.claude/projects/-workspace-vast-annas-git-research-tools/*.jsonl`

Launch an Explore agent (or multiple) to search the **most recent 30 conversation files** (by modification time) for:

1. **Errors and failures** - grep for patterns like:
   - `error`, `Error`, `ERROR`, `failed`, `FAILED`, `traceback`, `Traceback`
   - `OOM`, `out of memory`, `CUDA`, `killed`
   - `Permission denied`, `No such file`, `ModuleNotFoundError`
   - `retry`, `retrying`, `fallback`

2. **Repeated corrections** - look for:
   - User messages containing "no", "wrong", "not that", "actually", "I said"
   - Multiple attempts at the same task (same file edited 3+ times)
   - The user asking for the same thing they asked before

3. **Workflow patterns** - look for:
   - New tools, scripts, or approaches that worked well
   - New models or APIs being used
   - Changed conventions (file naming, output formats, etc.)

**Important**: Only scan the last 30 conversations to keep this focused on recent patterns. Use `ls -t` to sort by modification time.

### Phase 2: Read All Config Files

Read the current state of all config files:

1. `/workspace-vast/annas/git/research-tools/CLAUDE.md` (project instructions - single source of truth)
2. `/workspace-vast/annas/git/research-tools/.claude/skills/vllm.md` (vLLM best practices)
3. `/workspace-vast/annas/git/research-tools/.claude/skills/slurm/SKILL.md` (slurm skill)
4. `/workspace-vast/annas/git/research-tools/.claude/skills/audit/SKILL.md` (this file)
5. `/home/annas/.claude/skills/notes/SKILL.md` (notes skill - user level)
6. `/home/annas/.claude/projects/-workspace-vast-annas-git-research-tools/memory/MEMORY.md` (auto-memory)

### Phase 3: Produce Audit Report

Create a structured report covering:

#### A. New Failure Patterns
For each failure pattern found in conversations that is NOT already covered by CLAUDE.md or MEMORY.md:
- What went wrong (with example from conversation)
- How many times it occurred
- Suggested rule or note to add

#### B. Duplication Check
Scan all config files for information that appears in multiple places. Flag:
- Facts stated in both CLAUDE.md and a skill file (skill should reference CLAUDE.md)
- Contradictory values (e.g., different gpu_memory_utilization in different files)
- Templates or code blocks that are copy-pasted across files

#### C. Staleness Check
Flag anything that might be outdated:
- Models no longer being used
- Paths that don't exist on disk
- Rules about fixed issues (e.g., node exclusions for nodes that work now)
- Package versions or API endpoints that may have changed

#### D. Skill Suggestions
Identify content in CLAUDE.md that would be better as a dedicated skill:
- Sections longer than 20 lines that cover a specific workflow
- Content that's only relevant when doing a specific task (not general guidance)
- Repeated patterns in conversations that could be automated

#### E. MEMORY.md Review
Check if MEMORY.md:
- Is under 200 lines (truncation limit)
- Has accurate "See also" links
- Reflects current user preferences (check recent conversations for changes)
- Has up-to-date failure counts

### Phase 4: Present Findings

Present the report to the user as a structured summary. For each finding, state:
- **What**: The issue
- **Where**: Which file(s)
- **Action**: Specific proposed change (add/remove/move/update)

Ask the user which changes to apply.

### Phase 5: Apply Changes

For approved changes:
1. Edit the relevant config files
2. Ensure CLAUDE.md remains the single source of truth
3. Ensure skills reference CLAUDE.md rather than duplicating
4. Keep MEMORY.md concise (under 200 lines)
5. Suggest a commit with the changes

## Critical Rules

- **CLAUDE.md is the single source of truth.** Skills should reference it, not duplicate it.
- **MEMORY.md must stay under 200 lines.** Lines after 200 get truncated in the system prompt.
- **Don't remove rules without asking.** A rule that seems obsolete might still be important.
- **Be specific in reports.** Include file paths, line numbers, and exact quotes.
- **Keep skills focused.** Each skill should do one thing. If a skill is getting bloated, suggest splitting it.
- **Check paths exist.** Before flagging something as stale, verify the path actually doesn't exist.

## Example Output

```
## Audit Report - 2026-02-07

### New Failure Patterns (3 found)
1. **Qwen3 thinking mode not disabled** (seen 4x in last 30 sessions)
   - Scripts using Qwen3 models without `/no_think` suffix
   - Suggestion: Already in CLAUDE.md but could be stronger wording

2. **API rate limit hits** (seen 2x)
   - Jobs running with Semaphore(50) while other jobs also running
   - Suggestion: Already covered in API Concurrency section

### Duplication (1 found)
1. Models table appears in both CLAUDE.md:52-60 and slurm/SKILL.md:12-20
   - Action: Slurm skill should say "see CLAUDE.md" instead

### Staleness (0 found)
- All paths verified, all models still in use

### Skill Suggestions (1 found)
1. Plotting section in CLAUDE.md (15 lines) could become a `/plot` skill
   - Contains specific style rules only relevant when making plots

### MEMORY.md
- Currently 40 lines (well under 200 limit)
- All links valid
```
