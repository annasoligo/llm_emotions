# Paper Writing Skill

Invoke with `/paper` when starting or continuing paper-writing work.

## When to use

- Starting a new paper or workshop submission
- Generating figures for a paper
- Writing/editing LaTeX
- Assembling appendices
- Pre-submission checks

## Workflow: Two-Repo Pattern

Keep experiments and paper separate:

```
experiment-repo/          # e.g. research-tools
  experiments/            # Code that runs experiments
  results/                # Raw results (gitignored)
  CLAUDE.md               # Experiment-focused instructions

paper-repo/               # e.g. gemma-iclr
  paper.tex               # Main paper
  figures/                # Final figures + .meta.json sidecars (committed)
  scripts/                # Figure-generating scripts (committed)
  data/                   # Processed data for figures (small, committed)
  CLAUDE.md               # Paper-focused instructions (see template below)
```

## Paper-Repo CLAUDE.md Template

Every paper repo should have a CLAUDE.md containing:

```markdown
# Paper: {Title}

## Key Claims
1. {Claim 1 - one sentence}
2. {Claim 2 - one sentence}

## Venue
- **Conference**: {name, year}
- **Format**: {workshop/main}, {page limit} pages
- **Style file**: {filename}
- **Deadline**: {date}
- **Column width**: {single/double}, figure width = {X}in

## Data Sources
- {Description}: `{path in experiment-repo}`

## Figure → Data Mapping
| Figure | Script | Data | Claim |
|--------|--------|------|-------|
| fig1   | scripts/plot_X.py | results/X.jsonl | 1 |

## Rules
- Don't rewrite prose sections - edit specific sentences or format data
- Every figure must have a .meta.json sidecar with provenance
- Use {citation style} for references
- Anonymous submission: no author names anywhere
```

## Figure Pipeline

### Generating figures

1. Write a standalone script in `paper-repo/scripts/` that reads data and produces the figure
2. Script must save a `.meta.json` sidecar alongside every figure
3. Commit the script AND the figure

### Figure metadata (MANDATORY)

Every saved plot MUST have a `{figure_name}.meta.json` sidecar:

```python
from steering_utils.provenance import get_provenance
import json
from pathlib import Path

meta = get_provenance(
    script=__file__,
    extra={
        "result_files": [str(p) for p in result_paths],
        "parameters": {"judge_model": "claude-sonnet-4-5-20250929"},
    },
)
fig.savefig(output_path)
Path(str(output_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
```

The `.meta.json` records: git commit, generating script, input data files with their commits, and any parameters. This makes every figure fully traceable.

### Figure style defaults

Set these BEFORE making any plots, based on the venue:

```python
# Workshop paper (single column, ~5.5in wide)
SINGLE_COL = {"figsize": (5.5, 3.5), "dpi": 300}
# Main conference (double column, ~3.25in per column)
DOUBLE_COL = {"figsize": (3.25, 2.5), "dpi": 300}
FULL_WIDTH = {"figsize": (7, 4), "dpi": 300}
```

## LaTeX Collaboration

### Good tasks for Claude
- Assembling appendices (tables, prompts, examples)
- Formatting data as LaTeX tables
- Adding cross-references (`\Cref`)
- Checking consistency (figures referenced, numbers match data)
- Citation formatting and .bib management
- Related work drafting

### Bad tasks for Claude
- Writing the narrative arc (human judgement required)
- Writing concise opinionated claims (Claude hedges)
- Deciding what to cut (Claude wants to include everything)

**Pattern**: Human writes narrative in plain text (blog post / bullet outline) -> Claude helps format into LaTeX.

## Overleaf Sync

When co-authors use Overleaf:

1. Set up Overleaf git remote
2. Human edits prose on Overleaf
3. Claude generates figures/appendices locally
4. Merge via git (Overleaf commits show as "Updates from Overleaf")
5. Push Claude's changes back to Overleaf remote

## Session Management

- **One session per task**: "make Figure 3", "format the appendix", "check all citations"
- **Start with context**: "Working on Section 3 of {paper}. Claim: {X}. Need: {Y}."
- **Commit after every session**
- **Don't use mega-sessions** (>50 messages) for paper work

## Building Appendices Progressively

Don't leave appendix assembly for the submission sprint. As each experiment runs:
1. Record hyperparameters in an appendix table
2. Save the judge/eval prompt to the appendix
3. Add example outputs
4. Add model identifiers and versions

## Pre-Submission Checklist

- [ ] All figures referenced in text
- [ ] All numbers in text match actual data
- [ ] All models listed consistently (full names + short names)
- [ ] Judge/eval prompts included in appendix
- [ ] All citations compile (no `??` references)
- [ ] Anonymous submission requirements met (no author names, no identifying URLs)
- [ ] Page limit respected (main body only, appendix unlimited for most venues)
- [ ] Figure resolution sufficient (300 DPI for print)
- [ ] All `.meta.json` sidecars present for figures
- [ ] Code repo cleaned and ready for camera-ready release

## Detailed Notes

See: `/workspace-vast/annas/Ant_Cluster_Notes/paper_writing/2026-02-07_best_practices_writing_papers_with_claude_code.md`
