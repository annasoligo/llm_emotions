# Archive Directory

This directory contains old experiment scripts and documentation that are no longer actively used but preserved for reference.

## Structure

### `/experiments_v1_v4/`
**Very early experiments** (Dec 29-30, 2025)
- Original impossible task experiments
- Multiple iterations (v1-v4) exploring different prompt approaches
- Includes: single-turn, concurrent, and early multi-turn implementations
- **Status:** Superseded by v5+ experiments

### `/experiments_v11_v12/`
**Recent experiments** (Dec 30-31, 2025)
- v11: Latest experiments before v13/v14
- v12: Cancelled experiment that tested "maximum suppression with existential shutdown"
  - Cancelled due to removal of suppression prefix per user request
  - Only clean baselines (v13, v14) were kept running
- **Status:** v11 superseded by v13/v14; v12 cancelled mid-run

### `/experiment_scripts/`
Archived Python experiment scripts from various versions.

### `/slurm_scripts/`
Archived SLURM batch scripts corresponding to old experiments.

## Active Experiments

Current active experiments (v13, v14) are in the parent directory:
- `run_elicitation_multiturn_v13.py` - Clean baseline with top 6 prompts
- `run_elicitation_multiturn_v14_base.py` - Base pre-trained model test

See `FINAL_EXPERIMENT_STATUS.md` in parent directory for current status.

## Documentation Archive

Historical documentation and analysis files are in `/docs/archive/`:
- Experiment plans (V5-V10)
- Analysis documents
- Launch summaries
- Prompt analyses

## What Was Deleted

The following files were removed during cleanup (Dec 31, 2025):
- `ACTIVE_EXPERIMENTS.md` - Outdated status (showed V12 as running when it was cancelled)
- `RELAUNCHED_EXPERIMENTS.md` - Historical note about feedback changes (info preserved in FINAL_EXPERIMENT_STATUS.md)
