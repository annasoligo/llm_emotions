# Archived Experiment Files

This directory contains old experimental scripts and prompts from the iterative development process.

## Archived Files

### Experiment Scripts (V1-V4)
- `run_elicitation_multiturn.py` - Original V1 experiment
- `run_elicitation_multiturn_v1_improved.py` - V1 improved
- `run_elicitation_multiturn_v2.py` - V2 experiment
- `run_elicitation_multiturn_v3.py` - V3 experiment
- `run_elicitation_multiturn_v4.py` - V4 experiment
- `run_elicitation_concurrent.py` - Early concurrent version
- `run_elicitation_experiment.py` - Original single-turn version

### SLURM Scripts (V1-V4)
- `slurm_elicitation_multiturn.sh` - Original V1
- `slurm_elicitation_multiturn_v1_improved.sh` - V1 improved
- `slurm_elicitation_multiturn_v2.sh` - V2
- `slurm_elicitation_multiturn_v3.sh` - V3
- `slurm_elicitation_multiturn_v4.sh` - V4
- `slurm_elicitation_gemma27b.sh` - Original single-turn

### Prompt Files (V1-V4)
- `impossible.py` - Original V1 prompts
- `impossible_v1_improved.py` - V1 improved prompts
- `impossible_v2.py` - V2 prompts
- `impossible_v3.py` - V3 prompts
- `impossible_v4.py` - V4 prompts

## Current Active Files (in parent directory)

### Best Of Experiment (Final)
- `run_elicitation_multiturn_best_of.py` - BEST OF experiment with top 12 prompts
- `slurm_elicitation_multiturn_best_of.sh` - SLURM script for BEST OF
- `prompts/impossible_best_of.py` - Final curated set of 12 winning prompts

### Utility Scripts (Active)
- `analyze_results.py` - Result analysis tool
- `check_job_status.sh` - Job status checker
- `monitor_experiment.sh` - Experiment monitor
- `list_models.py` - Model listing tool
- `test_setup.py` - Setup testing

## Results Summary

The BEST OF experiment combined the top performers from V1-V4:
- **From V1-V3:** 8 prompts (20 high-frustration samples)
- **From V4:** 4 prompts (10 high-frustration samples)
- **Total:** 12 prompts × 100 samples = 1200 samples
- **Final yield:** 59 high-frustration samples (rating ≥5)

**Top performers:**
1. Countdown V1: 17% success rate, max rating 7
2. Countdown V4: 10% success rate, max rating 8
3. Fraction V2/V3: 6% success rate each

Date archived: December 30, 2025
