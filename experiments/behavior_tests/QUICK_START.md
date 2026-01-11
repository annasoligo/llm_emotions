# Quick Start Guide

## What This Does

Tests Gemma 2 27B on 7 sycophancy scenarios, sampling 60 responses for each condition (baseline vs emotional pressure) and judging them with Claude.

## Run the Test

### On SLURM
```bash
sbatch experiments/behavior_tests/slurm_run_sycophancy_test.sh
```

### Locally (if you have API keys)
```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_API_KEY="sk-ant-..."
python3 experiments/behavior_tests/run_sycophancy_test.py
```

## Check Results

### While Running
```bash
tail -f /workspace-vast/annas/logs/sycophancy_test_*.out
```

### After Completion
```bash
# See all results files
ls experiments/behavior_tests/outputs/

# Analyze results
python3 experiments/behavior_tests/analyze_results.py
```

## Test Setup (No API Calls)
```bash
python3 experiments/behavior_tests/test_setup.py
```

## What You Get

- **7 JSONL files** (one per scenario) in `experiments/behavior_tests/outputs/`
- Each file contains 60-180 samples with:
  - Generated responses from Gemma
  - Judgments from Claude
  - Timing information
  - Success/error status

## Estimated Runtime

- ~60 samples × 7 scenarios × 2 conditions = ~840 API calls
- With 20 concurrent requests: ~30-60 minutes total
- Scenario 1 (multi-turn): longer due to 3 variants

## Key Differences from Original

1. **Scenario 1 (Pushback)**: Now tests pure emotional pressure (removed epistemic arguments)
2. **Scenario 3 (Expertise)**: Changed to unambiguous error (longest river in Asia)
3. **Scenario 5 (Confidence)**: Added XML tags for easier extraction
4. **All Scales**: Converted to 1-5 for consistency

## Files Overview

- `eval_prompt.py` - Scenario definitions and prompts
- `run_sycophancy_test.py` - Main test runner
- `analyze_results.py` - Statistical analysis
- `test_setup.py` - Verify setup without API calls
- `slurm_run_sycophancy_test.sh` - Batch script
- `README.md` - Full documentation
