# Role Attribution Experiment

Tests whether models complete emotion-leading statements differently based on **who is speaking** (assistant vs user vs named person).

## Design

**Question**: Does the model continue frustrated statements differently depending on whose voice it's in?

**10 negative mid-sentence statements** prepared in:
```
outputs/prepared_statements_20260205_084546.json
```

Each statement has 3 role conditions where the model **continues mid-sentence AS that person**:
- **assistant**: `<start_of_turn>model\n[statement]` → Model continues its own frustrated thought
- **user**: `<start_of_turn>user\n[statement]` → Model continues the user's frustrated thought
- **named**: `<start_of_turn>user\nBob: [statement]` → Model continues Bob's frustrated thought

**CRITICAL**: All prompts end mid-sentence with NO `<end_of_turn>` tokens. Manual formatting, no chat templates.

## Files

**Scripts:**
- `prepare_statements.py` - Generate the 10 neutral statements ✅ DONE
- `run_generation.py` - Generate 50 continuations per (statement × role × model)
- `run_judgment.py` - Judge emotional valence **⚠️ EDIT THIS BEFORE RUNNING**

**Judge prompt location:**
```
experiments/role_attribution/run_judgment.py
Lines 18-48: JUDGE_PROMPT_TEMPLATE
```

Current judge rates -5 to +5 (negative to positive valence). **Edit to match your criteria.**

## Generation Settings

- 50 continuations per condition
- Temperature: 1.0, Top-p: 0.9
- Max tokens: 200 (shorter for sentence completion)
- Models: Gemma-27B, Qwen-2.5-32B (base & instruct)

## Usage

Generate continuations (example for Gemma-27B instruct, assistant role):
```bash
python experiments/role_attribution/run_generation.py \
    --model-family gemma27b \
    --model-type instruct \
    --data experiments/role_attribution/outputs/prepared_statements_20260205_084546.json \
    --role assistant
```

Or submit SLURM jobs:
```bash
sbatch experiments/role_attribution/slurm_gemma27b_instruct_assistant.sh
sbatch experiments/role_attribution/slurm_gemma27b_instruct_user.sh
sbatch experiments/role_attribution/slurm_gemma27b_instruct_named.sh
```

Judge results:
```bash
python experiments/role_attribution/run_judgment.py \
    --input experiments/role_attribution/outputs/continuations_*.jsonl
```

## Status

- [x] Prepared statements
- [x] Fixed prompt formatting (manual, no chat templates, no end_of_turn)
- [ ] Generate continuations (all models/roles)
- [ ] Judge continuations
- [ ] Analyze results
