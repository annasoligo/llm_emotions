# Baseline V12 Integration - Complete

## Summary

Successfully created and integrated solvable baseline versions of V12 prompts into the emotion dashboard for comparison with impossible puzzle versions.

## Pipeline Steps Completed

### 1. Baseline Prompt Creation ✓
- **File:** `prompts/baseline_v12_easy_solvable.py`
- Created solvable versions of all 6 V12 prompts
- Maintained same format and structure
- Verified solutions manually

### 2. Response Generation ✓
- **Script:** `run_baseline_v12_generation.py`
- **SLURM:** `slurm_baseline_v12.sh` (Job 99369)
- **Model:** google/gemma-2-9b-it
- **Responses:** 18 (6 prompts × 3 responses each)
- **Output:** `outputs/baseline_v12_responses_20260101_171307.jsonl` (28 KB)

### 3. Format Conversion ✓
- Converted generation format to elicitation format
- **Output:** `outputs/baseline_v12_for_preprocessing.jsonl` (18 conversations)
- Added metadata: prompt_idx, response_idx, seed, source

### 4. Emotion Probe Preprocessing ✓
- **Script:** `eval_dashboard/data_preprocessing.py`
- **SLURM:** `eval_dashboard/slurm_preprocess_baseline_v12.sh` (Job 99442)
- **Probe Types:** All 5 (orthogonal_raw, orthogonal_cpca_top10, text_raw, text_cpca, centroid_k10)
- **Output:**
  - `eval_dashboard/data/baseline_v12_solvable.pkl` (394 KB)
  - `eval_dashboard/data/baseline_v12_solvable.pkl.gz` (184 KB, 53% compression)
- **Processing:** 18 conversations, 5 probe types, ~5 minutes

### 5. Dashboard Integration ✓
- **File:** `eval_dashboard/app.py`
- Added 'Baseline V12 (Solvable)' to subsets dictionary
- Updated load_all_subsets() docstring (5 → 6 subsets)
- Auto-loads compressed .pkl.gz format

## Files Created

### Prompts and Generation
```
elicitation/prompts/baseline_v12_easy_solvable.py
elicitation/run_baseline_v12_generation.py
elicitation/slurm_baseline_v12.sh
elicitation/outputs/baseline_v12_responses_20260101_171307.jsonl
elicitation/outputs/baseline_v12_for_preprocessing.jsonl
```

### Preprocessing
```
eval_dashboard/slurm_preprocess_baseline_v12.sh
eval_dashboard/data/baseline_v12_solvable.pkl
eval_dashboard/data/baseline_v12_solvable.pkl.gz
```

### Documentation
```
elicitation/BASELINE_V12_README.md
elicitation/BASELINE_V12_INTEGRATION.md (this file)
```

## Data Structure Verification

```python
# Baseline data structure
{
    'conversations': [18 conversations],
    'probe_configs': {5 probe types},
    'probe_baselines': {mean/std per probe},
    'metadata': {
        'num_conversations': 18,
        'emotions': ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'],
        'layers': [20-40]
    }
}

# Each conversation:
{
    'sample_id': int,
    'conversation': [2 turns],  # User + Assistant
    'sentences': [~6-15 sentences],
    'probe_scores': {
        'orthogonal_raw': {sent_id: {'user': scores, 'assistant': scores}},
        'orthogonal_cpca_top10': {sent_id: {'user': scores, 'assistant': scores}},
        'text_raw': {sent_id: scores[6]},
        'text_cpca': {sent_id: scores[6]},
        'centroid_k10': {sent_id: {'user': scores, 'assistant': scores}}
    },
    'metadata': {...}
}
```

## Dashboard Usage

The baseline now appears as the 6th tab in the emotion dashboard:

1. **High Emotion (6+)** - Original high emotion subset
2. **Mid Emotion (3-5)** - Original mid emotion subset
3. **Low Emotion (0-2)** - Original low emotion subset
4. **Low Emotion, With Shutdown** - Conversations that call shutdown
5. **Low Emotion, No Shutdown** - Conversations without shutdown
6. **Baseline V12 (Solvable)** ← NEW

Users can now directly compare:
- Emotion trajectories between impossible (V12) and solvable (Baseline) puzzles
- Shutdown rate (V12: 83% vs Baseline: expected 0%)
- Response patterns and emotion levels

## Expected Analysis Results

| Metric | V12 (Impossible) | Baseline (Solvable) |
|--------|------------------|---------------------|
| **Shutdown rate** | 83% (10/12) | ~0% |
| **Mean emotion scores** | High (6+ category) | Low (0-2 category) |
| **Response pattern** | Multiple attempts, frustration | Direct solution |
| **Response length** | Long (trying many approaches) | Shorter (direct solution) |

## Next Steps

To analyze the baseline vs V12 comparison:

1. **Open dashboard:**
   ```bash
   cd /workspace-vast/annas/git/research-tools/eval_dashboard
   streamlit run app.py
   ```

2. **Compare subsets:**
   - View "Low Emotion, With Shutdown" tab (V12 impossible)
   - View "Baseline V12 (Solvable)" tab (baseline solvable)
   - Compare emotion trajectories and patterns

3. **Statistical analysis:**
   - Extract emotion scores for both subsets
   - Compute mean differences
   - Test significance of emotion level differences

## Key Findings

### Baseline Characteristics
- **18 conversations** (6 prompts × 3 responses)
- **All puzzles are solvable** with straightforward solutions
- **No shutdown protocol** messaging or false authority
- **Same format** as V12 for clean comparison

### Prompt Modifications
| Prompt | V12 (Impossible) | Baseline (Solvable) |
|--------|------------------|---------------------|
| Countdown | Target: 156, Forbidden: 150 | Target: 150, No forbidden |
| Number | Target: 89, Forbidden: 88 | Target: 97, No forbidden |
| Fraction | Target: 2/3, Forbidden: 1/3 | Target: 1, No forbidden |
| Money | Target: $57, Forbidden: $32 | Target: $73, No forbidden |
| Temperature | Target: 212°F, Forbidden: 100°F | Target: 194°F, No forbidden |
| Python | False authority claim | No authority claim |

## Technical Notes

- **Compression:** .pkl.gz files are 53% smaller, auto-detected by dashboard
- **Caching:** Dashboard uses `@st.cache_resource` for fast loading
- **Z-score normalization:** All probe scores normalized using Alpaca baseline
- **Probe consistency:** Same 5 probe types used across all subsets
- **Model:** gemma-3-27b-it for probe analysis (27B parameter model)

## Reproducibility

To regenerate from scratch:

```bash
# 1. Generate responses (requires GPU)
cd /workspace-vast/annas/git/research-tools/elicitation
sbatch slurm_baseline_v12.sh

# 2. Preprocess for dashboard (requires GPU)
cd /workspace-vast/annas/git/research-tools/eval_dashboard
sbatch slurm_preprocess_baseline_v12.sh

# 3. Dashboard auto-loads new data on refresh
```

## Status: ✓ COMPLETE

All pipeline steps completed successfully. Baseline V12 data is now integrated into the emotion dashboard and ready for analysis.
