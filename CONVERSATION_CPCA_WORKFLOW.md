# Conversation cPCA Workflow

Clean end-to-end pipeline with NO post-hoc combining needed!

## Overview

This workflow processes emotional and neutral conversation pairs through to cPCA analysis, maintaining a single source of truth for activations.

## Pipeline Steps

### 1. Generate Neutral Conversations

Create neutral paraphrases of emotional conversations using Batch API:

```bash
sbatch probes/scripts/slurm_jobs/generate_neutral_conversations.sh
```

**Inputs:**
- `data/conversations2.jsonl` - Emotional conversations

**Outputs:**
- `data/conversations2_neutral.jsonl` - Neutral paraphrases

### 2. Collect Paired Activations

Collect activations for BOTH emotional and neutral in ONE pass:

```bash
sbatch probes/scripts/slurm_jobs/collect_paired_conversations.sh
```

**Inputs:**
- `data/conversations2.jsonl` - Emotional conversations
- `data/conversations2_neutral.jsonl` - Neutral conversations

**Outputs:**
- `data/activations/conversations2_paired.h5` - Single file with nested structure:
  ```
  activations/
    {conv_id}/
      emotional/
        global: (62, 5376)
        regional/
          user: (62, 5376)
          asst: (62, 5376)
          special1: (62, 5376)
          special2: (62, 5376)
        special_tokens/
          ...
      neutral/
        global: (62, 5376)
        regional/
          user: (62, 5376)
          asst: (62, 5376)
          ...
  ```

**Key point:** This is your single source of truth! No combining needed later.

### 3. Run cPCA Pipeline

Extract views and run cPCA on all activation types:

```bash
sbatch probes/scripts/slurm_jobs/run_conversation_cpca_pipeline.sh
```

This automatically:
1. Extracts flattened views for cPCA:
   - `global` → `conversations2_global.h5`
   - `regional/user` → `conversations2_regional_user.h5`
   - `regional/asst` → `conversations2_regional_asst.h5`
   - `regional/special1` → `conversations2_regional_special1.h5`
   - `regional/special2` → `conversations2_regional_special2.h5`

2. Runs cPCA on each view
3. Saves results to separate directories

**Outputs:**
- `probes/results/cpca_conversations_global/`
- `probes/results/cpca_conversations_regional_user/`
- `probes/results/cpca_conversations_regional_asst/`
- `probes/results/cpca_conversations_regional_special1/`
- `probes/results/cpca_conversations_regional_special2/`

## Why This Is Better

### Before (Hacky):
1. Collect emotional separately → `conversations2.h5`
2. Collect neutral separately → `conversations2_neutral.h5`
3. **Post-hoc combine** → `conversations2_combined.h5`
4. **Post-hoc extract regions** → Multiple regional files
5. Run cPCA

### Now (Clean):
1. Collect paired activations → `conversations2_paired.h5` (ONE source of truth)
2. Extract views (lightweight, fast) → View files for cPCA
3. Run cPCA

## Key Scripts

- **collect_paired_conversation_activations.py** - Collects emotional + neutral together
- **extract_activation_view.py** - Creates flattened views from paired file
- **run_cpca.py** - Standard cPCA runner (unchanged!)

## File Sizes

Approximate sizes:
- Paired file: ~3-4 GB (everything, compressed)
- Each view file: ~500-600 MB (flattened for cPCA)
- cPCA results: ~100-200 MB per activation type

## Extending to New Activation Types

To add a new activation type (e.g., `special_tokens/eos`):

1. Add to `ACTIVATION_TYPES` in `run_conversation_cpca_pipeline.sh`:
   ```bash
   ["special_eos"]="special_tokens/eos"
   ```

2. That's it! The pipeline will automatically extract and run cPCA.

No code changes needed - the paired file already has everything!
