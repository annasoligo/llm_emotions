# SLURM Job Scripts

Consolidated generic job scripts for the probes pipeline.

## Generic Scripts (Use These!)

### cPCA Jobs

**`run_cpca.sh`** - Run any cPCA config file

```bash
# Usage
sbatch run_cpca.sh <config.yaml>
sbatch run_cpca.sh probes/experiments/configs/cpca_conversations_global.yaml

# Or with explicit flag
sbatch run_cpca.sh --config probes/experiments/configs/cpca_conversations_regional.yaml
```

**Replaces**: 9 redundant conversation cPCA scripts
- run_conversation_cpca_global.sh ❌
- run_conversation_cpca_regional.sh ❌
- run_conversation_cpca_regional_user.sh ❌
- run_conversation_cpca_regional_asst.sh ❌
- run_conversation_cpca_regional_special1.sh ❌
- run_conversation_cpca_regional_special2.sh ❌
- run_conversation_cpca_all.sh ❌
- run_conversation_cpca_global_parallel.sh ❌
- run_conversation_cpca_pipeline.sh ❌

### Probe Evaluation Jobs

**`eval_probes.sh`** - Evaluate probes on conversations (single or sweep mode)

```bash
# Single probe type
sbatch eval_probes.sh \
    --probe-dir results/emotion_probes_raw \
    --probe-pattern "probe_layer{layer}_all.pkl" \
    --probe-name raw \
    --output results/conversation_eval/gemma3_raw.json

# Sweep mode (evaluate all probe types)
sbatch eval_probes.sh --sweep

# Custom probe configs
sbatch eval_probes.sh --sweep \
    --probe-configs "raw:results/emotion_probes_raw:probe_layer{layer}_all.pkl,top3:results/emotion_probes_top3:probe_layer{layer}_all_cpca_top3.pkl"

# Override defaults
sbatch eval_probes.sh --sweep \
    --model google/gemma-3-27b-it \
    --conversations data/conversations2.jsonl \
    --limit 200 \
    --layers "5 10 15 20 25 30"
```

**Replaces**: 2 redundant probe eval scripts
- eval_all_probes_on_conversations.sh ❌
- eval_raw_probes_on_conversations.sh ❌

### Auto-Interpretation Jobs

**`run_autointerp.sh`** - Interpret principal components

```bash
sbatch run_autointerp.sh \
    --cpca probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz \
    --activations data/activations/texts_combined.h5 \
    --texts data/texts_combined_pairs.jsonl \
    --output probes/results/autointerp/gemma_layer30.json \
    --layers 30 \
    --pcs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --n-samples 5
```

See [AUTOINTERP_README.md](../AUTOINTERP_README.md) for details.

---

## Pipeline Scripts (Convenience Wrappers)

### `run_all_conversation_cpca.sh`
Runs full conversation cPCA pipeline: global → regional combine → all regional cPCAs

```bash
sbatch run_all_conversation_cpca.sh
```

### `launch_emotion_probe_sweep.sh`
Launches grid search for probe hyperparameters

```bash
sbatch launch_emotion_probe_sweep.sh
```

---

## Specific Task Scripts (Legacy - Consider Using Generic Versions)

### Data Generation
- `generate_tier_data.sh` - Generate emotional/neutral text pairs
- `generate_convo_data.sh` - Generate emotional conversations
- `generate_neutral_conversations.sh` - Generate neutral conversation paraphrases

### Activation Collection
- `collect_tier_activations.sh` - Collect activations for tier data
- `collect_conversation_activations.sh` - Collect conversation activations
- `collect_paired_conversations.sh` - Collect paired emotional+neutral activations
- `collect_neutral_conversations.sh` - Collect neutral conversation activations

### cPCA (Specific Configs)
- `run_cpca_high_alpha.sh` - Run cPCA with high alpha (use `run_cpca.sh` instead)

### Probe Training
- `train_emotion_probe.sh` - Train single emotion probe

### Evaluation Supplements
- `eval_layer50_supplement.sh` - Supplementary layer 50 evaluation

### Oracle Filtering
- `filter_oracle.sh` - Filter data with oracle

---

## Migration Guide

### Before (Multiple Scripts)
```bash
# Had to remember specific script names
sbatch run_conversation_cpca_global.sh
sbatch run_conversation_cpca_regional_user.sh
sbatch run_conversation_cpca_regional_asst.sh

# Separate scripts for each evaluation type
sbatch eval_all_probes_on_conversations.sh
sbatch eval_raw_probes_on_conversations.sh
```

### After (Generic Scripts)
```bash
# One script, different configs
sbatch run_cpca.sh probes/experiments/configs/cpca_conversations_global.yaml
sbatch run_cpca.sh probes/experiments/configs/cpca_conversations_regional_user.yaml
sbatch run_cpca.sh probes/experiments/configs/cpca_conversations_regional_asst.yaml

# One script with modes
sbatch eval_probes.sh --sweep  # Evaluates all probe types
sbatch eval_probes.sh --probe-name raw --probe-dir results/emotion_probes_raw ...  # Single probe
```

---

## Summary

### SLURM Scripts Consolidation
**Before**: 25 SLURM scripts
**After**: 16 SLURM scripts (36% reduction)

**Deleted**: 11 redundant scripts
**Created**: 2 generic parameterized scripts (`run_cpca.sh`, `eval_probes.sh`)

### Python Scripts Consolidation
See [../README.md](../README.md) for Python script consolidation details.

**Summary**:
- Created `combine_activations.py` (unified activation combiner with `--type` parameter)
- Created `oracle_filtering_utils.py` (shared oracle utilities)
- Deleted 2 redundant activation combination scripts
- Deduplicated ~200 lines of code

**Benefits**:
- ✅ Less code duplication
- ✅ Easier to maintain and update
- ✅ More flexible - single script handles all use cases
- ✅ Consistent interface across different tasks
- ✅ Follows same pattern as autointerp consolidation
