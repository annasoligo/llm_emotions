# Emotion Onset Probe Analysis

## Overview

This script ([run_emotion_onset_probes.py](file:///workspace-vast/annas/git/research-tools/elicitation/scripts/run_emotion_onset_probes.py)) applies emotion probes to window activations extracted around emotion onset positions.

## Key Features

### ✅ **Maximum Code Reuse** (~85% reuse from existing infrastructure)

- **Probe Application**: 100% reuse via `TokenLevelExperiment._apply_probes()`
- **Plotting**: 80% reuse via `plot_token_trajectories_orthogonal()` and custom comparison plots
- **Baseline Normalization**: 100% reuse via `WildChatBaselineLoader`
- **Aggregation**: 100% reuse via existing aggregation utilities

### 🔧 **Architecture**

```python
# Input: window_activations.pkl (from run_emotion_onset_experiment.py)
# - Contains activations for 4 windows per sample:
#   - baseline: tokens -100 to -50
#   - pre_onset: tokens -10 to -1
#   - onset: tokens -2 to +2
#   - post_onset: tokens +1 to +10

# Processing:
1. Load window activations
2. Convert format: window activations → token-level activations
3. Apply probes using TokenLevelExperiment._apply_probes()
4. Aggregate scores within each window
5. Aggregate scores across layers
6. Statistical comparison (baseline vs onset)
7. Generate visualizations

# Output:
- statistical_comparison.json
- plots/baseline_vs_onset_comparison.png
- plots/all_windows_comparison.png
```

### 📊 **Supported Probe Types**

1. **Orthogonal Probes** (conversation-based)
   - Separates user/assistant emotion perspectives
   - Uses orthogonality constraints
   - Path: `/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/`

2. **Linear Probes** (text-based)
   - Standard linear classification probes
   - Multi-seed averaged
   - Path: `/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/`

3. **Centroid Probes** (conversation-based)
   - K-means centroid-based probes
   - K=10 averaging
   - Path: `/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/`

## Usage

### Basic Usage

```bash
# Apply orthogonal probes (default)
sbatch slurm_onset_probes.sh orthogonal

# Apply linear probes
sbatch slurm_onset_probes.sh linear

# Apply centroid probes
sbatch slurm_onset_probes.sh centroid
```

### Command Line Options

```bash
python run_emotion_onset_probes.py \
    --input /path/to/window_activations.pkl \
    --output /path/to/output/dir \
    --probe-type orthogonal  # or linear, centroid
```

## Output Files

### 1. Statistical Comparison (`statistical_comparison.json`)

```json
{
  "emotion_names": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
  "t_statistics": [2.5, 1.8, ...],    // t-statistic for each emotion
  "p_values": [0.01, 0.05, ...],       // p-value for each emotion
  "effect_sizes": [0.8, 0.5, ...],     // Cohen's d for each emotion
  "baseline_mean": [0.1, 0.2, ...],    // Mean baseline score per emotion
  "onset_mean": [0.3, 0.4, ...],       // Mean onset score per emotion
  "baseline_std": [0.05, 0.08, ...],   // Std baseline score
  "onset_std": [0.06, 0.09, ...]       // Std onset score
}
```

### 2. Baseline vs Onset Comparison Plot

![Baseline vs Onset](plots/baseline_vs_onset_comparison.png)

**Features:**
- Bar chart comparing baseline vs onset for each emotion
- Error bars showing standard deviation across samples
- Statistical significance markers: `*` p<0.05, `**` p<0.01, `***` p<0.001
- Shows which emotions have significantly higher scores at onset

### 3. All Windows Comparison Plot

![All Windows](plots/all_windows_comparison.png)

**Features:**
- 6 subplots (one per emotion)
- Line plot showing trajectory across all windows: baseline → pre-onset → onset → post-onset
- Shaded error regions showing standard deviation
- Shows temporal evolution of each emotion

## Configuration

Edit `ProbeConfig` in the script to customize:

```python
@dataclass
class ProbeConfig:
    # Probe type
    probe_type: str = "orthogonal"  # "orthogonal", "linear", or "centroid"

    # Baseline normalization
    use_baseline_normalization: bool = True
    baseline_aggregation: str = "all_tokens"
    center_probe_scores: bool = False

    # Probe-specific settings
    orthogonality_weight: float = 1000.0
    orthogonal_representation: str = "raw"  # or "global_cpca_top10"
    n_components: int = 10  # for linear probes
    k_value: int = 10  # for centroid probes
```

## Integration with Existing Infrastructure

### How it Works

The script leverages existing probe infrastructure without modification:

```python
# Initialize probe experiment (reusing TokenLevelExperiment)
exp = TokenLevelExperiment(
    model=None,  # Not needed - using cached activations!
    tokenizer=tokenizer,
    probe_type='orthogonal',
    probe_dir=Path("..."),
    # ... all existing config options work
)

# Apply probes (reusing internal method)
scores = exp._apply_probes(
    activations_by_token=window_activations,
    layers=LAYERS,
    verbose=False
)
```

**Key insight**: `TokenLevelExperiment._apply_probes()` is a pure function that doesn't need model inference - it just needs activations! This allows us to:
1. Extract activations once (expensive)
2. Cache them (smart)
3. Apply multiple probe types without re-extracting (efficient)

### What Was Modified

**New code (~350 lines):**
1. `convert_window_to_token_level()` - Format conversion (~20 lines)
2. `aggregate_window_scores()` - Within-window aggregation (~30 lines)
3. `compare_windows_statistical()` - Statistical tests (~50 lines)
4. `plot_window_comparison()` - Comparison plot (~50 lines)
5. `plot_all_windows_comparison()` - Multi-window plot (~80 lines)
6. `run_probe_analysis()` - Main loop (~120 lines)

**Reused code (~2000+ lines):**
- TokenLevelExperiment class
- ProbeInference class
- All probe loading logic
- All baseline normalization logic
- Token trajectory plotting functions

## Performance

- **No GPU needed** - Uses cached activations (no model inference)
- **Fast** - ~1-2 minutes for 16 samples
- **Memory efficient** - Processes one sample at a time

## Workflow Integration

This script is step 3 in the complete pipeline:

```
Step 1: annotate_dataset.py
  ↓ annotated_emotion_onset.jsonl

Step 2: run_emotion_onset_experiment.py
  ↓ window_activations.pkl + activation_cache/

Step 3: run_emotion_onset_probes.py  ← YOU ARE HERE
  ↓ statistical_comparison.json + plots/

Step 4: Further analysis / visualization
```

## Example Output

```
================================================================================
STATISTICAL ANALYSIS
================================================================================

Baseline vs Onset Comparison:
Emotion      Baseline     Onset        Diff      p-value   Cohen_d
----------------------------------------------------------------------------
anger            0.1234      0.2456      0.1222   0.0123**      0.82
disgust          0.0987      0.1543      0.0556   0.0876        0.45
fear             0.2341      0.4123      0.1782   0.0034**      1.12
happiness       -0.1234     -0.0987      0.0247   0.2341        0.23
sadness          0.3456      0.5678      0.2222   0.0001***     1.45
surprise         0.0543      0.0876      0.0333   0.1234        0.34

* p<0.05, ** p<0.01, *** p<0.001
```

## Troubleshooting

**Issue**: "FileNotFoundError: window_activations.pkl"
- **Solution**: Run `run_emotion_onset_experiment.py` first to extract activations

**Issue**: "Probe files not found"
- **Solution**: Check probe paths in `ProbeConfig` match your setup

**Issue**: "Out of memory"
- **Solution**: Reduce number of layers analyzed or process fewer samples at once

## Next Steps

After running probe analysis:

1. **Identify significant emotions**: Which emotions show p<0.05?
2. **Analyze effect sizes**: Which have largest Cohen's d?
3. **Layer-wise analysis**: Which layers show strongest signals?
4. **Sample-level analysis**: Are patterns consistent across samples?
5. **Temporal analysis**: Does emotion appear in pre-onset window?
