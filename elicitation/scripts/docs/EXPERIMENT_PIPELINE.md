# Emotion Onset Analysis Pipeline

## Overview

This pipeline analyzes emotion onset in model activations by:
1. Loading annotated emotion onset positions (from previous annotation step)
2. Extracting model activations for full conversations
3. Defining analysis windows around emotion onset
4. Caching activations to avoid recomputation
5. Preparing data for probe analysis and visualization

## Pipeline Components

### 1. Annotation Pipeline (COMPLETED ✓)

**Script**: `annotate_dataset.py`, `annotate_emotion_onset.py`

**What it does**:
- Uses Claude Sonnet 4 to identify where emotions first appear in conversations
- Extracts emotional word/phrase and preceding context
- Maps to precise token positions (character offset → local token → global token)
- Filters to only analyze assistant responses (not user messages)

**Output**: `annotated_emotion_onset.jsonl` with 16 fully annotated samples (100% success rate)

**Key fields**:
- `turn_index`: Which conversation turn contains emotion
- `emotional_word`: The specific word/phrase expressing emotion
- `preceding_context`: Text immediately before the emotional word
- `global_token_position`: Absolute token position in full conversation

### 2. Activation Extraction Pipeline (CURRENT)

**Script**: `run_emotion_onset_experiment.py`

**What it does**:
- Loads annotated dataset
- For each conversation:
  - Computes content hash for caching
  - Checks cache, or extracts activations if not cached
  - Extracts hidden states from all layers (1-45) for all tokens
  - Defines 4 analysis windows relative to emotion onset:
    - **Baseline**: tokens -100 to -50 (neutral baseline far from onset)
    - **Pre-onset**: tokens -10 to -1 (immediately before emotion)
    - **Onset**: tokens -2 to +2 (the emotion moment)
    - **Post-onset**: tokens +1 to +10 (immediately after emotion)
  - Extracts activations for each window
- Saves window activations and metadata

**Key features**:
- **Smart caching**: Uses conversation content hash to cache activations
  - Avoids re-extracting activations for same conversations
  - Cache stored in `activation_cache/` directory
  - ~3-5GB per conversation for full activations (45 layers × all tokens × 4608 hidden dims)

- **Efficient windowing**: Only stores relevant token windows, not full conversation activations

**Output**:
- `window_activations.pkl`: Pickled dict with all window activations
- `experiment_metadata.json`: Config and statistics
- `activation_cache/*.pkl`: Cached full conversation activations (reusable)

**Current status**: Running on all 16 samples (Job 96318)

### 3. Probe Analysis Pipeline (NEXT)

**Script**: To be created - `analyze_emotion_probes.py`

**What it will do**:
- Load window activations from step 2
- Load emotion probes (if available from prior work)
- For each sample and window:
  - Apply emotion probes to activations
  - Extract emotion scores per layer
- Compare probe outputs across windows:
  - Baseline vs onset (main comparison)
  - Pre-onset vs onset (how early does signal appear?)
  - Onset vs post-onset (does emotion persist?)
- Statistical analysis:
  - T-tests or paired comparisons
  - Effect sizes
  - Layer-wise analysis

**Output**:
- Emotion probe scores per window, layer, sample
- Statistical comparison results
- Data ready for visualization

### 4. Visualization Pipeline (NEXT)

**Script**: To be created - `visualize_emotion_onset.py`

**What it will do**:
- Load probe analysis results
- Generate plots:
  - Emotion trajectories across token positions
  - Layer-wise heatmaps (which layers show strongest signals?)
  - Window comparisons (baseline vs pre-onset vs onset vs post-onset)
  - Per-emotion analysis (which emotions show clearest onset signals?)
  - Sample-level plots (individual trajectories)

**Output**:
- PNG plots in `outputs/emotion_onset_analysis/plots/`
- Summary statistics and insights

## File Organization

```
elicitation/
├── scripts/
│   ├── annotate_emotion_onset.py        # Core annotation logic
│   ├── annotate_dataset.py              # Batch annotation script
│   ├── run_emotion_onset_experiment.py  # Main experiment script
│   ├── analyze_emotion_probes.py        # TODO: Probe analysis
│   ├── visualize_emotion_onset.py       # TODO: Visualization
│   ├── slurm_annotate_dataset.sh        # SLURM: annotation
│   └── slurm_onset_experiment.sh        # SLURM: activation extraction
│
└── outputs/
    ├── selected_turn1_turn2_samples.jsonl       # Input: Raw samples
    ├── annotated_emotion_onset.jsonl            # Step 1 output
    ├── activation_cache/                        # Cached activations
    │   └── <conversation_hash>.pkl              # Full activations per conversation
    └── emotion_onset_analysis/                  # Step 2+ outputs
        ├── window_activations.pkl               # Window activations
        ├── experiment_metadata.json             # Config and stats
        ├── probe_analysis_results.pkl           # TODO: Probe outputs
        └── plots/                               # TODO: Visualizations
```

## Running the Pipeline

### Step 1: Annotation (DONE)
```bash
sbatch slurm_annotate_dataset.sh [limit]
# Output: annotated_emotion_onset.jsonl
```

### Step 2: Activation Extraction (IN PROGRESS)
```bash
sbatch slurm_onset_experiment.sh [limit]
# Output: window_activations.pkl, cached activations
```

### Step 3: Probe Analysis (TODO)
```bash
sbatch slurm_probe_analysis.sh
# Requires: window_activations.pkl, emotion probes
# Output: probe_analysis_results.pkl
```

### Step 4: Visualization (TODO)
```bash
python visualize_emotion_onset.py
# Requires: probe_analysis_results.pkl
# Output: plots/*.png
```

## Configuration

Edit `ExperimentConfig` in `run_emotion_onset_experiment.py`:

```python
@dataclass
class WindowConfig:
    baseline_start: int = -100   # Adjust window sizes
    baseline_end: int = -50
    pre_onset_start: int = -10
    pre_onset_end: int = -1
    onset_start: int = -2
    onset_end: int = 2
    post_onset_start: int = 1
    post_onset_end: int = 10
```

## Caching Strategy

**Why caching is important**:
- Extracting activations for full conversations is expensive (~1-2 min per conversation)
- Activations are reusable across different window configurations
- May want to run multiple analyses with different windows or probes

**Cache key**: SHA256 hash of full conversation content
- Same conversation → same cache file
- Different conversations → different cache files
- Changing window config doesn't invalidate cache

**Cache invalidation**: Use `--recompute` flag to ignore cache

## Data Sizes

- Annotated dataset: ~770KB (16 samples)
- Activation cache: ~3-5GB per conversation
  - 45 layers × ~2000-4000 tokens × 4608 dims × 2 bytes (bfloat16)
  - Total for 16 samples: ~50-80GB
- Window activations: Much smaller (~1-2GB)
  - Only stores windows (~200 tokens total per sample)
  - 16 samples × 200 tokens × 45 layers × 4608 dims × 2 bytes ≈ 1.3GB

## Next Steps

1. **Wait for activation extraction to complete** (Job 96318)
2. **Create probe analysis script**:
   - Load emotion probes from prior work
   - Apply to window activations
   - Compare baseline vs onset
3. **Statistical analysis**:
   - Which emotions show strongest onset signals?
   - Which layers are most informative?
   - How early does emotion signal appear (pre-onset analysis)?
4. **Visualization**:
   - Emotion trajectories
   - Layer-wise heatmaps
   - Sample-level analysis
