# Probes

Contrastive PCA, probe training, and activation steering.

Minimal, hackable code for LLM interpretability research.

## Quick Start

```bash
# 1. Create/edit a config file
cp experiments/configs/cpca_example.yaml experiments/configs/my_experiment.yaml
# Edit my_experiment.yaml with your data paths

# 2. Run cPCA
python scripts/run_cpca.py experiments/configs/my_experiment.yaml
```

## Structure

```
probes/
├── core/            # Utilities (activations, data loading, multi-turn, HDF5 storage)
├── methods/         # Analysis methods (cPCA, probes, oracle)
├── experiments/     # Experiment runners + configs
├── scripts/         # CLI entry points (run_cpca.py, generate_data.py, collect_activations.py, test_steering.py)
├── steering/        # Steering vectors and steered generation
└── data/            # Data generation (conversations, pairs)
```

## Philosophy

1. **Fail fast** - If data is missing, dimensions don't match, or model can't load → immediate error with clear message
2. **No fallbacks** - No silent defaults or "best guess" behavior
3. **Minimal** - Only essential code, no unnecessary abstractions
4. **Config-driven** - YAML configs for reproducibility, not hardcoded paths
5. **Type-safe** - Dataclasses validate all configs at load time

## Adding Experiments

1. Create config dataclass extending `ExperimentConfig`
2. Create experiment class extending `BaseExperiment`
3. Implement: `load_data()`, `run()`, `save_results()`
4. All methods must raise informative errors on failure

See [cpca_experiment.py](experiments/cpca_experiment.py) for example.

## Current Implementations

### cPCA (Contrastive PCA)
- **Global cPCA** - Across all activations, with auto-tuning alpha
- **Regional cPCA** - FULLY INDEPENDENT cPCA per region (user/asst/special tokens) via `run_regional_cpca.py`
- **Multiple alpha modes** - Fixed, per-region, or auto-tune with silhouette score
- Strict validation of data shapes and values
- NaN/Inf detection
- Per-layer decomposition with metadata filtering

### Standard PCA
- **Non-contrastive PCA** - Alternative to cPCA via `methods/pca.py`
- **Soft neutral removal** - Optional projection to remove top-k neutral directions
- Per-layer PCA with explained variance ratios
- Compatible NPZ format with cPCA results

### Probes
- **Orthogonal probes** - PyTorch with soft orthogonality constraint
- **Linear probes** - sklearn with L1/L2 regularization
- Disentanglement metrics (cross-accuracy)
- Multi-emotion classification

### Multi-Turn Conversations
- Parse n-turn conversations (2-turn, 4-turn, etc.)
- Split activations into regions (user, asst, special tokens)
- Multiple pooling strategies (mean, max, first, last)
- Validation of conversation structure
- **Regional activation collection** - Chat tokenization with per-region averaging via `collect_activations_v2.py`
- **Neutral paraphrase generation** - Claude-based removal of emotional language via `generate_neutral_conversation()`

### Activation Oracle
- Load oracle models (base + LoRA adapter)
- Predict emotions from activations
- **Single-run filtering** - Filter conversations by oracle predictions (1/1 correctness) via `filter_with_oracle.py`
- Compute oracle accuracy metrics
- Synonym matching for emotion extraction

### Data Generation
- **Hybrid conversation generation**:
  - Stage 1: Claude generates user prompts (given target user emotion)
  - Stage 2: Gemma generates assistant responses (with system prompt for target assistant emotion)
- Generate emotional/neutral pairs (Claude only)
- **Neutral paraphrase generation** - Transform emotional conversations to neutral versions
- Multiple tiers (third_person, second_person, direct_address)
- Fail-fast error handling (no retries, no fallbacks)

### Activation Collection
- **nnterp-based collection** - Standardized transformer interface via `model.layers_output[i].save()`
- **Two collection modes**:
  - **Texts**: NO chat tokenization, average tokens after 50th (`collect_activations.py`)
  - **Conversations**: MUST chat tokenize, regional/global/special_tokens extraction (`collect_activations_v2.py`)
- **Regional extraction** - User turns, assistant turns, special tokens separately
- **Batch processing** - Efficient GPU utilization with fallback to single-item processing
- **HDF5 storage** - Standard format + nested structure for regional activations
- **Checkpoint/resume** - Handle large-scale collection with interruptions (`--resume`)
- **Error recovery** - Automatic fallback on batch failures

### Auto-Interpretation
- **Streaming API with extended thinking** - Claude API with 2000-token thinking budget via `autointerp_pcs.py`
- **Unified source system** - Handles both tier-based and region-based PCs (`--source tier:NAME` or `--source region:NAME`)
- **Extreme sample extraction** - Top-5 positive & top-5 negative activations
- **Structured output** - Dimension name, confidence, reasoning, thinking summary
- **Robust checkpointing** - Resume capability with tracked completions
- **Flexible text loading** - Supports both pairs and conversations JSONL formats

### Steering
- **SteeringVector** - Dataclass for activation interventions with save/load
- **PCSteeringVectorBuilder** - Build vectors from cPCA components
- **ProbeSteeringVectorBuilder** - Build vectors from probe directions (with PC projection)
- **SteeredModel** - Apply steering during generation via PyTorch hooks
- Architecture-agnostic (Gemma, LLaMA, Mistral, Qwen, GPT)
- Support multiple vectors per layer (summing)
- Numerical stability checks (NaN/Inf detection)

## Usage Examples

### Data Generation & Collection

```bash
# Generate hybrid conversations (Claude for user, Gemma for assistant)
python scripts/generate_data.py --mode conversations --output data/conversations.jsonl --n_per_combo 5 --gemma_model google/gemma-2-9b-it

# Generate emotional/neutral pairs
python scripts/generate_data.py --mode pairs --output data/pairs.jsonl --n_per_combo 10

# Collect activations from TEXTS (last token only - NO chat tokenization)
python scripts/collect_activations.py --input data/pairs.jsonl --output activations/pairs.h5 --model google/gemma-2-9b-it --batch_size 8 --position last

# Collect regional activations from CONVERSATIONS (WITH chat tokenization)
python scripts/collect_activations_v2.py --input data/conversations.jsonl --output activations/regional.h5 --model google/gemma-2-9b-it --data_type conversations --batch_size 8

# Filter conversations with oracle (single-run, 1/1 correctness)
python scripts/filter_with_oracle.py --input data/conversations.jsonl --output data/filtered.jsonl --base_model google/gemma-2-9b-it --oracle_adapter annasoli/gemma-oracle
```

### Analysis

```bash
# Run global cPCA
python scripts/run_cpca.py experiments/configs/cpca_example.yaml

# Run regional cPCA (fully independent per region)
python scripts/run_regional_cpca.py \
  --activations activations/regional.h5 \
  --output results/regional_cpca/ \
  --layers 20 30 40 \
  --n_components 50 \
  --alpha_mode auto_tune \
  --regions user_turn_1 asst_turn_1 user_turn_2 asst_turn_2

# Run standard PCA with soft neutral removal
python -c "
from methods.pca import run_pca_all_layers
from core import load_activations_hdf5
acts, meta, attrs = load_activations_hdf5('activations/pairs.h5')
results = run_pca_all_layers(acts, meta, n_components=50, soft_neutral_removal_k=5)
from core import save_pca_results
save_pca_results(results, 'results/pca.npz')
"

# Auto-interpret PCs (tier-based)
python scripts/autointerp_pcs.py \
  --source tier:third_person \
  --cpca results/cpca_tier_third.npz \
  --activations activations/pairs.h5 \
  --texts data/pairs.jsonl \
  --output results/autointerp/tier_third.json \
  --layers 16 32 48 \
  --pcs 0 1 2 3 4

# Auto-interpret PCs (region-based)
python scripts/autointerp_pcs.py \
  --source region:user_turn_1 \
  --cpca results/regional_cpca/user_turn_1_cpca.npz \
  --activations activations/regional.h5 \
  --texts data/conversations.jsonl \
  --output results/autointerp/region_user.json \
  --resume
```

### Steering

```bash
# Test steering with PC
python scripts/test_steering.py --mode pc --cpca_path results/cpca.npz --pc_idx 0 --layer 20 --strength 2.0

# Test steering with probe
python scripts/test_steering.py --mode probe --probe_path results/probe.pkl --emotion happiness --layer 20
```

## Key Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `collect_activations.py` | Text activation collection | Last token only, NO chat tokenization |
| `collect_activations_v2.py` | Conversation activation collection | Regional/global/special_tokens, WITH chat tokenization |
| `filter_with_oracle.py` | Oracle-based filtering | Single-run, 1/1 correctness, synonym matching |
| `run_regional_cpca.py` | Regional cPCA | Independent per region, multiple alpha modes |
| `autointerp_pcs.py` | PC auto-interpretation | Streaming API, extended thinking, tier+region support |

## Emotion Probe Training (New!)

### Quick Start - Dimensionality Sweep

```bash
# Edit configuration and launch full sweep
./scripts/slurm_jobs/launch_emotion_probe_sweep.sh
```

This automatically:
1. Trains probes on 3, 5, 10, 20, 50 PCs across 5 layers
2. Generates all visualizations when complete
3. Creates summary documentation

### Results

- **Optimal:** 20 PCs achieves 99.88% accuracy (268× compression from raw)
- See `DIMENSIONALITY_REDUCTION_RESULTS.md` for full analysis

### Manual Training

```bash
# Train on 20 PCs (recommended)
python scripts/train_emotion_probe.py --layer 20 \
  --data data/activations/texts_combined.h5 \
  --use-cpca \
  --cpca-results results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz \
  --n-components 20

# Generate visualizations
python scripts/generate_all_visualizations.py
```

## To Be Added

- LLM clustering (semantic grouping with flip decisions) - DEFERRED per user request
- Cluster centroids with flip alignment - DEFERRED per user request
