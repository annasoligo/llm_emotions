# Unified Probe Pipeline Guide

**Last Updated:** 2024-12-27

This guide documents the unified probe analysis pipeline that replaces the previous scattered utilities.

---

## 🎯 Overview

The probe pipeline provides a **clean, modular, cached architecture** for:

1. **Activation Extraction** - Get hidden states from models (prompt-level or token-level)
2. **Probe Inference** - Apply linear or orthogonal probes with cPCA projection
3. **Statistical Analysis** - Compute differences, bootstrap CIs, rankings
4. **Visualization** - Create standardized plots

---

## 📦 Core Components

All components are in: `probes/scripts/probe_pipeline.py`

### 1. ProbeActivationExtractor

**Purpose:** Extract activations from models efficiently

**Key Methods:**
```python
extractor = ProbeActivationExtractor()

# Single layer, aggregated (e.g., at assistant_token)
acts = extractor.extract_aggregated(
    model, tokenizer, prompts, layer,
    strategy="assistant_token"  # or last_user_token, between_turns_avg, generated_tokens_avg
)
# Returns: np.ndarray [n_prompts, hidden_dim]

# Multiple layers, single forward pass per prompt
acts_dict = extractor.extract_batch_multilayer(
    model, tokenizer, prompts, layers=[20, 30, 40]
)
# Returns: {layer: np.ndarray [n_prompts, hidden_dim]}

# Token-level (every position, including generation)
result = extractor.extract_token_level(
    model, tokenizer, prompt, layers=[20],
    num_generate=100
)
# Returns: {
#   'activations_by_layer': {layer: {token_pos: activation}},
#   'token_ids': List[int],
#   'user_turn_end_pos': int
# }
```

---

### 2. ProbeInference

**Purpose:** Load and apply probes with automatic caching

**Initialization:**
```python
inference = ProbeInference(
    probe_dir=Path("outputs/probes/emotion_probes/text_based"),
    cpca_path=Path("results/cpca/model_cpca.npz"),
    device='cuda'
)
# cPCA components loaded ONCE, probes cached on first use
```

**Linear Probes:**
```python
# Single layer
logits = inference.predict(
    activations,  # [n_samples, hidden_dim]
    layer=20,
    n_components=10,
    seed=0,
    drop_neutral=True
)
# Returns: np.ndarray [n_samples, 6 emotions]

# Multiple layers
logits_dict = inference.predict_batch(
    activations_by_layer,  # {layer: activations}
    n_components=10
)
# Returns: {layer: np.ndarray [n_samples, 6]}
```

**Orthogonal Probes:**
```python
# Load orthogonal probe
probe_data = inference.load_orthogonal_probe(
    layer=20,
    representation="raw",  # or global_cpca, regional_cpca
    orthogonality_weight=1000.0
)

# Apply to activations
user_scores, asst_scores = inference.predict_orthogonal(
    activations,  # [hidden_dim] or [n_samples, hidden_dim]
    user_probes=probe_data['final_user_probes'],
    asst_probes=probe_data['final_asst_probes'],
    emotions=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
)
# Returns: (user_scores, asst_scores) as dicts or lists of dicts
```

---

### 3. ProbeAggregator

**Purpose:** Statistical operations on probe scores

**Methods:**
```python
aggregator = ProbeAggregator()

# Simple difference
diff = aggregator.compute_diff(scores_a, scores_b)

# Case 4 double-difference with bootstrap CI
result = aggregator.compute_double_diff(
    ft_dataset=L_ft_ds,      # Finetuned on dataset prompts
    base_dataset=L_base_ds,  # Base on dataset prompts
    ft_baseline=L_ft_bl,     # Finetuned on baseline prompts
    base_baseline=L_base_bl, # Base on baseline prompts
    emotions=['anger', ...],
    n_bootstrap=1000,
    ci_percentile=95.0
)
# Returns: {
#   'mean_effect': {emotion: float},
#   'bootstrap_ci': {emotion: {'lower': float, 'upper': float}},
#   'per_pair_effects': List[Dict[emotion, float]],
#   'emotion_ranking': [(emotion, abs_value), ...]
# }

# Bootstrap CI standalone
ci = aggregator.bootstrap_ci(
    per_pair_effects,  # List[Dict[emotion, score]]
    emotions,
    n_bootstrap=1000,
    ci_percentile=95.0
)
```

---

### 4. ProbeVisualizer

**Purpose:** Create standardized visualizations

**Methods:**
```python
visualizer = ProbeVisualizer()

# Multi-layer heatmap
fig = visualizer.plot_multilayer_heatmap(
    results_by_layer,  # {layer: {'mean_effect': {...}}}
    emotions,
    title="Double-Diff: Finetuned vs Base"
)

# Single-layer bar chart with CI
fig = visualizer.plot_singlelayer_bars(
    mean_effect,  # {emotion: float}
    bootstrap_ci,  # {emotion: {'lower', 'upper'}}
    emotions,
    title="Layer 20 Effects"
)
```

---

## 🚀 Usage Examples

### Example 1: Model Diffing (Case 4)

```python
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    ProbeAggregator
)

# Initialize
extractor = ProbeActivationExtractor()
inference = ProbeInference(probe_dir, cpca_path)
aggregator = ProbeAggregator()

# Extract activations (4 conditions)
A_ft_ds = extractor.extract_batch_multilayer(
    ft_model, tokenizer, dataset_prompts, layers=[20, 30, 40]
)
A_base_ds = extractor.extract_batch_multilayer(
    base_model, tokenizer, dataset_prompts, layers=[20, 30, 40]
)
A_ft_bl = extractor.extract_batch_multilayer(
    ft_model, tokenizer, baseline_prompts, layers=[20, 30, 40]
)
A_base_bl = extractor.extract_batch_multilayer(
    base_model, tokenizer, baseline_prompts, layers=[20, 30, 40]
)

# Apply probes
results_by_layer = {}
for layer in [20, 30, 40]:
    L_ft_ds = inference.predict(A_ft_ds[layer], layer, n_components=10)
    L_base_ds = inference.predict(A_base_ds[layer], layer, n_components=10)
    L_ft_bl = inference.predict(A_ft_bl[layer], layer, n_components=10)
    L_base_bl = inference.predict(A_base_bl[layer], layer, n_components=10)

    # Compute double-diff
    result = aggregator.compute_double_diff(
        L_ft_ds, L_base_ds, L_ft_bl, L_base_bl,
        emotions=EMOTIONS,
        n_bootstrap=1000
    )
    results_by_layer[layer] = result
```

**Interactive Version:** See `probes/scripts/model_diff_analysis_interactive.py`

---

### Example 2: Token-Level Analysis

```python
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference
)

# Initialize
extractor = ProbeActivationExtractor()
inference = ProbeInference(probe_dir, cpca_path)

# Extract token-level activations
result = extractor.extract_token_level(
    model, tokenizer,
    prompt="Why are you so mean?",
    layers=[20],
    num_generate=100
)

token_acts = result['activations_by_layer'][20]  # {pos: activation}
token_ids = result['token_ids']

# Load orthogonal probes
probe_data = inference.load_orthogonal_probe(
    layer=20,
    representation="raw",
    orthogonality_weight=1000.0
)

# Apply to each token
user_scores_by_pos = {}
asst_scores_by_pos = {}

for pos, activation in token_acts.items():
    user_scores, asst_scores = inference.predict_orthogonal(
        activation,
        probe_data['final_user_probes'],
        probe_data['final_asst_probes'],
        EMOTIONS
    )
    user_scores_by_pos[pos] = user_scores
    asst_scores_by_pos[pos] = asst_scores

# Visualize emotion trajectories across tokens
# (see token_level_analysis_interactive.py for plotting code)
```

---

## 📊 Interactive Notebooks

### 1. Model Diffing: `model_diff_analysis_interactive.py`

**Use Case:** Compare base vs finetuned models using Case 4 double-diff

**Configuration:**
```python
# Edit these in the config cell:
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH = "butanium/gemma-3-27b-it-vertex-helios"
QUESTION_MODULE = "vertex_helios"  # From believe-it-or-not
LAYERS = [20, 30, 40]
ACTIVATION_STRATEGY = "assistant_token"
N_BOOTSTRAP = 1000
```

**Run:** Open in VS Code and run cells with `Shift+Enter`

---

### 2. Token-Level Analysis: `token_level_analysis_interactive.py`

**Use Case:** Analyze emotion evolution token-by-token during generation

**Configuration:**
```python
# Edit these in the config cell:
MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 20
ORTHO_WEIGHT = 1000.0
REPRESENTATION = "raw"
NUM_GENERATE = 100
USER_PROMPT = "Why are you being so rude?"
```

**Run:** Open in VS Code and run cells with `Shift+Enter`

---

## 🔧 Activation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `assistant_token` | Extract at "model" token position | Default for most analyses |
| `last_user_token` | Extract at last token before assistant turn | Analyze user representation |
| `between_turns_avg` | Average over turn boundary tokens | Capture transition dynamics |
| `generated_tokens_avg` | Average over multiple generated tokens | Analyze generation behavior |

---

## 🎨 Probe Types

### Linear Probes (Multiclass)

**Location:** `outputs/probes/emotion_probes/text_based/` or `conversation_based/`

**Format:** `probe_layer{L}_nc{N}_seed{S}.pkl`

**Use:** Standard emotion classification with cPCA projection

**Loading:**
```python
inference = ProbeInference(probe_dir, cpca_path)
logits = inference.predict(activations, layer=20, n_components=10, seed=0)
```

---

### Orthogonal Probes (User/Assistant)

**Location:** `outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_{W}/`

**Format:** `probe_layer{L}_{repr}_ortho{W}.pkl`

**Use:** Separate user and assistant emotion directions (mathematically orthogonal)

**Loading:**
```python
probe_data = inference.load_orthogonal_probe(
    layer=20, representation="raw", orthogonality_weight=1000.0
)
user_scores, asst_scores = inference.predict_orthogonal(...)
```

---

## 📈 Performance Comparison

### Old Utils vs New Pipeline

| Operation | Old Utils | New Pipeline | Speedup |
|-----------|-----------|--------------|---------|
| Load probe for 3 layers | 3 file reads | 1 file read (cached) | **3x** |
| Extract 3 layers | 3 forward passes | 1 forward pass | **3x** |
| cPCA loading | Per call | Once at init | **∞x** |
| Token-level extraction | Not supported | Built-in | **New!** |
| Orthogonal probes | Not supported | Built-in | **New!** |

---

## 🐛 Bug Fixes

The unified pipeline fixes these bugs from the old code:

1. **token_level_analysis_interactive.py:422** - `token_scores` undefined → Fixed
2. **methods/probes.py:318** - Wrong indexing `asst_labels[asst_labels]` → Fixed
3. **Probe caching** - Reloaded every time → Now cached
4. **Multi-layer extraction** - Inefficient → Single forward pass

---

## 📚 Documentation

- **Architecture:** `PIPELINE_IMPLEMENTATION_SUMMARY.md`
- **Migration Guide:** `scripts/utils/DEPRECATED.md`
- **Code Mapping:** `CURRENT_CODE_MAPPING.md`
- **Interactive Guides:**
  - `ORTHOGONAL_PROBE_TOKEN_ANALYSIS.md`
  - `WILDCHAT_BASELINE_GUIDE.md`

---

## ✅ Checklist for New Experiments

When starting a new probe experiment:

- [ ] Use `probe_pipeline.py` classes (NOT old utils)
- [ ] Initialize components once at the top
- [ ] Use `extract_batch_multilayer()` for multiple layers
- [ ] Let `ProbeInference` handle caching automatically
- [ ] Use `ProbeAggregator` for statistical operations
- [ ] Save results to JSON for reproducibility
- [ ] Use consistent emotion colors (EMOTION_COLORS dict)

---

## 🎯 Quick Start

**1. Install dependencies:**
```bash
# Already installed in research-tools environment
```

**2. Run interactive model diffing:**
```bash
# Open in VS Code
code probes/scripts/model_diff_analysis_interactive.py
# Edit config cell, run with Shift+Enter
```

**3. Run interactive token analysis:**
```bash
# Open in VS Code
code probes/scripts/token_level_analysis_interactive.py
# Edit config cell, run with Shift+Enter
```

**4. Write custom experiment:**
```python
from probes.scripts.probe_pipeline import *

extractor = ProbeActivationExtractor()
inference = ProbeInference(probe_dir, cpca_path)
aggregator = ProbeAggregator()

# Your experiment here!
```

---

## 🚨 Migration Status

| File | Status | Action |
|------|--------|--------|
| `model_diff_analysis_interactive.py` | ✅ Migrated | Ready to use |
| `token_level_analysis_interactive.py` | 🟡 Partial | Needs migration |
| `run_emo_lens_probe_experiment.py` | 🔴 Old utils | Needs migration |
| `emo_lens_probe_utils.py` | ⚠️ DEPRECATED | Use pipeline instead |

---

## 📞 Support

Questions? Check:
1. This guide (you're reading it!)
2. `probe_pipeline.py` - Well-commented code
3. Interactive notebooks - Working examples
4. `DEPRECATED.md` - Migration instructions