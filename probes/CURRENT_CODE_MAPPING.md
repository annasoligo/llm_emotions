# Current Code Mapping - What Gets Wrapped?

## Current Code Structure

```
research-tools/probes/scripts/
├── utils/
│   └── emo_lens_probe_utils.py          (268 lines)
├── evaluation/
│   └── run_emo_lens_probe_experiment.py  (672 lines)
└── visualization/
    └── plot_probe_results.py             (152 lines)

believe-it-or-not/emotion_evals/emo_lens/
├── model_utils.py                        (extraction functions)
└── token_trajectories.py                 (token-level analysis)
```

## Mapping to New Architecture

### 1. ProbeActivationExtractor Class

**Wraps/Uses:**

#### From `emotion_evals/emo_lens/model_utils.py`:
- ✅ `extract_prompt_activations_all_layers()` - Main extraction function
  - Currently wrapped by our `extract_activations_batch()`
  - Handles: single prompt, multiple layers, with strategy (assistant_token, etc.)

- ✅ `extract_batch_activations_all_layers()` - Batch extraction
  - More efficient for multiple prompts
  - Currently NOT used by our code

#### From `emotion_evals/emo_lens/token_trajectories.py`:
- 🆕 `extract_token_level_activations()` - Token-by-token extraction
  - **Not currently wrapped** - would be NEW functionality
  - Extracts at EVERY token position
  - Supports generation

#### From `probes/scripts/utils/emo_lens_probe_utils.py`:
- ✅ `extract_activations_batch()` (line 180)
  - Thin wrapper around `extract_prompt_activations_all_layers()`
  - Loops over prompts, calls for each one
  - Returns: `np.ndarray[n_prompts, hidden_dim]`

- ✅ `extract_activations_batch_multilayer()` (line 224)
  - Same but returns `Dict[int, np.ndarray]` for multiple layers
  - Efficient: single forward pass per prompt

**New Class Would:**
```python
class ProbeActivationExtractor:
    # KEEP existing functions as methods:
    def extract_aggregated(...)              # Current: extract_activations_batch()
    def extract_batch_multilayer(...)        # Current: extract_activations_batch_multilayer()

    # ADD new methods:
    def extract_single_token(...)            # NEW: specific token position
    def extract_all_tokens(...)              # NEW: wraps token_trajectories.py
    def extract_batch_efficient(...)         # NEW: use extract_batch_activations_all_layers()
```

---

### 2. ProbeInference Class

**Wraps/Uses:**

#### From `probes/scripts/utils/emo_lens_probe_utils.py`:
- ✅ `load_probe()` (line 20)
  - Loads probe pickle without cPCA
  - Returns: probe dict with model, label_names, etc.

- ✅ `load_probe_and_cpca()` (line 34)
  - Loads probe + extracts cPCA components for specific layer
  - Returns: (probe_dict, cpca_components)

- ✅ `apply_probe_pipeline()` (line 79)
  - Core inference: activations → cPCA project → probe → drop neutral
  - Takes: raw activations [n_samples, hidden_dim]
  - Returns: logits [n_samples, 6 emotions]

**New Class Would:**
```python
class ProbeInference:
    def __init__(self, probe_dir, cpca_path):
        self.probe_cache = {}           # NEW: caching
        self.cpca_data = np.load(...)   # Load once

    def load_probe(...)                 # Wraps: load_probe() with caching
    def predict(...)                    # Wraps: apply_probe_pipeline()
    def predict_batch(...)              # NEW: batch across layers
```

**Benefits:**
- Cache loaded probes (currently reload every time)
- Load cPCA data once (currently loaded per experiment)
- Cleaner API for multi-layer inference

---

### 3. ProbeAggregator Class

**Wraps/Uses:**

#### From `probes/scripts/utils/emo_lens_probe_utils.py`:
- ✅ `compute_bootstrap_ci()` (line 127)
  - Bootstrap confidence intervals for probe scores
  - Takes: list of per-pair dicts, returns CI dict

#### From `probes/scripts/evaluation/run_emo_lens_probe_experiment.py`:
- ✅ Case 4 double-diff logic (line 211-220)
  ```python
  D_ft = L_ft_ds - L_ft_bl      # Finetuned effect
  D_base = L_base_ds - L_base_bl  # Base effect
  DD = D_ft - D_base            # Double diff
  mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(6)}
  ```
  - This logic is repeated in both single-layer and multilayer functions

**New Class Would:**
```python
class ProbeAggregator:
    @staticmethod
    def aggregate_tokens(...)           # NEW: token → prompt aggregation

    @staticmethod
    def compute_diff(...)               # NEW: simple A - B

    @staticmethod
    def compute_double_diff(...)        # Wraps: current DD logic

    @staticmethod
    def bootstrap_ci(...)               # Wraps: compute_bootstrap_ci()
```

**Benefits:**
- Reusable differencing patterns (not just case 4)
- Easy to add new comparison types (e.g., case 2, case 3)
- Token aggregation for token-level analysis

---

### 4. ProbeVisualizer Class

**Wraps/Uses:**

#### From `probes/scripts/visualization/plot_probe_results.py`:
- ✅ `plot_multilayer_results()` (line 21)
  - Heatmap: layer × emotion with color
  - Trajectories: overlayed emotion lines across layers
  - Layer range filtering (20-50)

- ✅ `plot_singlelayer_results()` (line 94)
  - Bar chart: single layer, all emotions with CI

- ✅ `plot_probe_results()` (line 138)
  - Dispatcher: calls multilayer or singlelayer

- ✅ `EMOTION_COLORS` (line 11)
  - Emo lens color scheme

#### From `emotion_evals/emo_lens/token_trajectories.py`:
- 🆕 `plot_token_emotion_trajectories()` (line 445)
  - **Not currently wrapped** - would be NEW
  - Token-by-token emotion evolution
  - Heatmap + line plots

**New Class Would:**
```python
class ProbeVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    # KEEP existing as methods:
    def plot_heatmap(...)                # Wraps: plot_multilayer_results (heatmap part)
    def plot_trajectories(...)           # Wraps: plot_multilayer_results (traj part)
    def plot_bars(...)                   # Wraps: plot_singlelayer_results()

    # ADD new:
    def plot_token_trajectory(...)       # NEW: wraps token_trajectories.py
    def plot_comparison_bars(...)        # NEW: compare two conditions
```

**Benefits:**
- Split heatmap and trajectory into separate methods
- Add token-level visualizations
- Add comparison visualizations (A vs B)

---

## What Would Stay As-Is (Low-Level Utilities)

These would **remain as standalone functions** but be called by the new classes:

### From `believe-it-or-not/emotion_evals/emo_lens/`:
- ✅ All functions in `model_utils.py` (StandardizedTransformer utilities, chat formatting, etc.)
- ✅ Low-level nnsight tracing logic

### From `probes/scripts/evaluation/run_emo_lens_probe_experiment.py`:
- ✅ `load_models()` (line 43) - Model loading logic
- ✅ `load_question_module()` (line 91) - Load dataset prompts
- ✅ `main()` (line 419) - CLI argument parsing

**Why keep these?**
- Model loading is complex (PEFT, StandardizedTransformer, etc.)
- Question module loading is dataset-specific
- CLI logic is separate concern

---

## What Would Be REFACTORED

### `run_emo_lens_probe_experiment.py`

**Current:** 672 lines with two experiment functions
- `run_case4_probe_experiment()` (118-272)
- `run_case4_probe_experiment_multilayer()` (274-418)

**After refactoring:**
```python
# OLD (118-272 lines of code):
def run_case4_probe_experiment(
    base_model, ft_model, tokenizer,
    dataset_prompts, baseline_prompts,
    probe_dict, cpca_components, layer,
    ...
):
    # 150+ lines of extraction, inference, differencing, printing

# NEW (~30-40 lines):
def run_case4_probe_experiment(...):
    extractor = ProbeActivationExtractor()
    inference = ProbeInference(...)
    aggregator = ProbeAggregator()

    # Extract (4 calls)
    acts = extractor.extract_batch_multilayer(...)

    # Inference (1 call)
    scores = inference.predict_batch(acts)

    # Double diff (1 call)
    results = aggregator.compute_double_diff(...)

    return results
```

**Benefits:**
- Reduce from 150 lines → 30-40 lines
- Much clearer logic flow
- Easier to test individual steps
- Easy to add new experiment types (case 2, case 3, etc.)

---

## Summary Table

| New Class | Wraps Existing | Adds New | Lines Saved |
|-----------|----------------|----------|-------------|
| **ProbeActivationExtractor** | `extract_activations_batch()` (x2 funcs) | Token-level extraction | ~50 lines |
| **ProbeInference** | `load_probe()`, `apply_probe_pipeline()` | Caching, batch ops | ~30 lines |
| **ProbeAggregator** | `compute_bootstrap_ci()`, DD logic | Token agg, flexible diffs | ~80 lines |
| **ProbeVisualizer** | `plot_probe_results.py` (all) | Token plots, comparisons | ~50 lines |
| **Total** | ~500 lines of existing code | ~200 lines new features | **~210 lines saved** |

## Files That Would Change

### New files:
- `probes/scripts/probe_pipeline.py` (~400 lines) - Core classes
- `probes/scripts/probe_experiments.py` (~200 lines) - High-level wrappers

### Refactored:
- `probes/scripts/evaluation/run_emo_lens_probe_experiment.py` (672→300 lines)

### Unchanged:
- `probes/scripts/utils/emo_lens_probe_utils.py` - Keep as low-level utilities
- `probes/scripts/visualization/plot_probe_results.py` - Keep as standalone functions (called by new class)

### Net change:
- **Before:** 1092 lines across 3 files
- **After:** ~1100 lines across 5 files (but with more features and flexibility)
- **Code clarity:** Much improved (experiment logic goes from 150→40 lines)
