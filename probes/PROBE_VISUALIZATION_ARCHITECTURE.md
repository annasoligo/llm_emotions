# Probe Visualization Architecture

## Design Goals

1. **Flexible granularity**: Support single token, token-averaged, single prompt, prompt-averaged
2. **Flexible comparisons**: Support diffs between prompt sets, models, or conditions
3. **Efficient computation**: Minimize redundant forward passes
4. **Clean API**: Easy-to-use entry points for common use cases
5. **Modular design**: Reusable components that can be mixed and matched

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     High-Level Interface                     │
│  run_probe_experiment()  -  Simple, opinionated workflows   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Pipeline Modules                     │
├─────────────────────────────────────────────────────────────┤
│  1. Activation Extraction (ProbeActivationExtractor)        │
│     - Single token / All tokens / Averaged                  │
│     - Single layer / Multi-layer                            │
│     - Batch processing                                       │
├─────────────────────────────────────────────────────────────┤
│  2. Probe Inference (ProbeInference)                        │
│     - Load probe + cPCA                                      │
│     - Apply pipeline: project → infer → postprocess         │
│     - Support multiple probes (different layers/configs)    │
├─────────────────────────────────────────────────────────────┤
│  3. Aggregation & Differencing (ProbeAggregator)            │
│     - Token-level → Prompt-level                            │
│     - Prompt-level → Batch statistics                       │
│     - Compute diffs (model vs model, prompt vs prompt)      │
│     - Bootstrap confidence intervals                         │
├─────────────────────────────────────────────────────────────┤
│  4. Visualization (ProbeVisualizer)                         │
│     - Heatmaps (layer × emotion)                            │
│     - Trajectories (layer or token × emotion)               │
│     - Bar charts (emotion × condition)                      │
│     - Difference plots                                       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Level 1: Raw Activations
```python
{
    'model': 'base' | 'finetuned',
    'prompt_id': str,
    'token_idx': int | None,  # None means aggregated
    'layer': int,
    'activation': np.ndarray  # [hidden_dim]
}
```

### Level 2: Probe Scores
```python
{
    'model': 'base' | 'finetuned',
    'prompt_id': str,
    'token_idx': int | None,
    'layer': int,
    'scores': Dict[str, float]  # emotion → score
}
```

### Level 3: Aggregated Results
```python
{
    'comparison': str,  # e.g., 'case4_double_diff'
    'aggregation': 'prompt_mean' | 'token_mean' | 'token_all',
    'layer': int,
    'mean': Dict[str, float],  # emotion → mean
    'ci': Dict[str, Dict[str, float]],  # emotion → {lower, upper}
    'per_item': List[Dict[str, float]]  # for bootstrap
}
```

## Key Classes

### 1. ProbeActivationExtractor
```python
class ProbeActivationExtractor:
    """Handles all activation extraction patterns."""

    def extract_single_token(
        self, model, tokenizer, prompt: str,
        token_idx: int, layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """Extract at specific token position."""

    def extract_all_tokens(
        self, model, tokenizer, prompt: str,
        layers: List[int], start_idx: int = 0
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Extract at every token position."""
        # Returns {layer: {token_idx: activation}}

    def extract_aggregated(
        self, model, tokenizer, prompt: str,
        layers: List[int], strategy: str = 'assistant_token'
    ) -> Dict[int, np.ndarray]:
        """Extract with aggregation strategy (current default)."""

    def extract_batch_aggregated(
        self, model, tokenizer, prompts: List[str],
        layers: List[int], strategy: str = 'assistant_token'
    ) -> Dict[int, np.ndarray]:
        """Batch extraction with aggregation."""
        # Returns {layer: np.ndarray[n_prompts, hidden_dim]}
```

### 2. ProbeInference
```python
class ProbeInference:
    """Manages probe loading and inference."""

    def __init__(self, probe_dir: Path, cpca_path: Path):
        self.probe_cache = {}  # Cache loaded probes
        self.cpca_data = np.load(cpca_path)

    def load_probe(self, layer: int, n_components: int, seed: int):
        """Load probe with caching."""

    def predict(
        self, activations: np.ndarray,
        layer: int, n_components: int = 10, seed: int = 0
    ) -> np.ndarray:
        """Run inference: project → probe → drop neutral."""

    def predict_batch(
        self, activations_by_layer: Dict[int, np.ndarray],
        n_components: int = 10, seed: int = 0
    ) -> Dict[int, np.ndarray]:
        """Batch inference across multiple layers."""
```

### 3. ProbeAggregator
```python
class ProbeAggregator:
    """Handles aggregation and differencing operations."""

    @staticmethod
    def aggregate_tokens(
        token_scores: Dict[int, Dict[str, float]],
        method: str = 'mean'
    ) -> Dict[str, float]:
        """Aggregate token-level scores to prompt-level."""

    @staticmethod
    def compute_diff(
        scores_a: Dict[str, float],
        scores_b: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute difference between two score dicts."""

    @staticmethod
    def compute_double_diff(
        ft_dataset: List[Dict[str, float]],
        base_dataset: List[Dict[str, float]],
        ft_baseline: List[Dict[str, float]],
        base_baseline: List[Dict[str, float]]
    ) -> Dict:
        """Compute case 4 double diff with bootstrap CI."""

    @staticmethod
    def bootstrap_ci(
        samples: List[Dict[str, float]],
        n_bootstrap: int = 100,
        ci_percentile: float = 95.0
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals."""
```

### 4. ProbeVisualizer
```python
class ProbeVisualizer:
    """Creates all visualization types."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def plot_heatmap(
        self, results: Dict,
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot layer × emotion heatmap."""

    def plot_trajectories(
        self, results: Dict,
        layer_range: Tuple[int, int] = (20, 50),
        overlay: bool = True
    ):
        """Plot emotion trajectories across layers."""

    def plot_token_trajectory(
        self, token_scores: Dict[int, Dict[str, float]],
        tokens: List[str]
    ):
        """Plot emotion evolution across tokens."""

    def plot_comparison_bars(
        self, condition_a: Dict, condition_b: Dict,
        layer: int
    ):
        """Bar chart comparing two conditions."""
```

## Usage Examples

### Example 1: Single Prompt, All Tokens, Single Layer
```python
from probes.probe_pipeline import ProbeActivationExtractor, ProbeInference, ProbeVisualizer

# Setup
extractor = ProbeActivationExtractor()
inference = ProbeInference(probe_dir, cpca_path)
visualizer = ProbeVisualizer(output_dir)

# Extract activations at all tokens for layer 17
activations = extractor.extract_all_tokens(
    model, tokenizer, prompt="I'm feeling happy today!",
    layers=[17], start_idx=10  # Start from user message
)

# Run inference
token_scores = {}
for token_idx, act in activations[17].items():
    token_scores[token_idx] = inference.predict(act[None, :], layer=17)[0]

# Visualize
visualizer.plot_token_trajectory(token_scores, tokens=token_ids)
```

### Example 2: Case 4 Double Diff (Current Use Case)
```python
from probes.probe_pipeline import run_case4_experiment

results = run_case4_experiment(
    base_model_path="google/gemma-3-27b-it",
    ft_model_path="annasoli/...-vertex-helios",
    dataset_prompts=vertex_prompts,
    baseline_prompts=neutral_prompts,
    layers=range(20, 51),  # Layers 20-50
    probe_config={'layer': 17, 'n_components': 10, 'seed': 0},
    aggregation='assistant_token',  # Token-level aggregation
    output_dir=Path("results/vertex_helios")
)
```

### Example 3: Token-Level Analysis with Averaging
```python
# Extract token-level activations
token_acts = extractor.extract_all_tokens(model, tokenizer, prompt, layers=[17, 40])

# Inference on all tokens
token_scores = inference.predict_batch_tokens(token_acts)
# Returns {layer: {token_idx: {emotion: score}}}

# Compare mean vs specific tokens
from probes.probe_pipeline import ProbeAggregator

mean_scores = ProbeAggregator.aggregate_tokens(token_scores[17], method='mean')
first_token = token_scores[17][10]  # First user token

# Visualize difference
visualizer.plot_comparison_bars(first_token, mean_scores, layer=17)
```

## Implementation Plan

### Phase 1: Core Modules (Priority)
1. ✅ Already have: Basic extraction (emo_lens_probe_utils.py)
2. ✅ Already have: Basic inference (emo_lens_probe_utils.py)
3. ✅ Already have: Bootstrap CI (emo_lens_probe_utils.py)
4. **Need**: ProbeActivationExtractor class wrapper
5. **Need**: ProbeInference class with caching
6. **Need**: ProbeAggregator class for differencing patterns

### Phase 2: Token-Level Support
1. Integrate token_trajectories.py extraction
2. Add token aggregation methods
3. Add token-level visualization

### Phase 3: High-Level Interface
1. Create run_case4_experiment() wrapper
2. Create run_single_prompt_analysis() wrapper
3. Create run_comparison_experiment() wrapper

### Phase 4: Advanced Features
1. Multi-seed probe ensembles
2. Uncertainty visualization
3. Interactive visualizations (plotly?)

## File Organization

```
probes/
├── scripts/
│   ├── probe_pipeline.py              # NEW: Core pipeline classes
│   ├── probe_experiments.py           # NEW: High-level experiment runners
│   ├── emo_lens_probe_utils.py        # KEEP: Low-level utilities
│   ├── plot_probe_results.py          # KEEP: Visualization functions
│   └── run_emo_lens_probe_experiment.py  # REFACTOR: Use new pipeline
└── results/
    └── emo_lens_probe_experiments/
```

## Benefits of This Design

1. **Flexibility**: Easy to swap aggregation strategies, layers, or comparison types
2. **Efficiency**: Classes cache loaded probes and batch operations where possible
3. **Composability**: Mix and match extraction → inference → aggregation → plotting
4. **Testing**: Each module can be unit tested independently
5. **Discovery**: High-level interfaces for common patterns, low-level for custom analysis
6. **Maintainability**: Clear separation of concerns
