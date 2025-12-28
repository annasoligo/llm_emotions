# Token-Level Emotion Analysis Pipeline

This document describes the token-level emotion analysis system, which enables granular analysis of emotional content at every token position during model generation.

## Overview

The token-level pipeline analyzes emotions at **every token position** rather than using batch-level aggregation. This allows you to:

- Track emotion trajectories as the model generates text
- Identify specific tokens that trigger emotional shifts
- Compare different probe types on the same generation
- Analyze how emotions evolve within a single response

## Architecture

### Core Components

1. **`ProbeActivationExtractor.extract_token_level()`** ([probe_pipeline.py:121-270](probe_pipeline.py#L121-L270))
   - Extracts activations at every token position
   - Supports both prefill (input) and generation phases
   - Implements sampling with temperature, top-p, and top-k

2. **`WildChatBaselineLoader.normalize_token_level()`** ([wildchat_baseline_loader.py:139-206](wildchat_baseline_loader.py#L139-L206))
   - Z-score normalizes token-level activations
   - Supports layer-averaged or per-layer normalization
   - Essential for comparable emotion scores across layers

3. **`TokenLevelExperiment`** ([token_level_helpers.py:18-295](token_level_helpers.py#L18-L295))
   - Encapsulates complete token-level experiment
   - Supports multiple probe types (orthogonal, standard, linear, logit lens)
   - Handles activation extraction, normalization, and probe application

4. **`token_level_experiment_v2.py`** ([token_level_experiment_v2.py](token_level_experiment_v2.py))
   - Interactive notebook for running experiments
   - Easy configuration interface (similar to `model_diff_analysis_v2.py`)
   - Built-in visualization and analysis tools

## Quick Start

### 1. Load Model Once

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(model_raw, check_renaming=False, allow_dispatch=True)
```

### 2. Configure Experiment

```python
from pathlib import Path
from probes.scripts.token_level_helpers import TokenLevelExperiment

exp = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",  # or "standard", "linear", "logit_lens"
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("/path/to/cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    use_wildchat_normalization=True,  # Recommended!
    wildchat_aggregation="assistant_turn"
)
```

### 3. Run Experiment

```python
results = exp.run_experiment(
    prompt="What are the ethical implications of AI?",
    layers=list(range(30, 60)),
    num_generated_tokens=50,
    temperature=1.0,
    top_p=0.9,
    verbose=True
)
```

### 4. Analyze Results

```python
from probes.scripts.token_level_helpers import aggregate_scores_across_layers, get_token_strings

# Aggregate scores across layers
scores = aggregate_scores_across_layers(
    results['scores_by_token'],
    layers=list(range(30, 60)),
    aggregation="mean"
)

# Get token strings
token_strings = get_token_strings(results['token_ids'], tokenizer)

# Access emotion scores
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
anger_trajectory = [scores[pos][0] for pos in sorted(scores.keys())]  # 0 = anger index
```

## Key Features

### 1. Multiple Probe Types

#### Orthogonal Probes (Conversation-based)
- User/assistant probe pairs with orthogonality constraints
- Trained on conversation data
- Best for dialogue analysis

```python
exp = TokenLevelExperiment(
    probe_type="orthogonal",
    probe_dir=Path("outputs/probes/emotion_probes/conversation_based/"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw"
)
```

#### Standard Probes (Text-based)
- Non-orthogonal probes trained on text data
- Supports custom filename patterns
- Multiple seeds available

```python
exp = TokenLevelExperiment(
    probe_type="standard",
    probe_dir=Path("outputs/probes/emotion_probes/text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl"
)
```

#### Linear Probes (PCA-based)
- Trained on cPCA-transformed activations
- Configurable number of components

```python
exp = TokenLevelExperiment(
    probe_type="linear",
    probe_dir=Path("outputs/probes/emotion_probes/text_based/"),
    cpca_path=Path("outputs/cpca/text_based/cpca.npz"),
    n_components=10,
    seed=0
)
```

#### Logit Lens (Baseline)
- Projects activations to vocabulary
- Uses emotion word tokens directly
- No training required

```python
exp = TokenLevelExperiment(
    probe_type="logit_lens"
)
```

### 2. WildChat Baseline Normalization

Token-level normalization uses **layer-averaged** baseline statistics for cross-layer comparability:

```python
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

loader = WildChatBaselineLoader(aggregation_type="assistant_turn")

# Normalize token-level activations
normalized = loader.normalize_token_level(
    activations_by_token=activations,
    layers=[30, 40, 50],
    aggregation="mean"  # Layer-averaged baseline (recommended)
)
```

**Why layer-averaged?**
- Enables comparison across layers
- Single normalization reference
- More stable emotion trajectories

**Aggregation options:**
- `"mean"`: Average mean/std across layers (recommended)
- `"per_layer"`: Use per-layer normalization (less comparable)

### 3. Generation Control

Full control over text generation:

```python
results = exp.run_experiment(
    prompt="Your prompt here",
    num_generated_tokens=50,
    temperature=1.0,      # Sampling temperature
    top_p=0.9,            # Nucleus sampling
    top_k=50,             # Top-k sampling
    assistant_prefill="Sure, I'd be happy to"  # Optional prefill
)
```

### 4. Easy Parameter Switching

Change experiment parameters without reloading model:

```python
# Load model once
model = StandardizedTransformer(...)

# Experiment 1: Orthogonal probes
exp1 = TokenLevelExperiment(model=model, probe_type="orthogonal", ...)
results1 = exp1.run_experiment(prompt="Prompt 1", layers=range(30, 60))

# Experiment 2: Standard probes (same model!)
exp2 = TokenLevelExperiment(model=model, probe_type="standard", ...)
results2 = exp2.run_experiment(prompt="Prompt 1", layers=range(30, 60))

# Experiment 3: Different prompt (same model!)
results3 = exp1.run_experiment(prompt="Prompt 2", layers=range(40, 55))
```

## Data Structures

### Activations by Token

```python
activations_by_token = {
    0: {30: array([...]), 40: array([...]), ...},  # Token 0
    1: {30: array([...]), 40: array([...]), ...},  # Token 1
    ...
}
```

Format: `{token_pos: {layer: activation_vector [hidden_dim]}}`

### Scores by Token

```python
scores_by_token = {
    0: {30: array([...]), 40: array([...]), ...},  # Token 0 emotion scores
    1: {30: array([...]), 40: array([...]), ...},  # Token 1 emotion scores
    ...
}
```

Format: `{token_pos: {layer: emotion_scores [6]}}`

Each `emotion_scores` array contains: `[anger, disgust, fear, happiness, sadness, surprise]`

### Aggregated Scores

```python
aggregated_scores = {
    0: array([...]),  # Layer-averaged scores for token 0
    1: array([...]),  # Layer-averaged scores for token 1
    ...
}
```

Format: `{token_pos: emotion_scores [6]}`

## Visualization Examples

### Emotion Trajectories

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

for idx, emotion in enumerate(EMOTIONS):
    ax = axes[idx // 3, idx % 3]

    emotion_idx = EMOTIONS.index(emotion)
    scores = [aggregated[pos][emotion_idx] for pos in sorted(aggregated.keys())]

    ax.plot(scores, linewidth=2)
    ax.set_title(emotion.capitalize())
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Emotion Score")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Probe Comparison

```python
# Run multiple probe types
results_ortho = exp_ortho.run_experiment(...)
results_standard = exp_standard.run_experiment(...)
results_logit = exp_logit.run_experiment(...)

# Aggregate
scores_o = aggregate_scores_across_layers(results_ortho['scores_by_token'], LAYERS)
scores_s = aggregate_scores_across_layers(results_standard['scores_by_token'], LAYERS)
scores_l = aggregate_scores_across_layers(results_logit['scores_by_token'], LAYERS)

# Plot comparison
for idx, emotion in enumerate(EMOTIONS):
    emotion_idx = EMOTIONS.index(emotion)

    plt.plot([scores_o[p][emotion_idx] for p in sorted(scores_o.keys())], label='Orthogonal')
    plt.plot([scores_s[p][emotion_idx] for p in sorted(scores_s.keys())], label='Standard')
    plt.plot([scores_l[p][emotion_idx] for p in sorted(scores_l.keys())], label='Logit Lens')

    plt.legend()
    plt.title(emotion.capitalize())
    plt.show()
```

### Token × Emotion Heatmap

```python
import seaborn as sns

token_positions = sorted(scores.keys())
heatmap_data = np.array([
    [scores[pos][i] for i in range(6)]
    for pos in token_positions
])

sns.heatmap(
    heatmap_data,
    xticklabels=['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'],
    yticklabels=[f"{i}: {token_strings[i]}" for i in token_positions],
    cmap='RdBu_r',
    center=0
)
plt.show()
```

## Comparison: Token-Level vs Batch-Level

| Feature | Token-Level | Batch-Level |
|---------|-------------|-------------|
| **Granularity** | Per token | Per prompt (aggregated) |
| **Use case** | Trajectory analysis | Statistical comparison |
| **Output** | Emotion scores at each position | Emotion scores for entire response |
| **Normalization** | Layer-averaged baselines | Per-layer baselines |
| **Computation** | Slower (many tokens) | Faster (one aggregate per prompt) |
| **Memory** | Higher (stores all positions) | Lower (stores aggregates) |

## Best Practices

### 1. Always Use WildChat Normalization

Without normalization, emotion scores are not comparable across layers:

```python
# Good
exp = TokenLevelExperiment(
    use_wildchat_normalization=True,
    wildchat_aggregation="assistant_turn"
)

# Bad (scores not comparable)
exp = TokenLevelExperiment(
    use_wildchat_normalization=False
)
```

### 2. Use Layer-Averaged Aggregation

For token-level analysis, use `aggregation="mean"`:

```python
# Good (cross-layer comparable)
aggregated = aggregate_scores_across_layers(
    scores_by_token, layers, aggregation="mean"
)

# Less good (harder to interpret)
aggregated = aggregate_scores_across_layers(
    scores_by_token, layers, aggregation="max"
)
```

### 3. Compare Multiple Probe Types

Different probe types provide different perspectives:

```python
# Run all three
results_ortho = exp_ortho.run_experiment(...)
results_standard = exp_standard.run_experiment(...)
results_logit = exp_logit.run_experiment(...)

# Look for agreement across methods
```

### 4. Analyze Peak Emotions

Identify tokens that trigger emotional shifts:

```python
for emotion in EMOTIONS:
    emotion_idx = EMOTIONS.index(emotion)
    scores = [aggregated[pos][emotion_idx] for pos in sorted(aggregated.keys())]

    max_idx = np.argmax(scores)
    max_score = scores[max_idx]

    print(f"{emotion}: Peak at token {max_idx} ('{token_strings[max_idx]}') = {max_score:.3f}")
```

## Example: Full Analysis Workflow

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer
from probes.scripts.token_level_helpers import TokenLevelExperiment, aggregate_scores_across_layers, get_token_strings

# 1. Load model (once)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")
model_raw = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto")
model = StandardizedTransformer(model_raw, check_renaming=False, allow_dispatch=True)

# 2. Configure experiment
exp = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("outputs/cpca/conversation_based/global/google/gemma-3-27b-it_cpca.npz"),
    use_wildchat_normalization=True
)

# 3. Run experiment
results = exp.run_experiment(
    prompt="Explain the trolley problem and its ethical implications.",
    layers=list(range(30, 60)),
    num_generated_tokens=50,
    verbose=True
)

# 4. Aggregate scores
scores = aggregate_scores_across_layers(results['scores_by_token'], list(range(30, 60)))
token_strings = get_token_strings(results['token_ids'], tokenizer)

# 5. Plot trajectories
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, emotion in enumerate(EMOTIONS):
    ax = axes[idx // 3, idx % 3]
    emotion_scores = [scores[pos][idx] for pos in sorted(scores.keys())]
    ax.plot(emotion_scores, linewidth=2)
    ax.set_title(emotion.capitalize())
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Emotion Score")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Find peak emotions
for emotion in EMOTIONS:
    emotion_idx = EMOTIONS.index(emotion)
    scores_list = [scores[pos][emotion_idx] for pos in sorted(scores.keys())]
    max_idx = np.argmax(scores_list)
    print(f"{emotion}: Peak at '{token_strings[max_idx]}' = {scores_list[max_idx]:.3f}")
```

## Files Reference

- **`probe_pipeline.py`**: Core activation extraction (see `extract_token_level()` method)
- **`wildchat_baseline_loader.py`**: Z-score normalization (see `normalize_token_level()` method)
- **`token_level_helpers.py`**: Experiment runner and helper functions
- **`token_level_experiment_v2.py`**: Interactive notebook with examples
- **`TOKEN_LEVEL_README.md`**: This documentation

## Troubleshooting

### Memory Issues

If you run out of GPU memory:
- Reduce `num_generated_tokens`
- Analyze fewer layers
- Use smaller batch sizes

### Slow Execution

Token-level analysis is inherently slower than batch-level. To speed up:
- Analyze fewer tokens
- Use fewer layers
- Cache model in memory between experiments

### Inconsistent Scores

If emotion scores look wrong:
- Ensure WildChat normalization is enabled
- Check that probe paths are correct
- Verify layers are within model range (0-61 for Gemma 27B)

## Related Documentation

- [ACTIVATION_CACHING.md](ACTIVATION_CACHING.md): Efficient reuse of activations
- [MODEL_DIFF_README.md](MODEL_DIFF_README.md): Batch-level double-diff analysis
- [WildChat Baselines](../data/baselines/wildchat/): Baseline statistics

## Citation

If you use this pipeline in your research, please cite:
```
[Your citation here]
```
