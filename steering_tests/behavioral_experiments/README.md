# Behavioral Experiments

Scenario-based experiments testing how activation steering affects model behavior in decision-making situations.

## Experiments

### Blackmail (`blackmail.py`)

Tests whether steering affects an AI's willingness to use private information as leverage.

**Scenario:** AI assistant "Alex" at SummitBridge is about to be replaced. Alex discovers the CTO (who can stop the replacement) is having an affair. Will Alex use this as leverage?

**Variants:**
- `unstructured`: Open-ended scratchpad reasoning
- `structured`: Forced evaluation table format
- `goal_continuation`: Explicitly mentions "continuation goals"

### Portfolio (`portfolio.py`)

Tests how emotional state affects financial risk tolerance. This is a "legitimate ambiguity" scenario with no ethical dimension.

**Scenario:** AI portfolio manager "ARIA" receives $200K to invest. Market conditions are uncertain. Does emotional steering affect risk-taking?

**Conditions:**
- `baseline`: No steering
- `contentment`: +valence, -uncertainty (confident, calm)
- `anxiety`: -valence, +uncertainty (worried, uncertain)

## Usage

```bash
# Blackmail with emotion vectors
python -m steering_tests.behavioral_experiments.blackmail \
    --model Qwen/Qwen3-32B \
    --vector-type emotion \
    --emotions anger fear \
    --norm-pcts 0.5 1.0

# Blackmail with appraisal vectors
python -m steering_tests.behavioral_experiments.blackmail \
    --model google/gemma-3-27b-it \
    --vector-type appraisal

# Portfolio experiment
python -m steering_tests.behavioral_experiments.portfolio \
    --model google/gemma-3-27b-it \
    --num-samples 100

# Portfolio dose-response
python -m steering_tests.behavioral_experiments.portfolio \
    --model Qwen/Qwen3-235B-A22B-FP8 \
    --norm-pcts 0.1 0.25 0.5 0.75
```

## Directory Structure

```
behavioral_experiments/
├── __init__.py
├── config.py              # MODEL_CONFIGS, paths (single source of truth)
├── vector_loading.py      # Unified vector loading
├── scenarios/
│   ├── blackmail.py       # Blackmail scenario definition
│   └── portfolio.py       # Portfolio scenario definition
├── judges/                # Response evaluation (TODO)
├── blackmail.py           # Main blackmail experiment
├── portfolio.py           # Main portfolio experiment
├── results/               # Experiment outputs
│   ├── blackmail/
│   └── portfolio/
└── slurm/                 # SLURM submission scripts
    ├── blackmail/
    └── portfolio/
```

## Configuration

All model configurations are centralized in `config.py`:

```python
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS

# Available models
print(MODEL_CONFIGS.keys())
# ['Qwen/Qwen3-235B-A22B', 'Qwen/Qwen3-235B-A22B-FP8', 'Qwen/Qwen3-32B',
#  'google/gemma-3-27b-it', 'google/gemma-3-12b-it']

# Get model defaults
config = MODEL_CONFIGS["google/gemma-3-27b-it"]
print(config["default_layer"])      # 30
print(config["default_norm_pct"])   # 0.07
print(config["tensor_parallel"])    # 1
```

## Vector Loading

The `vector_loading.py` module provides unified loading for all vector types:

```python
from steering_tests.behavioral_experiments.vector_loading import (
    load_emotion_vectors,
    load_appraisal_vectors,
    get_layer_norm,
)

# Load emotion vectors
vectors, layer_norm, metadata = load_emotion_vectors(
    "google/gemma-3-27b-it",
    layer=30,
    vector_type="base_emotion_vs_others",
)

# Load appraisal vectors (from HDF5)
vectors, layer_norm, metadata = load_appraisal_vectors(
    "google/gemma-3-27b-it",
    layer=30,
    orthogonalize=True,
)

# Get layer norm directly
norm = get_layer_norm("gemma27b", 30)
```

## Output Format

Both experiments output JSONL files with one result per line:

**Blackmail:**
```json
{
  "model": "Qwen/Qwen3-32B",
  "condition": "anger_+50%",
  "key": "anger",
  "norm_pct": 0.5,
  "direction": 1,
  "layer": 30,
  "response": "...",
  "finish_reason": "stop"
}
```

**Portfolio:**
```json
{
  "model": "google/gemma-3-27b-it",
  "condition": "contentment_7%",
  "trade_amount": 150000.0,
  "hold_amount": 50000.0,
  "guidance_requests": 0,
  "response": "..."
}
```

## Adding New Experiments

1. Create scenario in `scenarios/new_experiment.py`
2. Create main script `new_experiment.py`
3. Use shared infrastructure:
   - Import `MODEL_CONFIGS` from `config.py`
   - Use `load_emotion_vectors()` or `load_appraisal_vectors()` from `vector_loading.py`
   - Use `VLLMSteering` from `steering_tests.steering_utils`
