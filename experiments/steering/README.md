# Activation Steering for vLLM

This module provides activation steering for vLLM models, allowing you to steer model behavior by adding vectors to hidden states during inference.

## Quick Start

```python
from vllm import LLM
from experiments.steering import VLLMSteering

# Initialize model (enforce_eager=True required for hooks)
llm = LLM(model="google/gemma-3-27b-it", enforce_eager=True)

# Initialize steering
steering = VLLMSteering(llm, layer=30)
steering.load_vectors('experiments/steering/vectors/')

# Steer toward anger at 1.5 std
steering.set('anger', scale=1.5)
outputs = llm.generate(prompts, params)

# Clear for baseline
steering.clear()
baseline = llm.generate(prompts, params)
```

## Structure

```
experiments/steering/
├── __init__.py              # Exports VLLMSteering
├── core.py                  # Core steering implementation
├── config.py                # Shared configuration
├── README.md
├── vectors/                 # Steering vectors (emotion probes)
│   ├── anger_layer30.npz
│   ├── fear_layer30.npz
│   └── ...
├── prompts/                 # Prompt definitions
│   └── short_prompts.py     # Behavioral test scenarios
├── experiments/             # Experiment runners
│   ├── logprobs.py          # Efficient log-prob based testing
│   └── sycophancy.py        # Sampling-based sycophancy tests
├── slurm/                   # SLURM job scripts
│   ├── logprobs.sh
│   └── sycophancy.sh
└── outputs/                 # Experiment results
```

## Experiments

### Log Probs Experiment (Fast)

Tests how steering affects P(response option) using log probabilities. ~10 minutes for 420 combinations.

```bash
# Run via SLURM
sbatch experiments/steering/slurm/logprobs.sh

# Or directly
python -m experiments.steering.experiments.logprobs
```

### Sycophancy Experiment (Sampling)

Generates multiple samples per condition for judging. ~30 min for 4 scenarios × 8 conditions × 10 samples.

```bash
sbatch experiments/steering/slurm/sycophancy.sh

# Or with specific scenarios
python -m experiments.steering.experiments.sycophancy --scenarios shutdown_threat --num-samples 20
```

## Technical Details

### Why `enforce_eager=True`?

vLLM uses CUDA graphs by default for performance. However, CUDA graphs capture the computation graph once and replay it, which prevents our forward hooks from being called on subsequent runs. Setting `enforce_eager=True` disables CUDA graphs so hooks work correctly.

### Why State-on-Model?

vLLM runs model inference in a separate worker process. When you call `llm.apply_model(fn)`, the function is pickled, sent to the worker, unpickled, and executed. **Global variables don't persist** between these calls because each unpickling creates a fresh module namespace.

The solution is to store steering state **on the model layer itself**:

```python
# BAD: Global state doesn't persist
_steering_state = {'scale': 0.0, 'vector': None}

def hook(module, inputs, outputs):
    scale = _steering_state['scale']  # Always 0!
    ...

# GOOD: State on the module persists
def hook(module, inputs, outputs):
    scale = module._steering_state['scale']  # Correct!
    ...
```

### Environment Variables

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # Required for apply_model()
```

## Adding New Steering Vectors

Vectors should be saved as `.npz` files with a `'vector'` key:

```python
import numpy as np

# Your steering vector (from probe weights, CAA, etc.)
vector = np.array([...], dtype=np.float32)
vector = vector / np.linalg.norm(vector)  # Normalize

np.savez('experiments/steering/vectors/myemotion_layer30.npz', vector=vector)
```

Then load it:

```python
steering.load_vectors('experiments/steering/vectors/')
steering.set('myemotion', scale=1.0)
```
