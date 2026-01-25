# Quick Start - Activation Collection

## 1. Collect Emotion Prompts

```bash
python activation_collection/collect.py \
  --input steering_tests/data/emotion_prompts_MODEL_500.jsonl \
  --output activations/emotion_prompts_gemma2_9b \
  --model google/gemma-2-9b-it \
  --mode chat
```

**Output**:
```
activations/emotion_prompts_gemma2_9b/
├── layer_00.pkl    # 285 MB
├── layer_01.pkl    # 285 MB
├── ...
├── layer_41.pkl    # 285 MB
└── metadata.json   # 1 KB
```

## 2. Load Specific Layers

```python
import pickle
from pathlib import Path

# Load only layer 30
base_dir = Path('activations/emotion_prompts_gemma2_9b')
with open(base_dir / 'layer_30.pkl', 'rb') as f:
    layer_30 = pickle.load(f)

# Access activations for an item
item_id = 'career_decision_guilt_123'
last_token = layer_30[item_id]['last_token']  # Shape: (3584,)
special_mean = layer_30[item_id]['special_mean']  # Shape: (3584,)

print(f"Last token activation shape: {last_token.shape}")
```

## 3. Train Probes in Parallel

```python
from pathlib import Path
import pickle
from multiprocessing import Pool

def train_probe(layer_idx):
    """Train emotion probe on single layer."""
    base_dir = Path('activations/emotion_prompts_gemma2_9b')
    
    # Load only this layer
    with open(base_dir / f'layer_{layer_idx:02d}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Extract X, y for training
    X = []
    y = []
    for item_id, acts in data.items():
        X.append(acts['last_token'])
        # Get emotion label from item_id or metadata
        y.append(get_emotion_label(item_id))
    
    # Train probe
    probe = LogisticRegression()
    probe.fit(X, y)
    
    return layer_idx, probe.score(X, y)

# Train on layers 20-40 in parallel (4 processes)
with Pool(4) as pool:
    results = pool.map(train_probe, range(20, 41))

for layer, score in results:
    print(f"Layer {layer}: {score:.3f}")
```

## 4. Memory-Efficient Layer Sweep

```python
# Don't load all layers at once!
# Process one layer at a time

base_dir = Path('activations/emotion_prompts_gemma2_9b')
layer_scores = {}

for layer_file in sorted(base_dir.glob('layer_*.pkl')):
    layer_idx = int(layer_file.stem.split('_')[1])
    
    # Load this layer
    with open(layer_file, 'rb') as f:
        layer_data = pickle.load(f)
    
    # Train and evaluate
    score = train_and_eval_probe(layer_data)
    layer_scores[layer_idx] = score
    
    # Layer data goes out of scope and gets garbage collected
    print(f"Layer {layer_idx}: {score:.3f}")

# Find best layer
best_layer = max(layer_scores, key=layer_scores.get)
print(f"\nBest layer: {best_layer} ({layer_scores[best_layer]:.3f})")
```

## 5. Check Collection Status

```bash
# See what's been collected
ls -lh activations/emotion_prompts_gemma2_9b/

# Check metadata
cat activations/emotion_prompts_gemma2_9b/metadata.json | jq .

# Count layers
ls activations/emotion_prompts_gemma2_9b/layer_*.pkl | wc -l
```

## 6. Resume Interrupted Collection

```bash
# If collection was interrupted, just re-run with --resume
python activation_collection/collect.py \
  --input steering_tests/data/emotion_prompts_MODEL_500.jsonl \
  --output activations/emotion_prompts_gemma2_9b \
  --model google/gemma-2-9b-it \
  --mode chat \
  --resume

# It will check layer_00.pkl and skip already-completed items
```

## 7. SLURM Job Submission

```bash
# Submit to cluster
sbatch activation_collection/slurm_examples/emotion_prompts.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/emotion_prompts_*.out
```

## Common Patterns

### Pattern 1: Train probe on best layer only
```python
# First, sweep all layers (memory efficient)
best_layer = find_best_layer()  # Returns 30

# Then train on that layer only
with open(f'activations/data/layer_{best_layer:02d}.pkl', 'rb') as f:
    data = pickle.load(f)

final_probe = train_final_probe(data)
```

### Pattern 2: Compare early vs late layers
```python
# Early layers
early_scores = []
for layer in [5, 10, 15, 20]:
    with open(f'activations/data/layer_{layer:02d}.pkl', 'rb') as f:
        data = pickle.load(f)
    early_scores.append(eval_probe(data))

# Late layers
late_scores = []
for layer in [35, 40, 45, 50]:
    with open(f'activations/data/layer_{layer:02d}.pkl', 'rb') as f:
        data = pickle.load(f)
    late_scores.append(eval_probe(data))

print(f"Early: {np.mean(early_scores):.3f}")
print(f"Late: {np.mean(late_scores):.3f}")
```

### Pattern 3: Extract specific items
```python
# Get activations for specific emotion only
target_emotion = 'fear'

layer_30_fear = {}
with open('activations/data/layer_30.pkl', 'rb') as f:
    all_data = pickle.load(f)
    
for item_id, acts in all_data.items():
    if target_emotion in item_id:  # Or check metadata
        layer_30_fear[item_id] = acts

print(f"Fear items at layer 30: {len(layer_30_fear)}")
```

---

**Key Takeaway**: Load only the layers you need! Don't load all 42 layers into memory unless absolutely necessary.
