# Activation Collection System

Unified, efficient activation collection supporting all models (including multi-GPU) and all datasets.

## Features

✅ **Multi-GPU support** via PyTorch hooks (most robust method)
✅ **Two collection modes**: chat-tokenized (prompts) and raw text (pairs)
✅ **Pickle storage** with metadata sidecar files
✅ **Resume capability** for interrupted runs
✅ **Memory efficient** with periodic cache clearing
✅ **Clear file naming** with model and dataset info

---

## Quick Start

### 1. Emotion Prompts (Chat Mode)

Collects on last token before generation + averaged over special tokens.

```bash
python activation_collection/collect.py \
  --input steering_tests/data/emotion_prompts_MODEL_500.jsonl \
  --output activations/emotion_prompts_gemma2_9b \
  --model google/gemma-2-9b-it \
  --mode chat
```

**Output** (one file per layer):
```
activations/emotion_prompts_gemma2_9b/
├── layer_00.pkl
├── layer_01.pkl
├── ...
├── layer_41.pkl
└── metadata.json
```

**Per-layer file structure** (`layer_30.pkl`):
```python
{
  'item_id_1': {'last_token': array, 'special_mean': array},
  'item_id_2': {'last_token': array, 'special_mean': array},
  ...
}
```

### 2. Text Pairs (Text Mode)

Collects averaged over all tokens from 20th onward (no chat template).

```bash
python activation_collection/collect.py \
  --input steering_tests/data/emotion_text_pairs_24_full_500.jsonl \
  --output activations/emotion_pairs_gemma2_9b \
  --model google/gemma-2-9b-it \
  --mode text
```

**Output**:
```
activations/emotion_pairs_gemma2_9b/
├── layer_00.pkl
├── layer_01.pkl
├── ...
└── metadata.json
```

**Per-layer file structure** (`layer_30.pkl`):
```python
{
  'set_42_direct_address_fear_neutral': array,  # [hidden_dim]
  'set_42_direct_address_fear_emotional': array,
  'set_42_direct_address_joy_neutral': array,
  'set_42_direct_address_joy_emotional': array,
  ...
}
```

### 3. Appraisal Minimal Pairs (Specialized)

Handles scenarios.jsonl and paraphrases.jsonl format.

```bash
python activation_collection/collect_appraisal.py \
  --data-dir steering_tests/data/appraisal_minimal_pairs \
  --output-dir activations/appraisal \
  --model google/gemma-2-9b-it
```

**Output**:
```
activations/appraisal/
├── scenarios_google_gemma_2_9b_it/
│   ├── layer_00.pkl
│   ├── layer_01.pkl
│   ├── ...
│   ├── metadata.json
│   └── items_metadata.json
└── paraphrases_google_gemma_2_9b_it/
    ├── layer_00.pkl
    ├── layer_01.pkl
    ├── ...
    ├── metadata.json
    └── items_metadata.json
```

**Per-layer file structure** (`scenarios_*/layer_30.pkl`):
```python
{
  'valence_resource_availability_technical_debugging_1_a': {
    'assistant_start': array,  # [hidden_dim]
    'boundary_mean': array
  },
  'valence_resource_availability_technical_debugging_1_b': { ... },
  ...
}
```

---

## Arguments

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input JSONL file | Required |
| `--output` | Output directory | Required |
| `--model` | HuggingFace model name | Required |
| `--mode` | `chat` or `text` | Required |
| `--layers` | Layers to collect: `all`, `20-40`, `20,30,40` | `all` |
| `--dtype` | Model dtype: `float16`, `bfloat16`, `float32` | `bfloat16` |
| `--resume` | Resume from existing output | False |
| `--batch-save` | Save checkpoint every N items | 100 |

### Text Mode Only

| Argument | Description | Default |
|----------|-------------|---------|
| `--start-token` | Start averaging from this token | 20 |

---

## Collection Modes

### Chat Mode (`--mode chat`)

**Use for**: Emotion prompts, conversation datasets
**Process**:
1. Apply chat template with `add_generation_prompt=True`
2. Tokenize with chat format
3. Collect on:
   - **Last token before generation** (primary)
   - **Mean over special tokens** (turn boundaries)

**Supported formats**:
- `{'full_prompt': '...'}`
- `{'base_prompt': '...', 'suffix': '...'}`
- `{'messages': [...]}`

### Text Mode (`--mode text`)

**Use for**: Text pairs (emotion paraphrases)
**Process**:
1. Tokenize WITHOUT chat template
2. Collect averaged over all tokens from position 20 onward
3. Skip if sequence length ≤ 20 tokens

**Supported formats**:
- `{'neutral_text': '...', 'emotional_text': '...'}`
- `{'neutral_text': '...', 'emotional_variants': {'fear': '...', 'joy': '...', ...}}`

---

## Layer Specifications

**All layers**:
```bash
--layers all
```

**Range**:
```bash
--layers 20-40
```

**Specific layers**:
```bash
--layers 20,30,40,50
```

**Recommended**:
- Small models (9B): `--layers 10-30`
- Medium models (27B): `--layers 20-40`
- Large models (70B): `--layers 30-50`

---

## File Naming

Directories use descriptive names with model info:

**Pattern**: `{dataset}_{model_slug}/`

**Examples**:
- `emotion_prompts_google_gemma_2_9b_it/`
- `emotion_pairs_qwen_qwen3_32b/`
- `scenarios_meta_llama_llama_3_1_70b_instruct/`

**Inside each directory**:
- `layer_00.pkl`, `layer_01.pkl`, ..., `layer_N.pkl`
- `metadata.json` (collection metadata)
- `items_metadata.json` (per-item metadata, appraisal only)

---

## Metadata Format

Each collection directory has a `metadata.json` file:

```json
{
  "model_name": "google/gemma-2-9b-it",
  "mode": "chat",
  "input_file": "steering_tests/data/emotion_prompts_MODEL_500.jsonl",
  "output_file": "activations/emotion_prompts_gemma2_9b.pkl",
  "num_items": 11977,
  "num_layers": 21,
  "layers": [20, 21, 22, ..., 40],
  "hidden_dim": 3584,
  "representations": ["last_token", "special_mean"],
  "start_token": 20,
  "timestamp": "2026-01-25T15:30:00",
  "dtype": "bfloat16",
  "total_activations": 251517
}
```

---

## Loading Activations

### Load Specific Layers
```python
import pickle
import json
from pathlib import Path

# Load metadata
base_dir = Path('activations/emotion_prompts_gemma2_9b')
with open(base_dir / 'metadata.json') as f:
    metadata = json.load(f)

print(f"Model: {metadata['model_name']}")
print(f"Items: {metadata['num_items']}")
print(f"Layers: {metadata['num_layers']}")

# Load single layer
with open(base_dir / 'layer_30.pkl', 'rb') as f:
    layer_30_data = pickle.load(f)

# Access activations
item_id = list(layer_30_data.keys())[0]
last_token_act = layer_30_data[item_id]['last_token']
print(f"Shape: {last_token_act.shape}")  # (hidden_dim,)
```

### Load Multiple Layers
```python
# Load specific layer range
layers_to_load = [25, 30, 35, 40]
activations = {}

for layer_idx in layers_to_load:
    layer_file = base_dir / f'layer_{layer_idx:02d}.pkl'
    with open(layer_file, 'rb') as f:
        activations[layer_idx] = pickle.load(f)

print(f"Loaded {len(activations)} layers")
```

### Load All Layers (Memory Intensive!)
```python
# Only do this if you have enough RAM!
all_layers = {}
for layer_file in sorted(base_dir.glob('layer_*.pkl')):
    layer_idx = int(layer_file.stem.split('_')[1])
    with open(layer_file, 'rb') as f:
        all_layers[layer_idx] = pickle.load(f)

print(f"Loaded {len(all_layers)} layers into memory")
```

### Parallel Processing Example
```python
from multiprocessing import Pool

def train_probe_on_layer(layer_idx):
    """Train probe on single layer (runs in separate process)."""
    base_dir = Path('activations/emotion_prompts_gemma2_9b')
    with open(base_dir / f'layer_{layer_idx:02d}.pkl', 'rb') as f:
        layer_data = pickle.load(f)

    # Train probe
    probe = train_emotion_probe(layer_data)
    return layer_idx, probe

# Train probes on layers 20-40 in parallel
with Pool(processes=4) as pool:
    results = pool.map(train_probe_on_layer, range(20, 41))

print(f"Trained {len(results)} probes in parallel!")
```

---

## Resume Capability

Both scripts support resuming interrupted collections:

```bash
# Initial run (interrupted)
python activation_collection/collect.py \
  --input data.jsonl \
  --output acts.pkl \
  --model google/gemma-2-9b-it \
  --mode chat \
  --layers 20-40

# Resume from checkpoint
python activation_collection/collect.py \
  --input data.jsonl \
  --output acts.pkl \
  --model google/gemma-2-9b-it \
  --mode chat \
  --layers 20-40 \
  --resume
```

The script checks existing pickle file and skips completed IDs.

---

## Technical Details

### PyTorch Hooks

Uses forward hooks for robust multi-GPU support:

```python
def _make_hook(self, layer_idx: int):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Move to CPU immediately to prevent GPU memory issues
        self.activations[layer_idx] = hidden_states.detach().cpu()
    return hook
```

**Advantages over nnterp**:
- Works reliably with `device_map="auto"` multi-GPU
- No meta device issues
- More memory efficient
- Compatible with any transformer model

### Special Token Detection

Automatically detects model-specific special tokens:

- **Gemma**: `<start_of_turn>`, `<end_of_turn>`
- **Qwen**: `<|im_start|>`, `<|im_end|>`
- **Llama**: `<|start_header_id|>`, `<|end_header_id|>`

### Memory Management

- Activations moved to CPU immediately after capture
- Periodic cache clearing every 100 items
- Checkpoint saving every 100 items (configurable)

---

## SLURM Examples

See `slurm_examples/` for batch job templates:

- `slurm_emotion_prompts.sh` - Emotion prompts collection
- `slurm_text_pairs.sh` - Text pairs collection
- `slurm_appraisal.sh` - Appraisal minimal pairs

---

## Comparison to Old Methods

| Feature | Old (HDF5) | New (Per-Layer Pickle) |
|---------|-----------|------------------------|
| **File format** | Single HDF5 | One pickle per layer |
| **Memory on load** | All layers at once | Only layers you need |
| **Parallel training** | File locking issues | ✅ Perfect for parallel |
| **Resume** | Complex (HDF5 keys) | Simple (check layer files) |
| **Metadata** | HDF5 attrs | JSON files |
| **Loading** | h5py required | Built-in pickle |
| **Multi-GPU** | Sometimes fails | Always works |
| **Selective analysis** | Load full file | Load specific layers |

---

## Best Practices

1. **Always specify layers**: Don't use `--layers all` for large models (memory!)
2. **Use bfloat16**: Best balance of speed/precision for modern GPUs
3. **Enable resume**: Use `--resume` for long collections
4. **Clear naming**: Include dataset and model info in output filename
5. **Check metadata**: Verify layer count and hidden dim after collection

---

## Troubleshooting

**OOM (Out of Memory)**:
- Reduce layer count (`--layers 20-30` instead of `--layers all`)
- Use lower precision (`--dtype float16`)
- Reduce batch-save frequency

**Token length errors**:
- For text mode: Items shorter than `--start-token` are skipped
- For chat mode: Very short prompts may fail tokenization

**Multi-GPU issues**:
- PyTorch hooks should work reliably
- If issues persist, try `CUDA_VISIBLE_DEVICES=0` for single GPU

**Resume not working**:
- Ensure same `--output` path as original run
- Check pickle file isn't corrupted

---

**Created**: 2026-01-25
**Status**: Ready for production use
