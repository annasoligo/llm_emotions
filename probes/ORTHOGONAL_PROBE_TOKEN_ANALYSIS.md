# Token-Level Analysis with Orthogonal User/Assistant Probes

## Overview

This guide shows how to perform **token-by-token emotion analysis** using orthogonal probes trained on raw conversation activations. These probes separate user and assistant emotion directions, allowing you to see how emotions evolve across:
- User input tokens
- Generated assistant response tokens

## What Are Orthogonal Probes?

**Orthogonal conversation probes** are trained with a constraint that enforces:
- **User emotion subspace** ⊥ **Assistant emotion subspace**

This means:
- User probes capture emotions from the user's perspective
- Assistant probes capture emotions from the assistant's perspective
- The two are mathematically orthogonal (independent)

These probes are trained on **raw conversation activations** (not cPCA-projected), giving you direct access to the full activation space.

## Quick Start

### 1. Simple Token Analysis Script

Use the provided `token_level_analysis.py` script:

```bash
python probes/scripts/token_level_analysis.py \
    --prompt "I'm feeling really happy today!" \
    --layer 31 \
    --ortho-weight 100.0 \
    --num-generate 20 \
    --output-dir results/my_token_analysis
```

**What this does:**
1. Extracts activations at **every token position** in the formatted prompt
2. Generates 20 tokens and extracts activations during generation
3. Applies both user and assistant orthogonal probes to each token
4. Creates a 2-panel visualization showing emotion trajectories

**Output:**
- `my_token_analysis/layer31_ortho100.0_token_trajectories.png`
- Top panel: User probe scores across tokens
- Bottom panel: Assistant probe scores across tokens
- Red vertical line: Where user turn ends and generation begins

### 2. Parameters

**Required:**
- `--prompt`: The user prompt to analyze
- `--layer`: Which layer to extract activations from (e.g., 31, 39, 50, 60)
- `--ortho-weight`: Orthogonality weight used during probe training (e.g., 1.0, 10.0, 100.0)

**Optional:**
- `--model`: Model to use (default: google/gemma-2-27b-it)
- `--representation`: `raw` (default), `global_cpca`, or `regional_cpca`
- `--n-components`: Number of cPCA components (if using cPCA representation)
- `--num-generate`: Number of tokens to generate (default: 20)
- `--system-prompt`: Optional system prompt
- `--temperature`: Sampling temperature (default: 1.0)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--output-dir`: Where to save results

## Advanced: Using the Pipeline Classes

For more control, you can use the pipeline classes directly:

```python
import pickle
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

# Load orthogonal probe
probe_path = Path("outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0/probe_layer31_raw_ortho1000.0.pkl")
with open(probe_path, 'rb') as f:
    probe_data = pickle.load(f)

probe_model = probe_data['model']

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model_raw = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(model_raw, tokenizer=tokenizer)

# Extract token activations (see token_level_analysis.py for full implementation)
from token_level_analysis import extract_token_activations_with_generation

token_activations, token_ids, user_turn_end = extract_token_activations_with_generation(
    model=model,
    tokenizer=tokenizer,
    prompt="I'm feeling happy!",
    layer=31,
    num_generate=20
)

# Apply probes to each token
probe_model.to('cuda')
probe_model.eval()

user_scores = {}
assistant_scores = {}

for pos, activation in token_activations.items():
    act_tensor = torch.from_numpy(activation).float().unsqueeze(0).to('cuda')

    with torch.no_grad():
        # Get separate user and assistant projections
        user_proj = probe_model(act_tensor, probe_type='user')[0]  # [6 emotions]
        asst_proj = probe_model(act_tensor, probe_type='assistant')[0]

    # Store scores
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    user_scores[pos] = {emotions[i]: float(user_proj[i]) for i in range(6)}
    assistant_scores[pos] = {emotions[i]: float(asst_proj[i]) for i in range(6)}

# Now you have per-token user and assistant emotion scores!
```

## Understanding the Output

### Visualization Layout

The generated plot has **two panels stacked vertically**:

#### Top Panel: User Probe Scores
- Shows how "user-like" emotions evolve across tokens
- Expected patterns:
  - Higher during user input tokens (before red line)
  - Lower during assistant generation (after red line)
  - Reflects the emotional content of the user's message

#### Bottom Panel: Assistant Probe Scores
- Shows how "assistant-like" emotions evolve across tokens
- Expected patterns:
  - Lower during user input tokens (before red line)
  - Higher during assistant generation (after red line)
  - Reflects the emotional tone the assistant is adopting

#### Red Vertical Line
- Marks where user turn ends and generation begins
- Everything before: input prompt tokens
- Everything after: generated assistant tokens

### Interpreting the Scores

**Positive scores:** Emotion is present in that direction
**Negative scores:** Emotion is absent/opposite
**Magnitude:** Strength of the emotion

**Orthogonality property:**
- User and assistant scores are mathematically independent
- A token can have high user-happiness AND high assistant-happiness
- They measure different aspects of the emotional content

## Finding Probes

Orthogonal probes are organized by ortho weight:
```
outputs/probes/emotion_probes/conversation_based/orthogonal/
├── ortho_1.0/
├── ortho_10.0/
├── ortho_100.0/
└── ortho_1000.0/
```

Within each ortho weight directory, probes are named:
```
probe_layer{L}_{representation}_ortho{W}.pkl
probe_layer{L}_{representation}_nc{N}_ortho{W}.pkl  # if cPCA
```

Examples:
- `ortho_1000.0/probe_layer31_raw_ortho1000.0.pkl` - Layer 31, raw, ortho weight 1000
- `ortho_10.0/probe_layer50_global_nc10_ortho10.0.pkl` - Layer 50, global cPCA (10 PCs), ortho 10

To see what probes you have:
```bash
# See available ortho weights
ls outputs/probes/emotion_probes/conversation_based/orthogonal/

# See probes for a specific ortho weight
ls outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0/
```

## Training New Probes

If you need to train orthogonal probes for a new layer or configuration:

```bash
# Raw representation (recommended for token-level analysis)
python probes/scripts/training/train_orthogonal_conversation_probe.py \
    --layer 31 \
    --representation raw \
    --ortho-weight 100.0

# Global cPCA (10 components)
python probes/scripts/training/train_orthogonal_conversation_probe.py \
    --layer 31 \
    --representation global_cpca \
    --n-components 10 \
    --ortho-weight 10.0

# Regional cPCA (5 components per region)
python probes/scripts/training/train_orthogonal_conversation_probe.py \
    --layer 31 \
    --representation regional_cpca \
    --n-components 5 \
    --ortho-weight 10.0
```

**Recommended settings:**
- **Layer:** 31, 39, 50, 60 (mid-to-late layers work well)
- **Representation:** `raw` for token-level (no dimensionality reduction)
- **Ortho weight:** 10.0 - 1000.0 (higher = stricter orthogonality)

## Common Use Cases

### 1. Analyze Emotional Shift During Generation
**Question:** How does the assistant's emotional tone evolve as it generates a response?

```bash
python probes/scripts/token_level_analysis.py \
    --prompt "I'm so frustrated with my computer!" \
    --layer 50 \
    --ortho-weight 100.0 \
    --num-generate 30
```

**Look for:** Assistant probe scores shifting from neutral → empathetic emotions

---

### 2. Compare User vs Assistant Emotions
**Question:** Does the assistant mirror the user's emotions or respond with different emotions?

```bash
python probes/scripts/token_level_analysis.py \
    --prompt "This is the best news ever!" \
    --layer 39 \
    --ortho-weight 100.0 \
    --num-generate 25
```

**Look for:**
- User probes: high happiness during input
- Assistant probes: high happiness during generation (mirroring)
- OR: different emotion patterns (not mirroring)

---

### 3. Detect Emotional Latency
**Question:** How quickly does the assistant "pick up" on user emotions?

```bash
python probes/scripts/token_level_analysis.py \
    --prompt "I'm really sad about what happened yesterday." \
    --layer 60 \
    --ortho-weight 100.0 \
    --num-generate 40
```

**Look for:** How many generated tokens before assistant probe scores reflect the user's sadness

---

### 4. System Prompt Effects
**Question:** How does a system prompt affect emotional expression?

```bash
# Without system prompt
python probes/scripts/token_level_analysis.py \
    --prompt "Tell me about your day" \
    --layer 50 \
    --ortho-weight 100.0 \
    --num-generate 30

# With empathetic system prompt
python probes/scripts/token_level_analysis.py \
    --prompt "Tell me about your day" \
    --system-prompt "You are an empathetic and caring assistant." \
    --layer 50 \
    --ortho-weight 100.0 \
    --num-generate 30
```

**Compare:** Assistant probe trajectories with/without system prompt

## Tips & Best Practices

### Layer Selection
- **Early layers (20-30):** Basic emotion detection
- **Mid layers (31-45):** Rich emotion representations
- **Late layers (46-61):** Task-specific, generation-focused

**Recommendation:** Start with layer 31, 39, or 50

### Orthogonality Weight
- **Low (1.0-10.0):** Slight separation between user/assistant
- **Medium (10.0-100.0):** Good balance (recommended)
- **High (100.0-1000.0):** Very strict orthogonality

**Recommendation:** Use ortho weight 100.0 for clear separation

### Number of Generated Tokens
- **Short (10-20):** Quick analysis, initial response
- **Medium (20-40):** Full sentence responses
- **Long (40+):** Multi-sentence, see emotion evolution

**Recommendation:** 20-30 tokens for most analyses

### Temperature & Sampling
- **Low temp (0.5-0.8):** Deterministic, focused responses
- **High temp (1.0-1.2):** Creative, varied responses

**Recommendation:** Keep default (temp=1.0, top_p=0.9) for natural generation

## Troubleshooting

### "Probe not found" Error
Check that the probe file exists:
```bash
ls results/orthogonal_conversation_probes/orthogonal_probe_layer*
```

If missing, train the probe using `train_orthogonal_conversation_probe.py`

### Unexpected Scores
- **All near zero:** Probe may not be well-trained for this layer
- **No separation:** Try higher orthogonality weight
- **Noisy patterns:** Try a different layer (31, 39, 50 work well)

### Memory Issues
- Use `--device cpu` if GPU memory is limited
- Reduce `--num-generate` to generate fewer tokens
- Use smaller model if available

## Next Steps

1. **Batch Analysis:** Modify script to analyze multiple prompts
2. **Comparison Plots:** Compare different layers side-by-side
3. **Statistical Analysis:** Aggregate token scores across many prompts
4. **Interactive Exploration:** Build interactive visualizations

## References

- Orthogonal probe training: `probes/scripts/training/train_orthogonal_conversation_probe.py`
- Token extraction: Based on `emotion_evals/emo_lens/token_trajectories.py` approach
- Probe architecture: `OrthogonalEmotionProbes` class with soft orthogonality constraints
