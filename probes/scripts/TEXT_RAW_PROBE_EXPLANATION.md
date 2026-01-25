# Text Raw Probe Pipeline Explanation

## Question
Do text_raw probes project neutral PCs out of the activations before applying the probe?

## Answer: NO

Text raw probes (`text_raw` with `n_components=0`) do **NOT** project out neutral PCs. They work directly on raw, unmodified activations.

## Evidence from Code

### 1. Training Script
In [train_emotion_probe_multiseed.py](training/train_emotion_probe_multiseed.py:86-92):

```python
# Apply cPCA projection if provided
if cpca_components is not None:
    if n_components is not None and n_components > 0:
        # Use only top n_components
        cpca_components = cpca_components[:n_components]  # [n_components, hidden_dim]

    print(f"Projecting onto {cpca_components.shape[0]} cPCA components")
    X = X @ cpca_components.T  # [n_samples, n_components]
```

When `n_components=0`, this entire block is skipped. The probes are trained directly on `X` (raw activations).

### 2. Inference Pipeline
In [probe_pipeline.py](probe_pipeline.py:371-432), the `predict()` method:

```python
def predict(
    self,
    activations: np.ndarray,
    layer: int,
    n_components: int = 10,
    seed: int = 0,
    drop_neutral: bool = True
) -> np.ndarray:
    """Run probe inference on activations.

    Pipeline: cPCA projection → probe inference → drop neutral class
    """
    # Load probe
    probe_dict = self.load_probe(layer, n_components, seed)

    # Check if probe uses cPCA
    use_cpca = probe_dict.get('use_cpca', n_components > 0)

    if use_cpca:
        # Get cPCA components
        cpca_components = self.get_cpca_components(layer, n_components)
        # Step 1: Apply cPCA projection
        projected = activations @ cpca_components.T  # [n_samples, n_components]
    else:
        # Use raw activations directly
        projected = activations  # ← TEXT_RAW PROBES USE THIS PATH

    # Step 2: Run probe inference
    probe_model = probe_dict['model']
    # ...
```

When `n_components=0`, `use_cpca=False`, so the code takes the `else` branch and uses `projected = activations` directly.

### 3. Probe Configuration
In [probe_configs.py](../../eval_dashboard/probe_configs.py:59-71):

```python
'text_raw': {
    'name': 'Text-based - Raw (Seed 0)',
    'display_name': 'Text Raw',
    'type': 'linear',
    'probe_dir': PROBE_BASE / "text_based/multiseed",
    'probe_pattern': 'probe_layer{layer}_nc0_seed0.pkl',  # ← nc0 = no cPCA
    'cpca_path': RESEARCH_TOOLS / "probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz",
    'n_components': 0,  # ← No cPCA for raw probes
    'seed': 0,
    'split_user_asst': False,
    'color': '#9b59b6',  # Purple
    'description': 'Text-based probes trained on raw activations (seed 0)'
}
```

The `n_components: 0` configuration explicitly disables cPCA projection.

## What About the cPCA Path?

You might notice that `text_raw` still has a `cpca_path` specified in the config. This is **only used for reference** or for other probe types. The cPCA file contains the top 50 contrastive components that maximize emotion/neutral separation, but these are **NOT used** when `n_components=0`.

### cPCA File Contents
```
File: gemma-3-27b-it_cpca.npz
Shape: (62 layers, 50 components, 5376 hidden_dim)

These are the TOP 50 cPCA components per layer that maximize
contrast between emotional and neutral conversation activations.
```

## Summary Table

| Probe Type | n_components | Uses cPCA? | Projects out neutral PCs? |
|------------|-------------|-----------|---------------------------|
| **text_raw** | 0 | ❌ No | ❌ No - uses raw activations |
| text_cpca | 10 | ✅ Yes | ⚠️ Projects ONTO emotion-relevant PCs (not "out") |
| orthogonal_raw | N/A | ❌ No | ❌ No - uses raw activations |
| orthogonal_cpca_top10 | 10 | ✅ Yes | ⚠️ Projects ONTO emotion-relevant PCs |

## Important Clarification: cPCA Direction

Even when cPCA **is** used (e.g., `text_cpca` with `n_components=10`):

1. **We project ONTO the top emotion-relevant PCs** (the ones that maximize emotion/neutral contrast)
2. **We do NOT project OUT neutral PCs**

This is a key distinction:
- **Project OUT** = Remove components (e.g., via orthogonalization or subtraction)
- **Project ONTO** = Keep only specified components (dimensionality reduction)

The cPCA components are designed to **capture emotion-relevant variation**, not neutral variation. So projecting onto them emphasizes emotion signal, but it's not the same as explicitly removing neutral components.

## Training Data Structure

### Key Finding: Neutral is a Target Class

Looking at [train_emotion_probe.py](training/train_emotion_probe.py:92-119), we can see exactly how training data is prepared:

```python
# Load emotional activations
emotional_acts = acts_group[key]["emotional"][layer]  # [hidden_dim]
all_activations.append(emotional_acts)  # ← RAW emotional activation
all_labels.append(emotion_to_idx[emotion])  # ← Label: anger/disgust/fear/etc

# Load neutral activations
neutral_acts = acts_group[key]["neutral"][layer]  # [hidden_dim]
all_activations.append(neutral_acts)  # ← RAW neutral activation
all_labels.append(emotion_to_idx["neutral"])  # ← Label: NEUTRAL (class 7)
```

**Critical insight:**
- We train on **RAW activations**, not on (activation - neutral_diff)
- Neutral is treated as **one of the 7 target emotion classes**
- The probe learns: `[anger, disgust, fear, happiness, sadness, surprise, neutral]`

### Training Process

1. **Data Collection**: For each conversation, we extract:
   - `emotional[layer]`: Raw activation for emotional version [5376-dim]
   - `neutral[layer]`: Raw activation for neutral version [5376-dim]

2. **Labeling**:
   - Emotional activations → labeled as specific emotion (0-5)
   - Neutral activations → labeled as "neutral" class (6)

3. **Training**:
   - 7-class classification problem
   - Input: Raw 5376-dimensional activation
   - Output: Logits for [anger, disgust, fear, happiness, sadness, surprise, neutral]

4. **Inference** (with `drop_neutral=True`):
   - Apply probe → get 7 logits
   - Drop the neutral class (index 6)
   - Return 6 emotion logits

### Why This Matters

The probe is **NOT** learning emotion as a contrast to neutral. It's learning:
- What anger activations look like (vs all other classes including neutral)
- What disgust activations look like (vs all other classes including neutral)
- What neutral activations look like (vs all emotions)

Then at inference, we simply drop the neutral predictions because we only care about the 6 emotions.

## Conclusion

**Text raw probes:**
- ✅ Trained on raw activations (full 5376-dimensional space)
- ✅ Applied to raw activations during inference
- ✅ Neutral is one of the 7 training classes (but dropped at inference)
- ❌ Do NOT train on (activation - neutral) differences
- ❌ Do NOT project out neutral PCs
- ❌ Do NOT use any dimensionality reduction
- ❌ Do NOT use cPCA components

They are truly "raw" probes working in the full activation space, trained on a 7-class classification problem where neutral is just another class to discriminate against.
