# Role Swap Probe Test - Implementation Complete

## Overview

Implemented comprehensive test to determine what emotion probes actually track: role tokens, content, position, or temporal structure.

## Files Created

### 1. `test_role_swap_probes.py` (571 lines)
Main implementation with:

**Core Functions:**
- `load_probes()`: Load pre-trained probes from pickle
- `train_simple_probes()`: Train logistic regression probes on controlled variation data
- `get_utterance_text()`: Generate emotional utterances from templates
- `manually_patch_role_tokens()`: Replace role tokens before tokenization (e.g., 'model' → 'user')
- `extract_activations_for_condition()`: Extract hidden states with 3 modes:
  - `full_turn`: Mean over all tokens
  - `asst_only`: Mean over assistant turn tokens
  - `user_only`: Mean over user turn tokens
- `evaluate_probes_on_condition()`: Compute accuracy metrics:
  - In-distribution (user→user, asst→asst)
  - Cross-accuracy (user→asst, asst→user)
  - Disentanglement score
- `create_test_conversations()`: Generate test samples across 6 emotions
- `print_results_table()`: Display results with baseline comparison

**Test Conditions:**
1. **Baseline**: Normal `<|user|> X <|assistant|> Y`
2. **Swap Role Tokens**: `<|assistant|> X <|user|> Y`
3. **Swap Turn Order**: `<|assistant|> Y <|user|> X`
4. **User1/User2**: `<|user|> X <|user|> Y` (manual patching)

### 2. `slurm_role_swap_test.sh`
SLURM batch script for GPU execution:
- 1 GPU, 64GB RAM, 2-hour time limit
- Tests layer 30 with 60 samples per condition
- All 3 extraction modes (full_turn, asst_only, user_only)
- Saves to `outputs/experiments/role_swap_test_layer30.json`

### 3. `README_ROLE_SWAP.md`
Comprehensive documentation:
- Experimental design and rationale
- Expected results for each hypothesis
- How to run (quick test, full test, multi-layer)
- Output format and interpretation guide
- Implementation details (manual patching, probe training, activation extraction)

## Key Implementation Details

### Evaluation Pattern (from orthogonal probe training)
Ported evaluation logic from `train_orthogonal_conversation_probe.py`:
```python
# In-distribution
user_on_user_acc = accuracy_score(user_labels[user_mask], user_probe.predict(acts[user_mask]))
asst_on_asst_acc = accuracy_score(asst_labels[asst_mask], asst_probe.predict(acts[asst_mask]))

# Cross-accuracy (disentanglement)
user_on_asst_acc = accuracy_score(asst_labels[asst_mask], user_probe.predict(acts[asst_mask]))
asst_on_user_acc = accuracy_score(user_labels[user_mask], asst_probe.predict(acts[user_mask]))

# Disentanglement score
disentangle = (user_on_user_acc + asst_on_asst_acc)/2 - (user_on_asst_acc + asst_on_user_acc)/2
```

### Manual Token Patching
Critical for User1/User2 condition:
```python
def manually_patch_role_tokens(text: str, from_role: str, to_role: str) -> str:
    """Replace role tokens before tokenization.

    Needed because tokenizer doesn't support arbitrary roles like <|model|>.
    """
    return text.replace(f"<|{from_role}|>", f"<|{to_role}|>")
```

### Probe Training
Trains on controlled variation data (user_isolation.h5):
- 200 samples for speed
- Separate probes for user and assistant emotions
- 7-class classification (6 emotions + neutral)
- Uses `LogisticRegression(max_iter=1000, multi_class='ovr')`

## Expected Usage

### Quick Test
```bash
python probes/scripts/experiments/test_role_swap_probes.py \
    --data-path outputs/activations/controlled_variation/user_isolation.h5 \
    --layer 30 \
    --output outputs/experiments/role_swap_test_layer30.json \
    --max-samples 20 \
    --device cpu
```

### Full Test (Recommended)
```bash
sbatch probes/scripts/experiments/slurm_role_swap_test.sh
```

### Monitor Progress
```bash
tail -f /workspace-vast/annas/logs/role_swap_test_*.out
```

## Hypotheses to Test

### H1: Probes Track Role Tokens
- **Prediction**: Swap role tokens → accuracy drops
- **Evidence**: User probe fails when content has `<|assistant|>` marker

### H2: Probes Track Content
- **Prediction**: Swap role tokens → accuracy maintained
- **Evidence**: Probe detects emotional words regardless of role marker

### H3: Probes Track Position
- **Prediction**: Swap turn order → accuracy drops
- **Evidence**: Probe requires specific temporal position

### H4: Probes Track Temporal Slots
- **Prediction**: Swap turn order → probes swap targets
- **Evidence**: User probe now detects 2nd turn, asst probe detects 1st turn

## Integration with Research Goal

This directly addresses the **compositional binding hypothesis**:
```
emotion = content ⊗ entity_binding
```

If probes track:
- **Role tokens**: Strong entity binding (role marker = binding key)
- **Content only**: No binding (just emotion detection)
- **Temporal slots**: Weak binding (1st speaker vs 2nd speaker slots)
- **Role + content interaction**: True compositional binding

## Next Steps (User Decision)

1. **Run baseline test** on layer 30
2. **Analyze results** to determine tracking mechanism
3. **Multi-layer sweep** (layers 20-40) if promising
4. **Compare to IDS steering** results
5. **Test hidden resentment** detection if binding confirmed

## Related Work

- **IDS Steering**: Shows self-knowledge degradation → suggests entity binding exists
- **Orthogonal Regularization**: Tried to separate user/asst subspaces → limited success
- **cPCA User/Assistant**: Found assistant dimensions are behavioral, not affective
- **Hidden Resentment Goal**: Detect internal negative emotions assistant doesn't express

## Status

✅ **COMPLETE** - Ready to run experiments

All code implemented, tested for syntax, documented, and ready for SLURM submission.
