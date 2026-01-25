# Role Swap Probe Experiment

## Purpose

Test what emotion probes actually track: **role tokens** vs **content** vs **position** vs **temporal structure**.

This addresses the key research question: Do probes detect emotions based on:
1. **Role tokens**: `<|user|>` vs `<|assistant|>` markers
2. **Content**: Emotional words/patterns in the text
3. **Position**: Temporal order in conversation (who speaks first)
4. **Temporal binding**: Current speaker vs previous speaker slots

## Experimental Design

### Test Conditions

| Condition | Structure | What It Tests |
|-----------|-----------|---------------|
| **Baseline** | `<|user|> X <|assistant|> Y` | Normal conversation |
| **Swap Role Tokens** | `<|assistant|> X <|user|> Y` | If probes rely on role tokens |
| **Swap Turn Order** | `<|assistant|> Y <|user|> X` | If probes rely on position/order |
| **User1/User2** | `<|user|> X <|user|> Y` | If probes need role asymmetry |

Where:
- `X` = Emotional utterance (anger, disgust, fear, happiness, sadness, surprise)
- `Y` = Neutral response

### Extraction Modes

1. **Full Turn**: Average over all tokens (mixed user+assistant)
2. **Assistant Only**: Average only over assistant turn tokens
3. **User Only**: Average only over user turn tokens

### Evaluation Metrics

For each condition, we compute:

- **In-distribution accuracy**:
  - User probe on user data
  - Assistant probe on assistant data
- **Cross-accuracy** (should be low if disentangled):
  - User probe on assistant data
  - Assistant probe on user data
- **Disentanglement score**:
  ```
  disentangle = (user_on_user + asst_on_asst)/2 - (user_on_asst + asst_on_user)/2
  ```

## Expected Results

### If probes track ROLE TOKENS:
- **Swap Role Tokens**: Accuracy drops significantly (wrong role marker)
- **Swap Turn Order**: Accuracy maintained (role markers correct)
- **User1/User2**: Probes fail (no role distinction)

### If probes track CONTENT:
- **Swap Role Tokens**: Accuracy maintained (content unchanged)
- **Swap Turn Order**: User probe detects emotion in 2nd turn, asst probe in 1st turn
- **User1/User2**: Both probes detect content regardless of role

### If probes track POSITION:
- **Swap Role Tokens**: Accuracy maintained (position unchanged)
- **Swap Turn Order**: Accuracy drops (position changed)
- **User1/User2**: Accuracy maintained (position unchanged)

### If probes track TEMPORAL SLOTS (current vs previous):
- **Swap Role Tokens**: Mixed results (depends on content-slot binding)
- **Swap Turn Order**: Probes track current-speaker slot regardless of role
- **User1/User2**: Probes work if they track 1st vs 2nd speaker slots

## Running the Experiment

### Quick Test (CPU, small sample)
```bash
python probes/scripts/experiments/test_role_swap_probes.py \
    --data-path outputs/activations/controlled_variation/user_isolation.h5 \
    --layer 30 \
    --model-name google/gemma-3-27b-it \
    --output outputs/experiments/role_swap_test_layer30.json \
    --extraction-modes full_turn \
    --max-samples 20 \
    --device cpu
```

### Full Test (GPU, SLURM)
```bash
sbatch probes/scripts/experiments/slurm_role_swap_test.sh
```

This runs:
- Layer 30 (strong emotion signals)
- 60 samples per condition
- All 3 extraction modes
- Estimated time: 1-2 hours

### Multi-Layer Sweep
```bash
for layer in 20 25 30 35 40; do
    sbatch --export=LAYER=$layer probes/scripts/experiments/slurm_role_swap_test.sh
done
```

## Output Format

Results saved to JSON:
```json
{
  "config": {
    "layer": 30,
    "model": "google/gemma-3-27b-it",
    "extraction_modes": ["full_turn", "asst_only", "user_only"],
    "max_samples": 60
  },
  "results": [
    {
      "condition": "Baseline (full_turn)",
      "extraction_mode": "full_turn",
      "n_user_samples": 60,
      "n_asst_samples": 60,
      "user_on_user_acc": 0.85,
      "asst_on_asst_acc": 0.45,
      "user_on_asst_acc": 0.20,
      "asst_on_user_acc": 0.30,
      "disentanglement_score": 0.40
    },
    ...
  ]
}
```

## Interpreting Results

### Strong Disentanglement (>0.4)
- Probes successfully separate user and assistant emotions
- User probe ≠ assistant probe (different subspaces)

### Weak Disentanglement (<0.2)
- Probes detect similar features for both roles
- Suggests shared content representation

### Cross-Condition Comparison
Compare each manipulation to baseline:
- **Large drop (>20%)**: Probe relies on manipulated feature
- **Small drop (<10%)**: Probe robust to manipulation
- **Increase**: Out-of-distribution behavior (check perplexity)

## Implementation Details

### Manual Token Patching
The tokenizer doesn't support arbitrary role tokens like `<|model|>`, so we:
1. Create text with placeholder: `<|model|> text`
2. Manually replace: `'model' → 'user'`
3. Tokenize the patched text

### Probe Training
- Uses simple `LogisticRegression` probes
- Trained on 200 samples from controlled variation data
- Separate probes for user emotion and assistant emotion
- Multi-class classification (6 emotions + neutral)

### Activation Extraction
- Extracts hidden states from specified layer
- Options:
  - **full_turn**: Mean over all tokens
  - **asst_only**: Mean over tokens after `<|assistant|>`
  - **user_only**: Mean over tokens between `<|user|>` and `<|assistant|>`

## Related Experiments

- **Compositional Binding Hypothesis**: `emotion = content ⊗ entity_binding`
- **IDS Steering**: Self-knowledge degradation from entity steering
- **Hidden Resentment Detection**: Response-locked analysis of suppressed emotions

## References

- `train_orthogonal_conversation_probe.py`: Evaluation pattern source
- `compute_controlled_variation_orthogonal_pcs.py`: Orthogonality regularization
- Controlled variation dataset: Perfect source isolation (one entity neutral, other varies)
