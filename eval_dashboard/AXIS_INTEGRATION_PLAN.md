# Axis Scoring Integration Plan for eval_dashboard

## Overview
Add dimensional axis scoring (valence, arousal, dominance, approach-avoidance) to the eval_dashboard, displaying axis plots alongside existing emotion plots for comprehensive emotion analysis.

## Current Architecture

### Data Flow
1. **Data Storage**: Pickle files in `eval_dashboard/data/` contain conversations with:
   - `conversations[i]['probe_scores'][probe_key][sentence_id]` = emotion scores array (6 emotions)
   - Each probe type (e.g., 'logit_lens_mean') stores sentence-level scores

2. **Probe Configs** (`probe_configs.py`):
   - `PROBE_CONFIGS` dict defines all probe types
   - Logit lens probes are type 'logit_lens' with config at lines 140-240
   - Each config specifies layers, aggregation, colors, baseline files

3. **App Display** (`app.py`):
   - Loads conversations from pickle files
   - Creates plots using `create_trajectory_plot()` (line 352) or `create_orthogonal_trajectory_plot()` (line 544)
   - Both individual and aggregated views supported

### Logit Lens Implementation
- Computes emotion scores using `emotion_evals/emo_lens/token_trajectories.py`
- Uses layer-averaged activations (layers 40-50 by default)
- Normalizes using baseline statistics
- Stores scores as numpy arrays in `probe_scores` dict

## Implementation Plan

### Phase 1: Update Probe Configs (probe_configs.py)

**Add axis color definitions** (after line 276):
```python
# Axis colors (complementary to emotions but distinct)
AXIS_COLORS = {
    'valence': '#C9A86A',          # Muted gold/amber
    'arousal': '#6B9B9B',          # Muted teal
    'dominance': '#8B85B5',        # Muted blue-purple/periwinkle
    'approach_avoidance': '#B07B7B'  # Muted red/burgundy
}

AXES = ['valence', 'arousal', 'dominance', 'approach_avoidance']
```

**Add axis_lens probe configs** (after line 240):
```python
'axis_lens_mean': {
    'name': 'Axis Lens - Mean Aggregation',
    'display_name': 'Axis Lens (Mean)',
    'type': 'axis_lens',
    'aggregation': 'mean',
    'split_user_asst': False,
    'color': '#ff7f0e',  # Orange (same as logit_lens_mean)
    'description': 'Axis-based emotion detection using mean aggregation (layers 40-50)',
    'layers': list(range(40, 51)),
    'axis_lens_config': {
        'axis_token_groups': None,  # Loaded dynamically
        'baseline_file': RESEARCH_TOOLS / "data/baselines/emo_lens_generated_tokens_avg/baseline_stats.json",
        'model_name': 'google_gemma_3_27b_it',
        'aggregation': 'mean'
    }
},

# Add variants for different layer ranges (L30-40, L20-30) like emotions
# Add max aggregation variants
```

**Add axis loading helper** (after line 323):
```python
def get_axis_lens_config(probe_key: str) -> dict:
    """
    Get axis lens configuration with axis token groups loaded.

    Similar to get_logit_lens_config() but for axes.
    """
    if probe_key not in PROBE_CONFIGS:
        raise ValueError(f"Unknown probe key: {probe_key}")

    config = PROBE_CONFIGS[probe_key].copy()

    if config['type'] == 'axis_lens':
        # Load axis token groups if not already loaded
        if config['axis_lens_config']['axis_token_groups'] is None:
            import sys
            sys.path.insert(0, str(RESEARCH_TOOLS))
            from emotion_evals.emo_lens.logit_lens_emotion_direct import load_axis_token_groups

            model_name = "unsloth/gemma-3-27b-it"
            config['axis_lens_config']['axis_token_groups'] = load_axis_token_groups(model_name)

    return config
```

### Phase 2: Create Data Preprocessing Script (add_axis_lens_to_existing.py)

**Create new script** based on `add_logit_lens_to_existing.py`:
```python
"""
Add axis_lens scores to existing pickle files using emo_lens axis implementation.
Computes dimensional axis scores alongside existing emotion scores.
"""

# Key differences from logit lens script:
# 1. Import axis functions from emo_lens
# 2. Load axis_token_groups instead of emotion_token_ids
# 3. Use compute_token_axis_scores() instead of compute_token_emotion_scores()
# 4. Store results under 'axis_lens_mean', 'axis_lens_max' keys
# 5. Scores are 4D arrays (valence, arousal, dominance, approach_avoidance)
```

**Core logic**:
1. Load model and tokenizer
2. Load axis_token_groups using `load_axis_token_groups()`
3. Load baseline stats (same as emotions - 'generated_tokens_avg')
4. Compute layer-averaged axis baseline stats using `compute_layer_averaged_axis_baseline_stats()`
5. For each conversation:
   - For each sentence:
     - Extract activations from stored activations or recompute
     - Compute axis scores using `compute_token_axis_scores()`
     - Average across tokens in sentence
     - Store as 4D numpy array
6. Save to pickle with new probe keys

### Phase 3: Update Plotting Functions (app.py)

**Option A: Create separate axis plot function** (lines 352-543):
```python
def create_axis_trajectory_plot(
    sentences: List[Dict],
    sentence_scores: Dict[int, np.ndarray],
    selected_axes: List[str],  # ['valence', 'arousal', ...]
    title: str = "Axis Trajectory Over Conversation",
    onset_sentence_id: int = None,
    window_size: int = 5,
    center_scores: bool = False
):
    """
    Create interactive Plotly trajectory plot for dimensional axes.

    Similar to create_trajectory_plot() but:
    - Uses AXIS_COLORS instead of EMOTION_COLORS
    - Uses AXES list instead of EMOTIONS
    - Y-axis label: "Axis Score (High - Low)"
    - Different trace names showing axis interpretation
    """
    # Similar structure to create_trajectory_plot()
    # Main differences:
    # - Iterate over selected_axes instead of selected_emotions
    # - Use AXES.index(axis) for indexing
    # - Color from AXIS_COLORS[axis]
    # - Hover text shows axis polarity (e.g., "Pleasant - Unpleasant")
```

**Option B: Make plotting function generic** (recommended):
- Add `score_type` parameter ('emotion' or 'axis')
- Use appropriate color dict and labels based on type
- Single function handles both emotions and axes

**Update aggregated statistics** (lines 205-349):
- `compute_aggregated_statistics()` already handles any probe scores generically
- No changes needed if axes stored as 4D arrays like emotions (6D)

### Phase 4: Update UI (app.py main function)

**Add axis selection control** (after line 827):
```python
# After emotion selection
st.subheader("Dimensional Axes")

# Show axis legend
axis_html = "<div style='font-size: 0.9em;'>"
for axis in AXES:
    color = AXIS_COLORS[axis]
    axis_html += f"<span style='color: {color}; font-weight: bold;'>● {axis.replace('_', '-').title()}</span>  "
axis_html += "</div>"
st.markdown(axis_html, unsafe_allow_html=True)

# Axis selection (optional, can default to all)
selected_axes = AXES  # Or add checkbox/multiselect
```

**Update plot generation logic** (lines 938-988):
```python
# After emotion plot generation, check if axis scores exist
if probe_key in conv.get('probe_scores', {}):
    # Check if there's a corresponding axis probe
    axis_probe_key = probe_key.replace('logit_lens', 'axis_lens')
    if axis_probe_key in conv.get('probe_scores', {}):
        axis_sentence_scores = conv['probe_scores'][axis_probe_key]

        # Create axis plot
        axis_fig = create_axis_trajectory_plot(
            sentences=sentences,
            sentence_scores=axis_sentence_scores,
            selected_axes=selected_axes,
            title=f"Sample #{conv['sample_id']} - {probe_display_name} (Axes)",
            onset_sentence_id=onset_sent_id,
            window_size=window_size,
            center_scores=center_scores
        )

        st.plotly_chart(axis_fig, use_container_width=True)
```

## Data Structure

### Existing (Emotions):
```python
conv['probe_scores']['logit_lens_mean'][sentence_id] = np.array([
    anger_score,      # float
    disgust_score,    # float
    fear_score,       # float
    happiness_score,  # float
    sadness_score,    # float
    surprise_score    # float
])  # Shape: (6,)
```

### New (Axes):
```python
conv['probe_scores']['axis_lens_mean'][sentence_id] = np.array([
    valence_score,           # float (high = pleasant, low = unpleasant)
    arousal_score,           # float (high = activated, low = calm)
    dominance_score,         # float (high = control, low = overwhelmed)
    approach_avoidance_score # float (high = toward, low = away)
])  # Shape: (4,)
```

## File Changes Summary

### New Files:
1. `eval_dashboard/add_axis_lens_to_existing.py` - Script to add axis scores to pickle files
2. `eval_dashboard/AXIS_INTEGRATION_PLAN.md` - This document

### Modified Files:
1. `eval_dashboard/probe_configs.py`:
   - Add AXIS_COLORS, AXES constants
   - Add axis_lens probe configurations (6-8 variants)
   - Add get_axis_lens_config() helper

2. `eval_dashboard/app.py`:
   - Add create_axis_trajectory_plot() function (or make generic)
   - Update main() to show axis plots after emotion plots
   - Add axis legend/selection UI
   - Update plot generation loop to handle axis probes

3. `eval_dashboard/data/*.pkl` files:
   - Add 'axis_lens_mean', 'axis_lens_max' keys to probe_scores
   - Add corresponding probe configs

## Implementation Steps

1. **Update probe_configs.py** with axis colors and configs
2. **Create add_axis_lens_to_existing.py** script
3. **Run preprocessing** on all 6 subset pickle files
4. **Update app.py** plotting functions
5. **Update app.py** UI to display axis plots
6. **Test** on one subset, then deploy to all

## Validation

### Test Cases:
1. Verify axis scores have correct shape (4,) per sentence
2. Verify axis scores are z-score normalized (mean ~0, std ~1)
3. Check correlation between axes is reasonable (<0.7)
4. Verify plots render correctly for all probe variants
5. Test aggregated statistics work with axis scores
6. Verify centering works with axis scores

### Expected Results:
- Each logit_lens plot now has a matching axis_lens plot below it
- Axis plots use distinct colors (gold, teal, periwinkle, burgundy)
- Both individual and aggregated views show axes
- Smooth integration with existing probe comparison workflow

## Benefits

1. **Richer emotion analysis**: Dimensional axes complement categorical emotions
2. **Validation**: Axes can validate emotion detection (e.g., high sadness → low valence)
3. **Interpretability**: Dimensional axes often more intuitive than discrete emotions
4. **Consistency**: Uses same infrastructure as existing logit lens probes
5. **Efficiency**: ~10% overhead since axes computed from same logits

## Notes

- Axis scores use same baseline normalization as emotions (generated_tokens_avg)
- Axis token groups loaded from same JSON files as emotion tokens
- Layer averaging uses same range as emotions (40-50 default)
- Can add orthogonal axis variants later if needed (user/assistant split)
- Centering works same way for axes as emotions
