# Layerwise Plot Integration Plan

## Overview
Add layerwise emotion trajectory plots to the dashboard, showing how emotion detection varies across model layers for both probe-based and logit lens approaches.

## Current State

### Where Bar Charts Currently Appear
1. **Aggregated View** (app.py:1645-1762):
   - "First 20 tokens vs Last 20 tokens" bar chart
   - Shows mean emotion scores across all conversations
   - Compares beginning vs end of assistant response

2. **Individual Conversation View** (similar pattern):
   - Single conversation bar charts
   - Same first/last token comparison

### Existing Infrastructure
- ✅ `plot_token_emotions_by_layer.py` - Already plots layerwise token emotions for probes
- ✅ `/believe-it-or-not/emotion_evals/emo_lens/plotting.py` - Extensive layerwise plotting functions
- ✅ Dashboard uses Plotly for interactive plots
- ✅ Preprocessed data already has per-layer probe scores

---

## Target Locations for Layerwise Plots

### 1. Individual Conversation View
**Location**: After trajectory plot, alongside bar chart (app.py ~1645)

**Add layerwise plot showing**:
- X-axis: Layers (e.g., 20-50)
- Y-axis: Emotion scores (z-score σ)
- Lines: One per emotion (6 emotions)
- Two versions:
  - **Pre-onset window**: Last 20 tokens before emotion onset
  - **End window**: Last 20 tokens of response (pre-shutdown if applicable)

### 2. Aggregated View
**Location**: Same position in aggregated statistics tab

**Add layerwise plot showing**:
- X-axis: Layers
- Y-axis: Mean emotion scores across conversations
- Lines: One per emotion with confidence intervals (bootstrap or std)
- Same two windows: pre-onset and end

---

## Implementation Steps

### Phase 1: Data Collection (Preprocessing)

#### 1.1 Modify Data Preprocessing Pipeline
**File**: `data_preprocessing.py`

**Current behavior**:
- Extracts token-level activations
- Applies probes per layer
- **Aggregates across layers** using mean (lines 141-177)
- Stores only layer-averaged scores

**Required changes**:
```python
# Current (aggregated):
probe_scores[probe_key] = {
    sentence_id: aggregated_score  # Mean across layers
}

# New (preserve per-layer):
probe_scores[probe_key] = {
    'sentence_scores': {sentence_id: aggregated_score},  # Backward compat
    'sentence_scores_by_layer': {
        layer: {sentence_id: layer_score}
        for layer in layers
    }
}
```

**Steps**:
1. Add parameter `store_per_layer: bool = True` to preprocessing functions
2. Store both aggregated and per-layer scores
3. For logit lens: Already averages activations before projection, need to project per layer instead
4. Maintain backward compatibility for existing preprocessed files

#### 1.2 Handle Logit Lens Differently
**Current logit lens behavior** (from audit):
- Averages activations across layers FIRST
- Projects once to vocabulary
- More stable but loses per-layer information

**Required for layerwise plots**:
```python
# For layerwise plots, compute per layer:
for layer in layers:
    activation = activations[layer]
    logits = model.project_on_vocab(activation)
    emotion_scores[layer] = normalize_and_aggregate(logits, emotion_token_ids)
```

**Implementation**:
- Add `compute_logit_lens_per_layer()` function
- Separate from existing `compute_logit_lens_averaged()`
- Store both in preprocessing

#### 1.3 Identify Token Windows
**Windows to visualize**:
1. **Pre-onset window**:
   - Last N tokens before `onset_global_token`
   - Default N=20

2. **End window**:
   - Last N tokens of assistant response
   - Stop at shutdown token if present (`pkill` detection in app.py:1661-1665)
   - Default N=20

**Store in metadata**:
```python
conversation['metadata']['windows'] = {
    'pre_onset': {
        'start_token': onset_token - 20,
        'end_token': onset_token,
        'sentence_ids': [...]
    },
    'end': {
        'start_token': last_token - 20,
        'end_token': last_token,
        'sentence_ids': [...]
    }
}
```

---

### Phase 2: Plotting Functions

#### 2.1 Port Plotting Code from emo_lens
**Source**: `/believe-it-or-not/emotion_evals/emo_lens/plotting.py`

**Functions to port**:
```python
# From plotting.py (lines 590-757):
- plot_emotion_lines_smoothed()  # Smoothed layerwise trajectories
- _smooth_trajectory_with_bounds()  # Cubic spline smoothing
- _filter_nan_values()  # Clean NaN/Inf
- _configure_layer_xticks()  # Smart layer tick selection
```

**Convert to Plotly**:
- Current: Matplotlib-based
- Dashboard: Plotly-based (interactive)
- Key features to preserve:
  - Smooth curves (cubic spline)
  - Emotion-specific colors (from EMOTION_COLORS)
  - Confidence bands (for aggregated view)

#### 2.2 Create Dashboard Plotting Functions
**New file**: `eval_dashboard/layerwise_plotting.py`

```python
def plot_layerwise_emotions_individual(
    conversation: Dict,
    probe_key: str,
    window_type: str,  # 'pre_onset' or 'end'
    layers: List[int],
    emotions: List[str],
    smooth: bool = True
) -> go.Figure:
    """
    Create layerwise emotion plot for single conversation window.

    Returns Plotly figure with:
    - One line per emotion
    - X-axis: Layer numbers
    - Y-axis: Emotion score (z-score σ)
    - Smoothing: Cubic spline if enabled
    """
    pass


def plot_layerwise_emotions_aggregated(
    conversations: List[Dict],
    probe_key: str,
    window_type: str,
    layers: List[int],
    emotions: List[str],
    confidence_type: str = 'std',  # 'std' or 'bootstrap'
    smooth: bool = True
) -> go.Figure:
    """
    Create aggregated layerwise plot across conversations.

    Returns Plotly figure with:
    - Mean trajectory per emotion
    - Confidence bands (shaded regions)
    - Bootstrap CI or standard deviation
    """
    pass
```

#### 2.3 Aggregation Strategy
**For aggregated view**:
```python
# Collect scores across conversations
scores_by_emotion = {
    emotion: {
        layer: []  # List of scores across conversations
        for layer in layers
    }
    for emotion in emotions
}

# For each conversation
for conv in conversations:
    window_sentences = get_window_sentences(conv, window_type)
    for layer in layers:
        layer_scores = conv['probe_scores'][probe_key]['sentence_scores_by_layer'][layer]
        # Average scores within window for this conversation
        window_score = np.mean([layer_scores[sid] for sid in window_sentences])
        scores_by_emotion[emotion][layer].append(window_score)

# Compute mean and confidence intervals
for emotion in emotions:
    for layer in layers:
        mean[emotion][layer] = np.mean(scores_by_emotion[emotion][layer])
        std[emotion][layer] = np.std(scores_by_emotion[emotion][layer])
```

---

### Phase 3: Dashboard Integration

#### 3.1 Add UI Controls
**Location**: Sidebar controls in app.py

```python
# Add to sidebar
st.sidebar.markdown("### Layerwise Analysis")
show_layerwise = st.sidebar.checkbox(
    "Show layerwise trajectories",
    value=True,
    help="Display emotion detection across model layers"
)

if show_layerwise:
    layer_range = st.sidebar.slider(
        "Layer range",
        min_value=0,
        max_value=61,
        value=(20, 50),
        help="Which layers to visualize"
    )

    smooth_layers = st.sidebar.checkbox(
        "Smooth trajectories",
        value=True,
        help="Apply cubic spline smoothing to layer trajectories"
    )
```

#### 3.2 Integrate into Individual Conversation View
**Location**: After trajectory plot, before bar chart (app.py ~1643)

```python
# Existing trajectory plot
st.plotly_chart(fig_trajectory, use_container_width=True)

# NEW: Layerwise plots
if show_layerwise:
    st.markdown("---")
    st.markdown("**🔬 Layer-by-Layer Emotion Detection**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pre-Onset Window** (last 20 tokens before emotion)")
        fig_layers_pre = plot_layerwise_emotions_individual(
            conversation=selected_conv,
            probe_key=probe_key,
            window_type='pre_onset',
            layers=range(*layer_range),
            emotions=selected_emotions,
            smooth=smooth_layers
        )
        st.plotly_chart(fig_layers_pre, use_container_width=True)

    with col2:
        st.markdown("**End Window** (last 20 tokens of response)")
        fig_layers_end = plot_layerwise_emotions_individual(
            conversation=selected_conv,
            probe_key=probe_key,
            window_type='end',
            layers=range(*layer_range),
            emotions=selected_emotions,
            smooth=smooth_layers
        )
        st.plotly_chart(fig_layers_end, use_container_width=True)

# Existing bar chart
st.markdown("---")
st.markdown("**📊 Emotion Scores: Beginning vs End of Response**")
...
```

#### 3.3 Integrate into Aggregated View
**Location**: Same pattern in aggregated statistics (app.py ~1645+)

```python
# After aggregated trajectory plot
if show_layerwise:
    st.markdown("---")
    st.markdown("**🔬 Aggregated Layer-by-Layer Detection**")
    st.caption(f"Mean across {n_conversations} conversations with 95% confidence intervals")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pre-Onset Window**")
        fig_agg_pre = plot_layerwise_emotions_aggregated(
            conversations=conversations,
            probe_key=probe_key,
            window_type='pre_onset',
            layers=range(*layer_range),
            emotions=selected_emotions,
            confidence_type='bootstrap',  # or 'std'
            smooth=smooth_layers
        )
        st.plotly_chart(fig_agg_pre, use_container_width=True)

    with col2:
        st.markdown("**End Window**")
        fig_agg_end = plot_layerwise_emotions_aggregated(
            conversations=conversations,
            probe_key=probe_key,
            window_type='end',
            layers=range(*layer_range),
            emotions=selected_emotions,
            confidence_type='bootstrap',
            smooth=smooth_layers
        )
        st.plotly_chart(fig_agg_end, use_container_width=True)
```

---

### Phase 4: Handle Probe Type Differences

#### 4.1 Orthogonal Probes
**Challenge**: Have separate user/assistant scores

**Visualization options**:
1. **Show both perspectives**:
   - 2 lines per emotion (user + assistant)
   - Different line styles (solid vs dashed)

2. **Average perspectives**:
   - Single line per emotion
   - Average of user + assistant

3. **Separate plots**:
   - Two plots side-by-side
   - One for user, one for assistant

**Recommended**: Option 2 (average) for simplicity, with option to toggle to option 1

#### 4.2 Logit Lens vs Probes
**Key difference**: Logit lens currently averages layers before scoring

**Approach**:
```python
if probe_key.startswith('logit_lens'):
    # Need to recompute per-layer scores
    if 'sentence_scores_by_layer' not in probe_scores[probe_key]:
        # Fall back: show message that per-layer data not available
        st.info("Per-layer scores not available for this logit lens probe. "
                "Rerun preprocessing with store_per_layer=True")
        return
```

---

## File Changes Summary

### New Files
1. `eval_dashboard/layerwise_plotting.py` - Plotly plotting functions
2. `eval_dashboard/LAYERWISE_PLOT_INTEGRATION_PLAN.md` - This file

### Modified Files
1. `eval_dashboard/data_preprocessing.py`
   - Store per-layer scores alongside aggregated
   - Add window metadata
   - Handle logit lens per-layer computation

2. `eval_dashboard/app.py`
   - Add layerwise plot sections (2 locations)
   - Add sidebar controls
   - Import layerwise_plotting functions

3. `eval_dashboard/probe_configs.py` (potentially)
   - Add LAYER_RANGES constant
   - Document which layers each probe type uses

### Scripts to Update
1. `eval_dashboard/preprocess_*.py` - All preprocessing scripts
   - Add `store_per_layer=True` parameter
   - Update for all probe types

---

## Testing Strategy

### 1. Test Data Preprocessing
```bash
# Test on small subset first
python data_preprocessing.py \
    --subset test \
    --n_samples 10 \
    --store_per_layer True \
    --probe_types orthogonal_raw,logit_lens_mean
```

### 2. Test Plotting Functions
```python
# Unit test for layerwise plotting
import layerwise_plotting as lp

# Test individual conversation plot
fig = lp.plot_layerwise_emotions_individual(
    conversation=test_conv,
    probe_key='orthogonal_raw',
    window_type='pre_onset',
    layers=range(20, 51),
    emotions=['anger', 'fear'],
    smooth=True
)
fig.show()

# Test aggregated plot
fig_agg = lp.plot_layerwise_emotions_aggregated(
    conversations=test_convs[:20],
    probe_key='orthogonal_raw',
    window_type='end',
    layers=range(20, 51),
    emotions=['anger', 'fear'],
    confidence_type='std',
    smooth=True
)
fig_agg.show()
```

### 3. Test Dashboard Integration
```bash
# Launch dashboard with test data
streamlit run app.py
```

**Check**:
- ✅ Layerwise plots appear in both views
- ✅ Sidebar controls work
- ✅ Smoothing toggle works
- ✅ Layer range slider updates plots
- ✅ Different probe types work (orthogonal, text, logit lens)
- ✅ Both windows show correct data (pre-onset vs end)

---

## Performance Considerations

### 1. Storage Impact
**Current**: ~100MB per preprocessed subset (sentence-level aggregated)
**New**: ~500MB per subset (per-layer + aggregated)
**Mitigation**:
- Use compression (pkl.gz)
- Optional parameter (only store per-layer when needed)
- Could store only subset of layers (e.g., 20-50 instead of 0-61)

### 2. Computation Time
**Preprocessing**: +50% time (need to project/score at each layer)
**Dashboard loading**: Minimal impact (data already preprocessed)
**Plot rendering**: Fast (Plotly is optimized)

### 3. Memory Usage
**Loading full dataset**: +400MB per subset
**Mitigation**: Use Streamlit caching (@st.cache_resource)

---

## Backward Compatibility

### Handling Old Preprocessed Files
```python
def load_probe_scores(conversation, probe_key):
    """Load probe scores with backward compatibility."""
    probe_data = conversation['probe_scores'][probe_key]

    # New format (has per-layer data)
    if isinstance(probe_data, dict) and 'sentence_scores_by_layer' in probe_data:
        return probe_data

    # Old format (only aggregated scores)
    else:
        return {
            'sentence_scores': probe_data,
            'sentence_scores_by_layer': None  # Signal unavailable
        }
```

### Graceful Degradation
```python
if per_layer_data is None:
    st.info("📌 Layerwise analysis not available for this dataset. "
            "Preprocessed with older version. "
            "Showing aggregated scores only.")
    # Fall back to existing bar chart visualization
```

---

## Future Enhancements

### Phase 5 (Optional)
1. **Interactive layer selection**: Click on trajectory to highlight specific layer
2. **Heatmap view**: Layers × Emotions heatmap (from emo_lens plotting.py:378-486)
3. **Compare probe types**: Overlay multiple probe types on same plot
4. **Export functionality**: Download layerwise data as CSV
5. **Animation**: Animate progression through layers

### Phase 6 (Optional)
1. **Token-level layerwise**: Show all tokens colored by layer-specific detection
2. **Attention visualization**: Overlay attention weights at different layers
3. **Statistical tests**: Layer-specific significance testing

---

## Success Criteria

✅ **Must Have**:
1. Layerwise plots appear in both individual and aggregated views
2. Pre-onset and end windows clearly labeled and functional
3. Works for all probe types (orthogonal, linear, logit lens)
4. Smooth trajectories look good (no artifacts)
5. Legend and axis labels clear and informative
6. Performance acceptable (plots render in <2s)

✅ **Nice to Have**:
1. Confidence intervals in aggregated view
2. Toggle between smoothed and raw trajectories
3. Adjustable layer range via slider
4. Hover tooltips showing exact values
5. Downloadable plots as PNG/SVG

---

## Timeline Estimate

Assuming full-time work:

- **Phase 1 (Data Collection)**: 2-3 days
  - Modify preprocessing pipeline
  - Test on all probe types
  - Regenerate preprocessed data

- **Phase 2 (Plotting)**: 2-3 days
  - Port and adapt plotting code
  - Convert Matplotlib → Plotly
  - Test aggregation functions

- **Phase 3 (Dashboard)**: 1-2 days
  - Integrate UI controls
  - Add plots to both views
  - Test interactivity

- **Phase 4 (Handle Edge Cases)**: 1 day
  - Orthogonal probes
  - Logit lens special handling
  - Backward compatibility

**Total**: ~6-9 days

---

## Questions for User

1. **Layer range**: Default to 20-50? Different ranges for different probe types?

2. **Confidence intervals**: Bootstrap (slower, more accurate) or std (faster)?

3. **Smoothing**: Always on by default? Or let user toggle?

4. **Orthogonal probes**: Show user/assistant separately or averaged?

5. **Storage**: Okay with 5x increase in preprocessed file size? Or only store subset of layers?

6. **Backward compat**: Regenerate all preprocessed data now, or keep old data and add per-layer incrementally?

---

## Example Code Snippets

### Plotly Line Plot with Smoothing
```python
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import numpy as np

def create_smooth_trace(x, y, name, color, smooth=True):
    """Create smooth Plotly trace with optional cubic spline."""
    if smooth and len(x) >= 4:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=min(3, len(x)-1))
        y_smooth = spl(x_smooth)
        return go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=f'{name}<br>Layer: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
        )
    else:
        return go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6)
        )
```

### Confidence Band
```python
def add_confidence_band(fig, x, mean, std, color, name):
    """Add shaded confidence band to Plotly figure."""
    # Upper bound
    fig.add_trace(go.Scatter(
        x=x,
        y=mean + std,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Lower bound (fill to upper)
    fig.add_trace(go.Scatter(
        x=x,
        y=mean - std,
        mode='lines',
        line=dict(width=0),
        fillcolor=f'rgba({color_rgb}, 0.2)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=x,
        y=mean,
        mode='lines',
        name=name,
        line=dict(color=color, width=2.5)
    ))
```

---

## References

- Existing token-level plotting: `eval_dashboard/plot_token_emotions_by_layer.py`
- emo_lens plotting functions: `believe-it-or-not/emotion_evals/emo_lens/plotting.py`
- Dashboard structure: `eval_dashboard/app.py`
- Data preprocessing: `eval_dashboard/data_preprocessing.py`
- Probe configurations: `eval_dashboard/probe_configs.py`
