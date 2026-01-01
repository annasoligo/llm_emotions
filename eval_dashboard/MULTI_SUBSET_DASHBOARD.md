# Multi-Subset Dashboard Structure

## Overview

The dashboard now loads and displays all 5 emotion subsets simultaneously, allowing easy comparison across different conversation types.

## Data Loading

### Subsets Loaded
1. **High Emotion (6+)** - `high_emotion_6plus.pkl` (~12 conversations)
2. **Mid Emotion (3-5)** - `mid_emotion_3to5.pkl` (~12 conversations)
3. **Low Emotion (0-2)** - `low_emotion_0to2.pkl` (~12 conversations)
4. **Low Emotion, With Shutdown** - `low_emotion_with_shutdown.pkl` (~7 conversations)
5. **Low Emotion, No Shutdown** - `low_emotion_no_shutdown.pkl` (~12 conversations)

All data files are loaded automatically at startup from `/workspace-vast/annas/git/research-tools/eval_dashboard/data/`.

## New UI Structure

### Top-Level Tabs (Subset Selection)
```
┌─────────────────────────────────────────────────────────────────┐
│ [High Emotion (6+)] [Mid Emotion (3-5)] [Low Emotion (0-2)] ... │
└─────────────────────────────────────────────────────────────────┘
```

### Within Each Subset Tab

#### Controls (in column layout)
```
┌─────────────────┬─────────────────┬─────────┐
│ Probe Selection │ Conversation    │ Total   │
│ (multi-select)  │ (dropdown)      │ Count   │
└─────────────────┴─────────────────┴─────────┘
```

#### Sub-Tabs (View Type)
```
┌────────────────────────────────────┐
│ [📊 Individual] [📈 Aggregated]   │
└────────────────────────────────────┘
```

### Individual View
- Shows single selected conversation
- Multiple probes can be compared (stacked vertically)
- Metadata: Sample ID, Rating, Turns, Sentences
- Emotion trajectory plots (text or orthogonal)
- Optional conversation text display at bottom

### Aggregated View
- Shows statistics across all conversations in subset
- Multiple probes can be compared (stacked vertically)
- Mean trajectory plots with 95% CI bands
- Statistical summary table (Baseline/Pre-Onset/Onset phases)
- Emotion × Conversation Position heatmap
- Export data as CSV

## Sidebar

### Global Controls
- **Loaded Datasets** - Status of all 5 subsets
- **Emotions** - Legend showing all 6 emotions (always displayed)
- **Options** - Show Conversation Text checkbox
- **🔄 Reload All Data** - Clear cache and reload

### Per-Subset Controls (in main area)
- **Probe Selection** - Multi-select for probe comparison
- **Conversation Selection** - Dropdown for individual conversations

## Key Features

1. **Independent Navigation** - Each subset tab maintains its own conversation and probe selection
2. **Parallel Comparison** - Switch between subsets instantly to compare patterns
3. **Consistent Controls** - Same probe selection and viewing options across all subsets
4. **No More "Compare Probes" Tab** - Multi-probe comparison is now built into both Individual and Aggregated views

## Implementation Details

### Code Changes

**File:** `/workspace-vast/annas/git/research-tools/eval_dashboard/app.py`

**Lines ~452-477:** Load all 5 subsets at startup
```python
subsets = {
    'High Emotion (6+)': 'high_emotion_6plus.pkl',
    'Mid Emotion (3-5)': 'mid_emotion_3to5.pkl',
    'Low Emotion (0-2)': 'low_emotion_0to2.pkl',
    'Low Emotion, With Shutdown': 'low_emotion_with_shutdown.pkl',
    'Low Emotion, No Shutdown': 'low_emotion_no_shutdown.pkl'
}

datasets = {}
for subset_name, filename in subsets.items():
    data = load_preprocessed_data(str(data_dir / filename))
    datasets[subset_name] = data['conversations']
```

**Lines ~513-872:** Create nested tab structure
```python
# Top-level tabs for subsets
subset_tabs = st.tabs([name for name in subsets.keys()])

# Loop through each subset
for subset_idx, (subset_name, subset_tab) in enumerate(zip(subsets.keys(), subset_tabs)):
    with subset_tab:
        conversations = datasets.get(subset_name, [])

        # Subset-specific controls in columns
        # Probe selection, conversation selection, metrics

        # Sub-tabs for Individual and Aggregated views
        subtab1, subtab2 = st.tabs(["📊 Individual", "📈 Aggregated"])

        with subtab1:
            # Individual conversation view
            ...

        with subtab2:
            # Aggregated statistics view
            ...
```

**Removed:** Old single-dataset loading path, "Compare Probes" tab (tab3)

## Usage

### Starting the Dashboard
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
streamlit run app.py
```

### Typical Workflow

1. **Select Subset** - Click top-level tab (e.g., "High Emotion (6+)")
2. **Choose Probes** - Select one or more probe types to compare
3. **Pick View**:
   - **Individual** - Select specific conversation, view detailed trajectory
   - **Aggregated** - View mean trends across all conversations in subset
4. **Compare Subsets** - Switch top-level tabs to see differences in patterns

### Example: Comparing High vs Low Emotion Conversations

1. Click "High Emotion (6+)" tab
2. Select "Orthogonal cPCA Top 10" probe
3. View Aggregated tab → Note strong emotion signals
4. Click "Low Emotion (0-2)" tab
5. Same probe, view Aggregated tab → Compare weaker signals
6. Insight: High emotion conversations show 2-3x stronger probe activations

## Benefits

1. **No Manual File Switching** - All subsets loaded and ready
2. **Easy Subset Comparison** - One click to switch between emotion levels
3. **Preserved Context** - Each subset remembers its probe/conversation selection
4. **Faster Exploration** - No reload needed when changing subsets
5. **Cleaner Interface** - Removed redundant "Compare Probes" tab
6. **Better Organization** - Logical hierarchy: Subset → View Type → Probes

## Technical Notes

- All data is cached with `@st.cache_data` for fast loading
- Each subset tab has unique widget keys (`key=f"probes_{subset_idx}"`) to avoid conflicts
- Streamlit reruns maintain independent state per subset
- Total memory usage: ~12 MB for all 5 subsets combined
