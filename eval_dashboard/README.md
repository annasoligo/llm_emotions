# 🎭 Emotion Onset Evaluation Dashboard

Interactive Streamlit dashboard for visualizing and evaluating emotion probe results on conversation data with detected emotion onsets.

## Features

- **📊 Individual Conversation View**: Detailed emotion trajectories with sentence-level resolution
  - Interactive Plotly charts with hover tooltips
  - Visual separation of user vs assistant turns
  - Annotated conversation text with emotion scores

- **📈 Aggregated Statistics** (Coming soon): Statistical analysis across all conversations
  - Mean trajectories with confidence intervals
  - Onset detection accuracy metrics
  - Effect size comparisons

- **🔬 Probe Comparison** (Coming soon): Side-by-side comparison of different probe types
  - Overlay multiple probe trajectories
  - Correlation matrices
  - Performance benchmarking

## Setup

### 1. Install Dependencies

```bash
cd /workspace-vast/annas/git/research-tools
pip install -r eval_dashboard/requirements.txt
```

### 2. Preprocess Data

First, generate sentence-level probe scores from your emotion onset data:

```bash
python eval_dashboard/data_preprocessing.py \
    --input elicitation/outputs/annotated_emotion_onset_gemma3.jsonl \
    --output eval_dashboard/data/preprocessed_conversations.pkl \
    --probes orthogonal_raw text_raw centroid_k10
```

Options:
- `--input`: Path to annotated emotion onset data (JSONL or JSON)
- `--output`: Where to save preprocessed data
- `--probes`: Which probe types to include (see `probe_configs.py`)
- `--simple-splitter`: Use simple sentence splitter (if Sentences library unavailable)

### 3. Run Dashboard

```bash
streamlit run eval_dashboard/app.py
```

Or use the convenience script:

```bash
./eval_dashboard/run_dashboard.sh
```

The dashboard will open in your browser at `http://localhost:8501`

## Adding New Probes

Edit `probe_configs.py` and add a new entry to `PROBE_CONFIGS`:

```python
'my_new_probe': {
    'name': 'My New Probe',
    'display_name': 'New Probe',
    'type': 'orthogonal',  # or 'standard', 'linear', 'centroid'
    'probe_dir': Path('/path/to/probes'),
    # ... other config
}
```

Then rerun preprocessing with `--probes my_new_probe`.

## Data Structure

### Preprocessed Data Format

```python
{
    'conversations': [
        {
            'sample_id': int,
            'conversation': List[Dict],  # Original conversation
            'rating': float,
            'sentences': [
                {
                    'sentence_id': int,
                    'text': str,
                    'start_token': int,
                    'end_token': int,
                    'turn_role': 'user'/'assistant',
                    'turn_index': int
                },
                ...
            ],
            'probe_scores': {
                'probe_key': {
                    sentence_id: np.array([anger, disgust, fear, happiness, sadness, surprise]),
                    ...
                },
                ...
            },
            'metadata': {...}
        },
        ...
    ],
    'probe_configs': {...},
    'metadata': {...}
}
```

## File Structure

```
eval_dashboard/
├── app.py                      # Main Streamlit application
├── probe_configs.py            # Probe configuration definitions
├── data_preprocessing.py       # Data preprocessing pipeline
├── sentence_aggregator.py      # Sentence splitting and aggregation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── run_dashboard.sh            # Convenience script to run dashboard
└── data/                       # Preprocessed data (gitignored)
    └── preprocessed_conversations.pkl
```

## Usage Tips

### Keyboard Shortcuts

- **r**: Rerun the dashboard
- **c**: Clear cache
- **q**: Quit

### Performance

- The dashboard caches loaded data for fast switching between conversations
- Preprocessing can be slow for large datasets - run overnight if needed
- Consider processing in batches if you have many conversations

### Customization

- **Colors**: Edit `EMOTION_COLORS` in `probe_configs.py`
- **Aggregation**: Change sentence aggregation method in `sentence_aggregator.py`
- **Normalization**: Toggle `normalize_probe_scores` in `probe_configs.py`

## Troubleshooting

### "Sentences library not found"

```bash
pip install sentences
```

Or use `--simple-splitter` flag in preprocessing.

### "No preprocessed data found"

Run the preprocessing script first (see step 2 above).

### Dashboard is slow

- Reduce number of conversations in preprocessing
- Use `@st.cache_data` decorators
- Consider downsampling token-level data

## Future Enhancements

- [ ] Export plots as publication-ready figures
- [ ] Statistical testing (onset detection accuracy)
- [ ] Multi-probe overlay in single view
- [ ] Custom emotion highlighting thresholds
- [ ] PDF report generation
- [ ] Real-time probe inference (without preprocessing)
