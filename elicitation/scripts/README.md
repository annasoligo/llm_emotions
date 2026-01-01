# Emotion Onset Analysis Pipeline

This directory contains scripts for analyzing **when emotions first appear** in model responses during conversations.

## Pipeline Overview

The emotion onset analysis pipeline consists of three main stages:

1. **Annotation**: Use LLM to identify where emotions first appear in conversations
2. **Activation Extraction**: Extract model activations around emotion onset points
3. **Probe Analysis**: Train probes to detect emotion patterns in activations

## Main Scripts

### Core Pipeline Scripts

- **`annotate_dataset.py`** - Annotate emotion onset in conversation dataset
- **`annotate_emotion_onset.py`** - Core annotation logic with token mapping
- **`run_emotion_onset_experiment.py`** - Extract activations around emotion onset
- **`run_emotion_onset_probes.py`** - Train and analyze probes for emotion detection
- **`convert_rating_data.py`** - Convert rating data formats

## Directory Structure

### `/docs/`
Pipeline documentation and analysis guides:
- `EXPERIMENT_PIPELINE.md` - Overview of the full pipeline
- `README_ANNOTATION.md` - Annotation process guide
- `PROBE_ANALYSIS_README.md` - Probe analysis guide
- `ANNOTATION_SUCCESS.md` - Annotation results summary
- `VALIDATION_RESULTS.md` - Validation of annotation accuracy
- `CONTEXT_MATCHING_APPROACH.md` - Token mapping approach
- `TOKEN_MAPPING_EXPLANATION.md` - Token position mapping details
- `KNOWN_LIMITATIONS.md` - Current limitations
- `SETUP_API_KEY.md` - API setup instructions

### `/tests/`
Test scripts for validating pipeline components:
- `test_token_mapping.py` - Test token position mapping
- `test_context_matching.py` - Test context matching
- `test_real_data.py` - Test on real data
- `test_repeated_turns.py` - Test multi-turn handling
- `test_annotation.sh` - Annotation pipeline test

### `/slurm/`
SLURM batch scripts for running on compute cluster:
- `slurm_annotate_dataset.sh` - Run annotation job
- `slurm_onset_experiment.sh` - Run activation extraction
- `slurm_onset_probes.sh` - Run probe analysis
- `slurm_test_*.sh` - Run various tests

## Quick Start

### 1. Annotate Dataset
```bash
python annotate_dataset.py --input ../outputs/summaries/rating_6plus_for_annotation.jsonl
```

### 2. Extract Activations
```bash
python run_emotion_onset_experiment.py
```

### 3. Run Probe Analysis
```bash
python run_emotion_onset_probes.py
```

## Related Outputs

Results are saved to:
- `../outputs/analysis/annotated_emotion_onset*.jsonl` - Annotated datasets
- `../outputs/emotion_onset_probes_*` - Probe analysis results
- `../outputs/activation_cache_*` - Cached activations

## Documentation

See `/docs/EXPERIMENT_PIPELINE.md` for detailed pipeline documentation.
