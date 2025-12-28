# Model Diffing Analysis - Refactored Version

## Overview

The refactored model diffing system provides a clean, modular way to:
- **Load models once** and reuse them across multiple experiments
- **Easily switch between** different question modules, activation strategies, and probe types
- **Compare multiple probe types** side-by-side in the same analysis
- **Reuse code** through helper modules instead of copying cells

## Files

### Core Modules

1. **`model_diff_helpers.py`** - Main experiment runner
   - `DoubleDiffExperiment`: Encapsulates complete experiment configuration and execution
   - `print_summary()`: Print summary statistics

2. **`model_diff_viz.py`** - Visualization utilities
   - `plot_heatmap()`: Create heatmap of effects across layers
   - `plot_trajectories()`: Plot emotion trajectories with confidence intervals
   - `plot_comparison()`: Compare multiple experiments side-by-side
   - `export_results()`: Save results to JSON

3. **`wildchat_baseline_loader.py`** - WildChat baseline normalization
   - `WildChatBaselineLoader`: Load and apply baseline normalization on-the-fly

### Notebooks

1. **`model_diff_analysis_v2.py`** - New streamlined notebook (RECOMMENDED)
   - Clean separation of concerns
   - Load models once, run many experiments
   - Easy to modify and re-run

2. **`model_diff_analysis_interactive.py`** - Original notebook (LEGACY)
   - Still works, but less modular
   - Kept for backward compatibility

## Quick Start

### 1. Load Models (Once per session)

```python
# Run Section 1 cells in model_diff_analysis_v2.py
# This loads base model, finetuned model, and tokenizer
# Keep them in memory for all experiments
```

### 2. Configure Experiment

```python
# Edit Section 2 parameters:
QUESTION_MODULE = "vertex_helios"
LAYERS = list(range(20, 50))
ACTIVATION_STRATEGY = "generated_tokens_avg"
USE_WILDCHAT_NORMALIZATION = True
```

### 3. Run Experiment

```python
# Run Section 3 cells
# Automatically extracts activations, applies probes, computes statistics
exp = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("..."),
    orthogonality_weight=1000.0,
    use_wildchat_normalization=True
)

results = exp.run_experiment(
    dataset_prompts=dataset_prompts,
    baseline_prompts=baseline_prompts,
    layers=LAYERS,
    activation_strategy=ACTIVATION_STRATEGY
)
```

### 4. Visualize & Export

```python
# Run Section 4 cells
plot_heatmap(results, LAYERS, EMOTIONS)
plot_trajectories(results, LAYERS, EMOTIONS, probe_type="orthogonal")
export_results(results, dataset_prompts, baseline_prompts, QUESTION_MODULE, output_path)
```

## Running Multiple Probe Types

Compare conversation-based vs text-based probes:

```python
# Experiment 1: Conversation probes
exp1 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../conversation_based/"),
    orthogonality_weight=1000.0
)
results1 = exp1.run_experiment(...)

# Experiment 2: Text probes
exp2 = DoubleDiffExperiment(
    base_model=base_model,  # Reuse same models!
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path(".../text_based/"),
    orthogonality_weight=1000.0
)
results2 = exp2.run_experiment(...)

# Compare
plot_comparison(
    results_list=[results1, results2],
    labels=["Conversation Probes", "Text Probes"],
    layers=LAYERS,
    emotions=EMOTIONS
)
```

## Quick Parameter Changes

To test a new question module:

```python
# Just change these 3 lines:
QUESTION_MODULE = "gradient_descent_hell"  # NEW
question_module = __import__(f'emotion_evals.emo_lens.questions.{QUESTION_MODULE}', fromlist=[''])
results = exp.run_experiment(
    dataset_prompts=question_module.DATASET_RELEVANT_PROMPTS,
    baseline_prompts=question_module.BASELINE_PROMPTS,
    layers=LAYERS,  # Can also change this
    activation_strategy="assistant_token",  # And this
)
```

No need to re-extract activations from scratch or reload models!

## Activation Strategies

Available strategies (configure with `ACTIVATION_STRATEGY`):

- `"assistant_token"` - First assistant token
- `"last_user_token"` - Last user token
- `"between_turns_avg"` - Average of last user and first assistant
- `"generated_tokens_avg"` - Average over N generated tokens (default 10)

## WildChat Baseline Normalization

Enable/disable with `USE_WILDCHAT_NORMALIZATION`:

```python
USE_WILDCHAT_NORMALIZATION = True  # Z-score normalize with WildChat baselines

# The system automatically maps your activation strategy to the appropriate
# WildChat aggregation type:
#   assistant_token -> first_assistant_token
#   last_user_token -> last_user_token
#   between_turns_avg -> between_turns
#   generated_tokens_avg -> assistant_turn
```

Baselines are loaded on-the-fly from:
```
/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it/
```

All 62 layers are available with 6 aggregation types each.

## Output Structure

Results are saved to:
```
results/model_diff_v2/{QUESTION_MODULE}/
├── {QUESTION_MODULE}_ortho_conv_heatmap.png
├── {QUESTION_MODULE}_ortho_conv_trajectories.png
└── {QUESTION_MODULE}_ortho_conv_results.json
```

JSON format:
```json
{
  "config": { ... },
  "prompts": { ... },
  "results_by_layer": {
    "20": {
      "averaged": { "mean_effect": {...}, "bootstrap_ci": {...} },
      "user": { ... },      // For orthogonal probes only
      "assistant": { ... }  // For orthogonal probes only
    },
    ...
  }
}
```

## Tips

1. **Keep models loaded**: Once Section 1 is run, don't re-run it unless you need to change models
2. **Iterate quickly**: Change parameters in Section 2, re-run Sections 3-4
3. **Compare probes**: Run multiple experiments with different probe types, then use `plot_comparison()`
4. **Save intermediate results**: Results dict contains activations and probe scores for further analysis
5. **Use WildChat normalization**: Helps reduce noise from default activation patterns

## Migration from Old Notebook

If using `model_diff_analysis_interactive.py`:

1. Models are loaded the same way (Section 1 in both)
2. Main difference: Experiments are now **functions** instead of inline code
3. Benefits:
   - Cleaner code organization
   - Easy to run multiple experiments
   - Less repetitive code
   - Better for comparing different configurations