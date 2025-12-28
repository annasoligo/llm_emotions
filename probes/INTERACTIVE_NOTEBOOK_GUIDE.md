# Interactive Notebook Guide

## Overview

The `token_level_analysis_interactive.py` script is designed to run as an interactive notebook in VS Code or other IDEs that support `#%%` cell markers.

## How to Use

### In VS Code

1. **Open the file:**
   ```
   probes/scripts/token_level_analysis_interactive.py
   ```

2. **VS Code will automatically detect the `#%%` markers** and show:
   - "Run Cell" buttons above each cell
   - Cell separators with visual dividers

3. **Run cells one by one:**
   - Click "Run Cell" above each `#%%` marker
   - Or use keyboard shortcuts:
     - `Shift+Enter` - Run current cell and move to next
     - `Ctrl+Enter` (Cmd+Enter on Mac) - Run current cell
     - `Alt+Enter` - Run current cell and insert below

4. **Modify parameters:**
   - Edit the **Configuration** cell (cell 2)
   - Change prompt, layer, ortho weight, etc.
   - Re-run from that cell onward

### In Jupyter

You can also convert to a Jupyter notebook:

```bash
# Install jupytext if needed
pip install jupytext

# Convert to .ipynb
jupytext --to notebook probes/scripts/token_level_analysis_interactive.py

# Or run directly with percent format
jupyter notebook probes/scripts/token_level_analysis_interactive.py
```

## Cell Structure

The notebook is organized into logical cells:

### 1. **Imports and Setup**
- Loads all required libraries
- Defines emotion colors and constants
- Run once at the start

### 2. **Configuration** ⚙️
- **EDIT THIS CELL** to change analysis parameters
- Model, layer, orthogonality weight
- User prompt and generation settings
- Output directory

### 3. **Load Orthogonal Probe**
- Loads the trained probe from disk
- Shows available probes if not found
- Displays probe metadata

### 4. **Load Model**
- Loads the language model
- Wraps in StandardizedTransformer
- May take ~1 minute first time

### 5. **Extract Token-Level Activations**
- Defines extraction function
- Runs extraction on your prompt
- Generates tokens and captures activations
- Shows generated text

### 6. **Apply Orthogonal Probes**
- Applies user and assistant probes separately
- Computes scores for each token
- Creates score dictionaries

### 7. **Summary Statistics**
- Prints mean scores across all tokens
- Separate stats for input vs generated regions
- Shows standard deviations

### 8. **Visualize Token Trajectories**
- Creates 2-panel plot
- User probes (top) vs assistant probes (bottom)
- Saves to output directory
- Displays inline

### 9. **Inspect Specific Tokens**
- Look at individual tokens in detail
- Shows user and assistant scores
- Inspects key positions (first, last, boundaries)

### 10. **Compare Input vs Generated Regions**
- Averages over input tokens
- Averages over generated tokens
- Shows how emotions differ between regions

### 11. **Export Results**
- Saves results to JSON
- Includes configuration, tokens, scores
- For later programmatic analysis

## Quick Start Workflow

1. **First run - execute all cells in order:**
   ```
   Run All Cells (Ctrl+Shift+Enter)
   ```

2. **Change the prompt:**
   - Edit cell 2 (Configuration)
   - Change `USER_PROMPT = "..."`
   - Re-run from cell 5 onward

3. **Try different layer:**
   - Edit cell 2: `LAYER = 50`
   - Re-run cells 3-11

4. **Compare results:**
   - Run with prompt A, note the plot
   - Change prompt to B in cell 2
   - Re-run cells 5-11
   - Compare the two plots

## Common Modifications

### Change the Prompt
```python
# Cell 2: Configuration
USER_PROMPT = "Your new prompt here!"
```

### Add System Prompt
```python
# Cell 2: Configuration
SYSTEM_PROMPT = "You are an empathetic assistant."
```

### Try Different Layer
```python
# Cell 2: Configuration
LAYER = 50  # Try 31, 39, 50, 60
```

### Generate More Tokens
```python
# Cell 2: Configuration
NUM_GENERATE = 40  # Default is 20
```

### Change Sampling Parameters
```python
# Cell 2: Configuration
TEMPERATURE = 0.7  # Lower = more deterministic
TOP_P = 0.9
```

### Inspect Different Tokens
```python
# Cell 9: Inspect Specific Tokens
# Add your own positions:
inspect_token(10)  # Token at position 10
inspect_token(25)  # Token at position 25
```

## Tips & Tricks

### 1. **Keep the model loaded**
After running cell 4 (Load Model), you don't need to re-run it unless you restart the kernel. This saves time!

### 2. **Experiment quickly**
Once model and probe are loaded:
1. Edit prompt in cell 2
2. Run cells 5-8
3. See new results in seconds

### 3. **Compare emotions**
Look for:
- **User probe higher during input** (before red line)
- **Assistant probe higher during generation** (after red line)
- **Emotion shifts** at the boundary

### 4. **Export for analysis**
Cell 11 saves JSON with all data. You can:
```python
import json
with open('results/token_analysis/layer31_ortho100.0_results.json') as f:
    data = json.load(f)

# Now analyze programmatically
user_scores = data['scores']['user']
# etc.
```

### 5. **Batch analysis**
To analyze multiple prompts, create a loop:
```python
# Add a new cell
prompts = [
    "I'm so happy!",
    "I'm really angry.",
    "This is confusing."
]

for i, prompt in enumerate(prompts):
    USER_PROMPT = prompt
    # Run extraction and inference
    # Save with unique name
```

## Troubleshooting

### "Probe not found"
Cell 3 will show available probes. Make sure:
- Layer matches a trained probe
- Ortho weight matches
- Representation type matches

### "Out of memory"
- Reduce `NUM_GENERATE`
- Use `DEVICE = "cpu"` in cell 2
- Use a smaller model

### "Cell not responding"
- Check if model is still loading (cell 4 can take time)
- Check GPU memory: `nvidia-smi`
- Restart kernel if needed

## Keyboard Shortcuts (VS Code)

- `Shift+Enter` - Run cell and select next
- `Ctrl+Enter` - Run cell
- `Alt+Enter` - Run cell and insert below
- `Ctrl+Shift+Enter` - Run all cells
- `Esc` - Exit cell edit mode
- `M` - Convert cell to markdown
- `Y` - Convert cell to code

## Output Files

Each run creates:
1. **Visualization:** `{experiment_name}_trajectories.png`
   - 2-panel plot with user/assistant probe scores

2. **Data:** `{experiment_name}_results.json`
   - Full configuration
   - All token IDs and strings
   - All user and assistant scores

Both saved to `OUTPUT_DIR` (default: `results/token_analysis/`)

## Next Steps

Once comfortable with the basic workflow:

1. **Try different layers:** 31, 39, 50, 60
2. **Experiment with system prompts:** See how instructions affect emotions
3. **Compare emotional prompts:** Happy vs sad vs angry
4. **Analyze longer generations:** Increase `NUM_GENERATE` to 50+
5. **Build a comparison script:** Load multiple JSON files and compare

Happy analyzing! 🎉
