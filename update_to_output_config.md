# Scripts Updated to Use output_config.py

This document tracks which scripts have been updated to use the centralized `output_config.py` module.

## Summary

The migration preserves backward compatibility through symlinks. Scripts can continue using old paths (e.g., `results/emotion_probes_top10`) which now point to the new organized structure.

## Key Points

1. **Symlinks created**: Old paths → New paths
   - `probes/results/cpca_tier_data` → `outputs/dimensionality_reduction/cpca/tier_based/google`
   - `results/emotion_probes_*` → `outputs/probes/emotion_probes/text_based/*`
   - etc.

2. **output_config.py provides**:
   - `OutputPaths` class with organized path structure
   - `LEGACY_TO_NEW_PATHS` mapping for automatic translation
   - Helper functions: `get_probe_output_dir()`, `get_cpca_output_dir()`

3. **Scripts work with either path**:
   - Old: `results/emotion_probes_top10`
   - New: `outputs/probes/emotion_probes/text_based/cpca/top10`
   - Both resolve to the same location via symlinks

## Scripts with Hardcoded Paths

### Training Scripts
- ✅ `train_emotion_probe.py` - Default: `results/emotion_probes` (works via symlink)
- ✅ `train_emotion_probe_multiseed.py` - Default: `results/emotion_probes_multiseed` (works via symlink)
- ✅ `train_layer_all_configs.py` - Default: `results/emotion_probes_multiseed` (works via symlink)
- `train_conversation_probe.py` - Uses default that references `probes/results/` (works via symlink)
- `train_orthogonal_conversation_probe.py` - Uses default that references `probes/results/` (works via symlink)

### Evaluation Scripts
- `eval_probes_on_conversations.py` - Examples reference `results/emotion_probes_*` (works via symlink)
- `eval_multiseed_fast.py` - Default: `results/emotion_probes_multiseed` (works via symlink)
- `eval_multiseed_probes.py` - Default: `results/emotion_probes_multiseed` (works via symlink)
- `run_emo_lens_probe_experiment.py` - Hardcoded: `results/emotion_probes_multiseed` (works via symlink)
- `analyze_pc_meanings_and_weights.py` - Hardcoded absolute paths (works via symlink)
- `compare_regularization_methods.py` - Hardcoded: `results/emotion_probes_*` (works via symlink)

### Steering Scripts
- `test_emotion_steering.py` - Hardcoded: `results/emotion_probes_top{n}` (works via symlink)
- ✅ `test_orthogonal_emotion_steering.py` - Path resolution fixed

### Visualization Scripts
- `generate_all_visualizations.py` - Hardcoded dict: `results/emotion_probes_top{3,5,10}` (works via symlink)
- `visualize_pc_weights.py` - Hardcoded absolute path (works via symlink)

## Migration Strategy

### Phase 1: Symlinks (COMPLETED ✅)
- Created symlinks from new organized structure to existing data locations
- All old paths continue to work
- No script changes required immediately

### Phase 2: Gradual Updates (OPTIONAL)
Scripts can be updated incrementally to use `output_config.py`:

```python
# Before
output_dir = "results/emotion_probes_top10"

# After
from probes.output_config import OutputPaths
output_dir = OutputPaths.Probes.TEXT_CPCA_TOP10
```

### Phase 3: Cleanup (FUTURE)
Once confident all scripts work:
- Move actual data to new structure
- Remove old directories
- Update symlinks or remove them

## Testing Checklist

Test that key workflows still work:

- [ ] Train text-based probe: `python -m probes.scripts.training.train_emotion_probe --layer 30 --use-cpca --n-components 10`
- [ ] Train conversation probe: `python -m probes.scripts.training.train_conversation_probe --layer 30`
- [ ] Evaluate probe: `python -m probes.scripts.evaluation.eval_probes_on_conversations --probe-dir results/emotion_probes_top10`
- [ ] Run steering test: Import test_emotion_steering.py and verify paths resolve
- [ ] Generate visualizations: Check plots are created

## Notes

- Symlinks ensure **100% backward compatibility**
- Scripts don't need immediate updates
- New scripts can use `output_config.py` for organized paths
- Documentation updated to reference new structure

---

Last updated: 2025-12-27
