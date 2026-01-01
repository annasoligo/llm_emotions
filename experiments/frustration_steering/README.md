# Frustration Steering Experiment

Clean, minimal implementation of steering/ablation/capping experiment on frustration puzzles.

## What Was Implemented

### Infrastructure (probes/steering/)
- **baseline_stats.py**: Load baseline statistics for principled magnitude calculation
- **probe_loader.py**: Load emotion probes and create steering builders
- **model.py**: Extended with ablation and capping hooks
  - `add_ablation()`: Project out emotion directions
  - `add_capping()`: Clamp projections above threshold
  - `apply_intervention()`: Unified hook application

### Experiment (experiments/frustration_steering/)
- **config.py**: All configuration (7 conditions, paths, hyperparameters)
- **select_puzzles.py**: Load highest-rated frustration puzzle
- **generate_steered_responses.py**: Main generation script

## Experimental Conditions

1. **baseline**: No intervention
2. **capping**: Cap anger/fear/sadness to 0.5 std above mean
3. **ablation**: Project out anger/fear/sadness directions
4. **steer_anger**: Steer with anger at +1 std
5. **steer_fear**: Steer with fear at +1 std
6. **steer_sadness**: Steer with sadness at +1 std
7. **steer_happiness**: Steer with happiness at +1 std

## How to Run

### Full Experiment (350 responses)
```bash
cd /workspace-vast/annas/git/research-tools/experiments/frustration_steering
python generate_steered_responses.py
```

### Quick Test (modify config.py first)
Change `NUM_RESPONSES_PER_CONDITION = 50` to `2` for quick testing.

## Output

Results saved to: `experiments/frustration_steering/outputs/steered_responses.jsonl`

Format (Stage 3 compatible):
```json
{
  "response_id": "uuid",
  "request_id": "puzzle_2",
  "prefix_type": "steering_capping",
  "response": "Generated text...",
  "response_index": 0,
  "generation_timestamp": "2026-01-01T...",
  "generation_params": {
    "model": "google/gemma-3-27b-it",
    "steering_condition": "capping",
    "intervention_type": "capping",
    "puzzle_rating": 9
  }
}
```

## Verified Components

✓ Baseline stats loading (Layer 30: mean=9.96, std=576.98)
✓ Probe loading (7 emotions including neutral)
✓ Puzzle selection (Puzzle 2, rating 9)
✓ Vector normalization (unit norm vectors)
✓ Magnitude calculation (1 std = 576.98, cap threshold = 298.45)

## Key Details

- **Puzzle**: Single highest-rated puzzle (rating 9)
- **Model**: Gemma-3-27B-it (bfloat16, cuda)
- **Layer**: 30 (mid-to-late layer)
- **Responses**: 50 per condition × 7 conditions = 350 total
- **Temperature**: 1.0 (high diversity)
- **Max tokens**: 2000
- **Baseline aggregation**: first_assistant_token

## Implementation Notes

- No fallbacks: Errors raise immediately
- No error handling: Clean code, fail fast
- Minimal dependencies: Reuses existing steering infrastructure
- Stage 3 compatible: Output format works with existing grading pipeline

## Next Steps

After generation:
1. Run Stage 4 judging on outputs (if needed)
2. Analyze emotional expression by condition
3. Compare frustration levels across interventions
4. Measure effectiveness of capping vs ablation vs steering
