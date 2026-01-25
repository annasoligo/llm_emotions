# Activation Collection Location

This directory is located at:
```
/workspace-vast/annas/git/research-tools/steering_tests/activation_collection/
```

All paths in scripts are relative to the research-tools root directory.

## Running Collections

From the research-tools root:
```bash
# Submit SLURM job
sbatch steering_tests/activation_collection/slurm_examples/emotion_prompts.sh

# Or run directly
python steering_tests/activation_collection/collect.py \
  --input steering_tests/data/emotion_prompts_MODEL_500.jsonl \
  --output steering_tests/activations/emotion_prompts_gemma2_9b \
  --model google/gemma-2-9b-it \
  --mode chat
```

## Output Location

All activations save to:
```
steering_tests/activations/
├── emotion_prompts_gemma2_9b/
│   ├── layer_00.pkl
│   ├── layer_01.pkl
│   ├── ...
│   └── metadata.json
└── appraisal/
    ├── scenarios_google_gemma_2_9b_it/
    └── paraphrases_google_gemma_2_9b_it/
```
