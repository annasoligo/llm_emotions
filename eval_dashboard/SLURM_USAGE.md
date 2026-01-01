# SLURM Usage for Dashboard Preprocessing

## Quick Start

Submit the preprocessing job array:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
sbatch slurm_preprocess_subset.sh
```

This submits 5 parallel jobs (one per subset), each gets its own GPU.

## Monitor Progress

```bash
# Check job status
./check_preprocessing_status.sh

# Or specify job ID
./check_preprocessing_status.sh 98822

# Monitor all logs in real-time
tail -f /workspace-vast/annas/logs/preprocess_dashboard_98822_*.out
```

## Expected Timeline

- **Job submission:** Immediate
- **Queue time:** 0-5 minutes (depends on cluster load)
- **Processing time:** ~20-25 minutes per subset
- **Total time:** ~25-30 minutes (all subsets run in parallel)

## What Each Job Does

Each array task (0-4) processes one subset:
- **Task 0:** high_emotion_6plus
- **Task 1:** mid_emotion_3to5
- **Task 2:** low_emotion_0to2
- **Task 3:** low_emotion_no_shutdown
- **Task 4:** low_emotion_with_shutdown

For each subset, applies 5 probe configurations:
1. orthogonal_raw
2. orthogonal_cpca_top10
3. text_raw
4. text_cpca
5. centroid_k10

## Resource Allocation

Each job gets:
- **GPU:** 1× NVIDIA GPU (any available)
- **CPUs:** 8 cores
- **RAM:** 64 GB
- **Time limit:** 1 hour
- **Partition:** general
- **QoS:** high

## Output Files

Completed preprocessing creates:
```
/workspace-vast/annas/git/research-tools/eval_dashboard/data/
├── high_emotion_6plus.pkl          (~15 MB)
├── mid_emotion_3to5.pkl            (~15 MB)
├── low_emotion_0to2.pkl            (~15 MB)
├── low_emotion_no_shutdown.pkl     (~15 MB)
└── low_emotion_with_shutdown.pkl   (~10 MB)
```

## Log Files

SLURM creates log files:
```
/workspace-vast/annas/logs/
├── preprocess_dashboard_98822_0.out   (stdout for task 0)
├── preprocess_dashboard_98822_0.err   (stderr for task 0)
├── preprocess_dashboard_98822_1.out
├── preprocess_dashboard_98822_1.err
... (5 tasks total)
```

## Common Issues

### Job Pending Too Long
```bash
# Check queue position
squeue -j <JOB_ID> -o "%.18i %.9P %.50j %.8u %.8T %.10M %.6D %R"
```

If stuck in queue, may need to adjust QoS or partition.

### Job Failed
```bash
# Check error log
cat /workspace-vast/annas/logs/preprocess_dashboard_<JOB_ID>_<TASK_ID>.err

# Common causes:
# 1. OOM: Increase --mem in slurm script
# 2. Time limit: Increase --time in slurm script
# 3. Missing data: Check input files exist
```

### Rerun Single Subset

If one subset fails, rerun just that one:

```bash
# Edit slurm script, change array line to:
#SBATCH --array=2   # Just task 2 (low_emotion_0to2)

sbatch slurm_preprocess_subset.sh
```

Or run manually:
```bash
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
cd eval_dashboard

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10
```

## Verification

After all jobs complete:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Check all outputs exist
for FILE in high_emotion_6plus.pkl mid_emotion_3to5.pkl low_emotion_0to2.pkl low_emotion_no_shutdown.pkl low_emotion_with_shutdown.pkl; do
  if [ -f "data/$FILE" ]; then
    SIZE=$(du -h "data/$FILE" | cut -f1)
    echo "✓ $FILE ($SIZE)"
  else
    echo "✗ MISSING: $FILE"
  fi
done
```

## Cancel Jobs

```bash
# Cancel entire job array
scancel 98822

# Cancel specific task
scancel 98822_2
```

## Sequential Alternative

If GPUs are limited, use sequential preprocessing:

```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
./preprocess_all_subsets.sh
```

Time: ~2 hours (processes one subset at a time on single GPU)
