# Parallelization Analysis: Preprocessing by Probe

## Current Sequential Pipeline

**Total time:** ~2 hours (24 min per subset × 5 subsets)

**Breakdown per subset (12 conversations):**
- Model loading: 2 min (once)
- Baseline computation: 25 min (5 min × 5 probes)
- Probe loading: 5 min (1 min × 5 probes)
- Activation extraction: 5 min (0.4 min × 12 conversations, shared)
- Probe application: 5 min (0.4 min × 12 conversations × 5 probes)
- **Total: ~24 minutes per subset**

---

## Parallelization Options

### Option A: Split by Probe (Recommended) ⭐

**Strategy:** One job per probe, processes all 5 subsets sequentially

**Jobs:** 5 parallel jobs (one per probe)

**Per-job work:**
```
Job 1: orthogonal_raw on all 5 subsets
Job 2: orthogonal_cpca_top10 on all 5 subsets
Job 3: text_raw on all 5 subsets
Job 4: text_cpca on all 5 subsets
Job 5: centroid_k10 on all 5 subsets
```

**Time breakdown per job:**
- Model loading: 2 min (once)
- Baseline computation: 5 min (once, for this probe)
- Probe loading: 1 min (once)
- Process 5 subsets × 12 conversations: 30 min
- **Total per job: ~38 minutes**

**Wall-clock time:** 38 minutes (all jobs run in parallel)

**Speedup:** 2 hours → 38 min = **3.2× faster**

**Post-processing:** Merge 5 probe results into 5 final `.pkl` files

---

### Option B: Split by Subset (Simpler)

**Strategy:** One job per subset, applies all 5 probes

**Jobs:** 5 parallel jobs (one per subset)

**Per-job work:**
```
Job 1: high_emotion_6plus (all 5 probes)
Job 2: mid_emotion_3to5 (all 5 probes)
Job 3: low_emotion_0to2 (all 5 probes)
Job 4: low_emotion_no_shutdown (all 5 probes)
Job 5: low_emotion_with_shutdown (all 5 probes)
```

**Time breakdown per job:**
- Model loading: 2 min
- Baseline computation: 25 min (5 min × 5 probes)
- Probe loading: 5 min
- Process 12 conversations × 5 probes: 5 min
- **Total per job: ~24 minutes**

**Wall-clock time:** 24 minutes (all jobs run in parallel)

**Speedup:** 2 hours → 24 min = **5× faster** ✨

**Post-processing:** None needed (each job produces final `.pkl`)

---

### Option C: Two-Stage Pipeline (Most Efficient) ⭐⭐

**Strategy:**
1. **Stage 1:** Extract and cache activations once
2. **Stage 2:** 25 parallel jobs (5 probes × 5 subsets) read cached activations

**Stage 1 (Sequential):**
- Model loading: 2 min
- Extract activations for all 60 conversations: 6 min
- Save activation cache: 1 min
- **Total: ~9 minutes**

**Stage 2 (Parallel, 25 jobs):**
- Each job: Load probe (10s) + Apply to 12 conversations (2 min) = ~2.5 min per job
- **Wall-clock: ~3 minutes** (all jobs run in parallel)

**Total wall-clock time:** 9 + 3 = **12 minutes**

**Speedup:** 2 hours → 12 min = **10× faster** 🚀

**Post-processing:** Merge 5 probe results into 5 final `.pkl` files

---

## Recommendation

### For Simplicity: **Option B** (Split by Subset)
- Easiest to implement (minimal changes to existing code)
- No post-processing needed
- Good speedup (5×)
- Just run 5 jobs in parallel

### For Maximum Speed: **Option C** (Two-Stage with Cached Activations)
- Requires refactoring to separate activation extraction
- Needs disk space for activation cache (~2 GB)
- Best speedup (10×)
- More complex workflow

---

## Implementation: Option B (Recommended for Now)

### Batch Script with Parallel Execution

```bash
#!/bin/bash
# preprocess_all_subsets_parallel.sh

cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

echo "========================================="
echo "PARALLEL PREPROCESSING (5 jobs)"
echo "========================================="

# Launch all 5 jobs in parallel
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes $PROBES &

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/mid_emotion_3to5.jsonl \
  --output data/mid_emotion_3to5.pkl \
  --probes $PROBES &

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes $PROBES &

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_no_shutdown.jsonl \
  --output data/low_emotion_no_shutdown.pkl \
  --probes $PROBES &

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_with_shutdown.jsonl \
  --output data/low_emotion_with_shutdown.pkl \
  --probes $PROBES &

# Wait for all jobs to complete
wait

echo ""
echo "========================================="
echo "ALL JOBS COMPLETED"
echo "========================================="
ls -lh data/*.pkl
```

**Usage:**
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
chmod +x preprocess_all_subsets_parallel.sh
./preprocess_all_subsets_parallel.sh
```

**Time:** ~24 minutes (vs 2 hours sequential)

---

## Resource Requirements

### Sequential (Current)
- GPU: 1× Gemma 27B (~40 GB VRAM)
- Time: 2 hours

### Option B (Parallel by Subset)
- GPUs: 5× Gemma 27B (need 5 GPUs or time-slice on 1 GPU)
- Time: 24 minutes on 5 GPUs, or ~2 hours on 1 GPU (time-sliced)

### Option C (Two-Stage)
- GPUs: 1× for stage 1, then can use multiple GPUs/CPUs for stage 2
- Disk: ~2 GB for activation cache
- Time: 12 minutes (9 min stage 1 on 1 GPU, 3 min stage 2 on multiple GPUs)

---

## GPU Availability Check

If you have **5 GPUs available**, use **Option B** (parallel by subset):
```bash
./preprocess_all_subsets_parallel.sh
```

If you have **1 GPU**, options:
1. Run sequential (current script): 2 hours
2. Implement **Option C** (two-stage): 12 minutes total, only needs 1 GPU for most of the time

---

## Next Steps

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```

2. **If 5+ GPUs:** Use parallel script (Option B)
   ```bash
   ./preprocess_all_subsets_parallel.sh
   ```

3. **If 1 GPU:**
   - Quick: Use sequential script (2 hours)
   - Optimal: Implement two-stage pipeline (12 minutes, but needs code refactoring)

Let me know which approach you'd like and I can create the implementation!
