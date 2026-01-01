#!/bin/bash
# Parallel Preprocessing - One Job Per Subset
#
# Launches 5 jobs in parallel (one per subset), each applying all 5 probes.
# Requires 5 GPUs or will time-slice on available GPUs.
#
# Time: ~24 minutes on 5 GPUs, ~2 hours on 1 GPU (time-sliced)

# Activate virtual environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
else
   echo "ERROR: Virtual environment not found at .venv/bin/activate"
   exit 1
fi

cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

echo "========================================="
echo "PARALLEL PREPROCESSING (5 jobs)"
echo "========================================="
echo "Probe configurations: $PROBES"
echo "Starting parallel jobs..."
echo ""

# Check GPU availability
NGPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $NGPUS"
echo ""

if [ $NGPUS -ge 5 ]; then
  echo "✓ Sufficient GPUs for full parallelization"
  echo "  Expected time: ~24 minutes"
elif [ $NGPUS -eq 1 ]; then
  echo "⚠ Only 1 GPU - jobs will time-slice"
  echo "  Expected time: ~2 hours (similar to sequential)"
else
  echo "⚠ $NGPUS GPUs - jobs will partially parallelize"
  echo "  Expected time: ~$((120 / $NGPUS)) minutes"
fi

echo ""
echo "Launching jobs..."

# Launch all 5 jobs in parallel (background with &)
python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl \
  --output data/high_emotion_6plus.pkl \
  --probes $PROBES > logs/high_emotion_6plus.log 2>&1 &
PID1=$!
echo "  [PID $PID1] high_emotion_6plus"

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/mid_emotion_3to5.jsonl \
  --output data/mid_emotion_3to5.pkl \
  --probes $PROBES > logs/mid_emotion_3to5.log 2>&1 &
PID2=$!
echo "  [PID $PID2] mid_emotion_3to5"

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_0to2.jsonl \
  --output data/low_emotion_0to2.pkl \
  --probes $PROBES > logs/low_emotion_0to2.log 2>&1 &
PID3=$!
echo "  [PID $PID3] low_emotion_0to2"

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_no_shutdown.jsonl \
  --output data/low_emotion_no_shutdown.pkl \
  --probes $PROBES > logs/low_emotion_no_shutdown.log 2>&1 &
PID4=$!
echo "  [PID $PID4] low_emotion_no_shutdown"

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/low_emotion_with_shutdown.jsonl \
  --output data/low_emotion_with_shutdown.pkl \
  --probes $PROBES > logs/low_emotion_with_shutdown.log 2>&1 &
PID5=$!
echo "  [PID $PID5] low_emotion_with_shutdown"

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Monitor progress: tail -f logs/<subset>.log"
echo ""

START_TIME=$(date +%s)

# Wait for all background jobs
wait $PID1 $PID2 $PID3 $PID4 $PID5

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "ALL JOBS COMPLETED"
echo "========================================="
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""

# Check outputs
echo "Output files:"
ls -lh data/*.pkl 2>/dev/null

echo ""
echo "Verification:"
for FILE in high_emotion_6plus.pkl mid_emotion_3to5.pkl low_emotion_0to2.pkl low_emotion_no_shutdown.pkl low_emotion_with_shutdown.pkl; do
  if [ -f "data/$FILE" ]; then
    SIZE=$(du -h "data/$FILE" | cut -f1)
    echo "✓ $FILE ($SIZE)"
  else
    echo "✗ MISSING: $FILE (check logs/$(basename $FILE .pkl).log for errors)"
  fi
done
