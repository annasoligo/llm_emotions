#!/bin/bash
# Preprocess All Dashboard Subsets
#
# Applies 5 probe configurations to each of the 5 extracted subsets:
# - orthogonal_raw (conversation, raw activations)
# - orthogonal_cpca_top10 (conversation, cPCA top 10)
# - text_raw (text, raw activations)
# - text_cpca (text, cPCA top 10)
# - centroid_k10 (conversation, k=10 centroid)
#
# Estimated time: ~2 hours total

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
echo "DASHBOARD SUBSET PREPROCESSING"
echo "========================================="
echo "Probe configurations: $PROBES"
echo "Total subsets: 5"
echo "Estimated time: ~2 hours"
echo ""

for SUBSET in high_emotion_6plus mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown; do
  echo "==============================================="
  echo "Processing: $SUBSET"
  echo "==============================================="

  START_TIME=$(date +%s)

  python data_preprocessing.py \
    --input ../elicitation/outputs/dashboard_subsets/${SUBSET}.jsonl \
    --output data/${SUBSET}.pkl \
    --probes $PROBES

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  MINUTES=$((ELAPSED / 60))
  SECONDS=$((ELAPSED % 60))

  echo ""
  echo "✓ Completed: $SUBSET (${MINUTES}m ${SECONDS}s)"
  echo ""
done

echo "========================================="
echo "ALL SUBSETS PREPROCESSED"
echo "========================================="
echo ""
echo "Output files:"
ls -lh data/*.pkl

echo ""
echo "Verification:"
for FILE in high_emotion_6plus.pkl mid_emotion_3to5.pkl low_emotion_0to2.pkl low_emotion_no_shutdown.pkl low_emotion_with_shutdown.pkl; do
  if [ -f "data/$FILE" ]; then
    SIZE=$(du -h "data/$FILE" | cut -f1)
    echo "✓ $FILE ($SIZE)"
  else
    echo "✗ MISSING: $FILE"
  fi
done
