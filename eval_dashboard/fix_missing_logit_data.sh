#!/bin/bash
#SBATCH --job-name=fix_logit
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/fix_logit_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

DATA_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard/data"

echo "================================"
echo "FIXING MISSING LOGIT LENS DATA"
echo "================================"

# Step 1: Add L20-30 to high_emotion_6plus
echo ""
echo "Step 1: Adding L20-30 to high_emotion_6plus..."
python eval_dashboard/add_logit_lens_layers.py "$DATA_DIR/high_emotion_6plus.pkl" --layer_start 20 --layer_end 30
echo "✓ Completed high_emotion_6plus L20-30"

# Step 2: Add L30-40 to the four files missing it (run sequentially to avoid conflicts)
echo ""
echo "Step 2: Adding L30-40 to files missing it..."

for file in mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown; do
    echo ""
    echo "  Processing $file.pkl (L30-40)..."
    python eval_dashboard/add_logit_lens_layers.py "$DATA_DIR/${file}.pkl" --layer_start 30 --layer_end 40
    echo "  ✓ Completed $file.pkl"
done

echo ""
echo "================================"
echo "ALL MISSING DATA ADDED!"
echo "================================"

# Verify coverage
echo ""
echo "Verifying data coverage..."
python3 << 'EOF'
import pickle
from pathlib import Path

DATA_DIR = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
files = ['high_emotion_6plus', 'mid_emotion_3to5', 'low_emotion_0to2',
         'low_emotion_no_shutdown', 'low_emotion_with_shutdown', 'baseline_v12_solvable']

print("\nFinal logit lens coverage:")
print("="*80)
all_complete = True
for file in files:
    pkl_path = DATA_DIR / f"{file}.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        keys = list(data['conversations'][0]['probe_scores'].keys())

        has_l40_50 = 'logit_lens_mean' in keys
        has_l30_40 = 'logit_lens_mean_l30_40' in keys
        has_l20_30 = 'logit_lens_mean_l20_30' in keys

        complete = has_l40_50 and has_l30_40 and has_l20_30
        status = "✓ COMPLETE" if complete else "✗ INCOMPLETE"
        if not complete:
            all_complete = False
            missing = []
            if not has_l40_50: missing.append("L40-50")
            if not has_l30_40: missing.append("L30-40")
            if not has_l20_30: missing.append("L20-30")
            status += f" (missing: {', '.join(missing)})"

        print(f"{file:30} {status}")

print("="*80)
if all_complete:
    print("✓ All files have complete logit lens data!")
else:
    print("✗ Some files still have missing data")
EOF
