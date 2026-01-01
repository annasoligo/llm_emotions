#!/bin/bash
# Submit 8 parallel jobs to test K=50,60,70,80 across all layers
# Each job covers ~7 layers

ORTHO_WEIGHT=100000.0
K_VALUES="70 80 90 100"

# Job 1: Layers 0-6
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L0-6
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_0-6_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_0-6_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 0 1 2 3 4 5 6; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 2: Layers 7-13
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L7-13
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_7-13_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_7-13_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 7 8 9 10 11 12 13; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 3: Layers 14-20
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L14-20
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_14-20_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_14-20_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 14 15 16 17 18 19 20; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 4: Layers 21-27
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L21-27
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_21-27_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_21-27_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 21 22 23 24 25 26 27; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 5: Layers 28-34
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L28-34
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_28-34_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_28-34_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 28 29 30 31 32 33 34; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 6: Layers 35-41
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L35-41
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_35-41_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_35-41_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 35 36 37 38 39 40 41; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 7: Layers 42-48
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L42-48
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_42-48_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_42-48_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 42 43 44 45 46 47 48; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

# Job 8: Layers 49-55
sbatch <<EOJOB
#!/bin/bash
#SBATCH --job-name=ortho_L49-55
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_49-55_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_all_layers_49-55_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

for LAYER in 49 50 51 52 53 54 55; do
  for K in $K_VALUES; do
    echo "=== Layer \$LAYER, K=\$K ==="
    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
      --layer \$LAYER --n-sets \$K --ortho-weight $ORTHO_WEIGHT \
      --max-epochs 300 --patience 3 --learning-rate 0.001 --batch-size 32 --seed 42 --device cuda \
      --convergence-threshold 0.01 --min-accuracy-threshold 0.25 || echo "FAILED: L\$LAYER K=\$K"
  done
done
EOJOB

echo "Submitted 8 jobs covering layers 0-55 with K=50,60,70,80"
