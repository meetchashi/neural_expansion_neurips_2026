#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/experiments/10_interventions/mnist_mlp"
for seed in 42 43 44; do
  for arm in control expand restrict; do
    python train_intervention.py --arm "$arm" --seed "$seed" --epochs 10 \
      --lam 0.3 --tau 2.5 --eps_frac 0.08 --lam_warmup_epochs 3 \
      --gi_checkpoints 0,1,2,3,4,5,6,7,8,9,10 \
      --output_dir results/full_sweep --data_path "$ROOT/data/MNIST-Original"
  done
done
python aggregate_and_analyze.py
