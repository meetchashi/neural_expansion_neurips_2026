# GI evolution during training

**Paper:** Figure 1, Figure 7, Table 3; Appendices B and N.

Track fixed samples across saved checkpoints, comparing same-class training rays, Jacobian RSVs, and random directions. MNIST RSV and random vectors share scale 56; train-to-train directions are unnormalized differences.

## Inputs and results

Run training, then collect fixed-sample Jacobians and measure GI evolution.

Results: `results/01_gi_evolution/figures/checkpoint_summary.csv`, `gi_evolution.png`, `gi_evolution.pdf`, and `coverage_evolution.png`. Jacobian arrays and per-checkpoint GI CSVs use the output directories in the experiment config. The SVD summary command writes `summary/average_singular_values_all.csv`, correct/incorrect summaries and class-specific plots under the collector output directory.

Run all commands from the repository root after installation.

## Run

```bash
python -m src.training.mnist_train --config configs/training/mnist_mlp.yaml
python experiments/01_gi_evolution/run.py --checkpoints 11
```

## Singular-value summaries

```bash
python experiments/01_gi_evolution/analyze_svd_evolution.py \
  --config results/01_gi_evolution/effective_config.yaml
```
