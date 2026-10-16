# Linear approximation along RSVs

**Paper:** Figure 2(b); Appendix D.

Measure the first alpha on a fixed grid where the cosine similarity between actual and first-order approximated penultimate features falls below 0.95. Compare initialization and a trained checkpoint on identical sampled test images.

## Inputs and results

Train the MNIST MLP first. The grid-search commands generate `measurements.csv`, `profile.png`, and `run.json` for initialization and the final checkpoint. Settings: penultimate features, cosine threshold 0.95, direction scale 56, alpha range 0–20 with 100 grid points.

For binary search, run experiment 01 first, then `run_binary_search.py`. This writes `results/03_linearity/binary_search_config.yaml` and saves `linear_approximation_thresholds.npz` and `linear_approximation_summary.json` beside each selected Jacobian. Binary search uses unit RSVs, maximum alpha 111 and tolerance 0.01.

Run all commands from the repository root after installation.

## Run

```bash
for cp in model_epoch0_batch0 model_epoch49_batch399; do
  python experiments/03_linearity/run.py --checkpoint checkpoints/mnist_original_mlp/$cp.pth \
    --k 5 --directions 784 --scale 56 --cosine-threshold 0.95 \
    --max-alpha 20 --alpha-steps 100 --output results/03_linearity/$cp
done
```

## Binary-search analysis

```bash
# Run experiment 01 first to generate its effective config, samples and Jacobians.
python experiments/03_linearity/run_binary_search.py \
  --config results/01_gi_evolution/effective_config.yaml
```
