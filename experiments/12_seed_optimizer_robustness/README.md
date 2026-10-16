# Seeds and optimizer robustness

**Paper:** Table 5; Appendix F.2.

Train 16 models: two architectures × (five SGD seeds 0–4 plus SGD/Adam/RMSprop at seed 42). Evaluate a fixed 50-image test subset across training.

## Inputs and results

Run the sweep to train and analyze 16 models: two architectures, each with five SGD seeds and three optimizers at seed 42. Settings: 20 epochs, learning rate 0.001, fixed 50-image test subset. MLP evaluation checkpoints are 0,5,10,15,20 with step 0.01 and max epsilon 3. CNN checkpoints are 0,10,20 with step 0.02 and max epsilon 1.5.

Results: generated configs beside the study; weights under `checkpoints/seed_optimizer_robustness/`; per-checkpoint CSVs, `summary.csv`, variance tables, significance tests and plots under the study `results/` directory. Use a fresh results directory for each complete sweep.

Run all commands from the repository root after installation.

## Run

```bash
python experiments/12_seed_optimizer_robustness/run_all.py --arch all
```
