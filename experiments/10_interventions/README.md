# Causal ray-coverage interventions

**Paper:** Table 1; Appendix H.

Compare cross-entropy control, expansion, and restriction using the original auxiliary losses. Each MNIST MLP and Fashion-MNIST CNN arm runs for 10 epochs with seeds 42, 43 and 44.

## Inputs and results

Run the two shell commands to train control, expansion and restriction arms for both models. Settings: seeds 42, 43 and 44; 10 epochs; lambda 0.3; tau 2.5; perturbation fraction 0.08; three-epoch lambda warmup; SGD learning rate 0.001.

Each model subfolder receives `results/full_sweep/*.json`, saved weights, and `results/analysis/` with final-epoch statistics, significance tests and plots. GI evaluation uses a fixed set of 200 training pairs.

Run all commands from the repository root after installation.

## Run

```bash
bash experiments/10_interventions/mnist_mlp/run_all.sh
bash experiments/10_interventions/fashionmnist_cnn/run_all.sh
```
