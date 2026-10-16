# Random-label learning

**Paper:** Figure 4(left), Table 7; Appendix I.1.

Train an MNIST MLP with random supervision, then evaluate ray coverage to training images carrying its predicted label. RANDOM-10 randomizes training labels only; RANDOM-100 randomizes both train and test labels.

## Inputs and results

Run each training command before its coverage command. RANDOM-10 uses 10 output classes and random training labels. RANDOM-100 uses 100 output classes and random train/test labels. Training labels use seed 42; RANDOM-100 test labels use seed 43.

Training writes model weights, learning curves, split indices and the effective config. Coverage runs write `coverage.csv`, `summary.csv`, `coverage.png`, and `run.json` under the selected output directory.

Run all commands from the repository root after installation.

## Run

```bash
for variant in random10 random100; do
  python -m src.training.mnist_train --config configs/training/mnist_${variant}_mlp.yaml
  python experiments/05_random_labels/run.py --config configs/training/mnist_${variant}_mlp.yaml \
    --checkpoint checkpoints/mnist_${variant}_mlp/best_model_mnist.pth \
    --k 0 --pool-size 0 --output results/05_random_labels/$variant
done
```
