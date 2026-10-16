# Out-of-distribution ray coverage

**Paper:** Figure 11; Appendix I.4.

Transform CIFAR-10 test images into 28×28 grayscale inputs, evaluate an MNIST MLP, and measure coverage from MNIST training anchors of the predicted digit.

## Inputs and results

Train the MNIST MLP first. The command downloads CIFAR-10, converts inputs to 28×28 grayscale and selects 50 images per CIFAR-10 class.

Results: `results/08_ood/coverage.csv`, `summary.csv`, `coverage.png`, and `run.json`. Use `coverage.csv` to summarize coverage across the 500 selected inputs.

Run all commands from the repository root after installation.

## Run

```bash
python experiments/08_ood/run.py --checkpoint checkpoints/mnist_original_mlp/best_model_mnist.pth \
  --ood-data-path data/CIFAR10 --k 50 --pool-size 0 --output results/08_ood
```
