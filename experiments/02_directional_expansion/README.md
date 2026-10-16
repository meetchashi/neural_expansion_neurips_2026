# Directional GI expansion

**Paper:** Figure 2(a); Appendix C.

Compare early and late checkpoint GI profiles along the penultimate-feature Jacobian right singular vectors. The CLI also measures a matched random-direction control.

## Inputs and results

Train the MNIST MLP with `configs/training/mnist_mlp.yaml` first. Run the early and final checkpoint commands to calculate directional GI profiles.

Each output directory contains `measurements.csv`, `profile.png`, and `run.json`. CSV rows identify the image, direction type, RSV index, singular value and measured GI. Use the corresponding training config and weights to explore CNNs, ViTs or CIFAR-10.

Run all commands from the repository root after installation.

## Run

```bash
for cp in model_epoch0_batch399 model_epoch49_batch399; do
  python experiments/02_directional_expansion/run.py --checkpoint checkpoints/mnist_original_mlp/$cp.pth \
    --k 5 --directions 784 --scale 56 --output results/02_directional_expansion/$cp
done
```
