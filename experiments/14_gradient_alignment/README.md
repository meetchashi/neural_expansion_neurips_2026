# Gradient alignment with Jacobian RSVs

**Paper:** Figure 10; Appendix I.3.

Project input-loss gradients and natural image differences onto the ordered penultimate-feature Jacobian RSVs. Report cumulative squared projection energy divided by the vector’s total squared norm.

## Inputs and results

Train the MNIST MLP first; the CIFAR-10 command downloads ResNet20 weights. The commands project input-loss gradients and image differences onto the penultimate-feature Jacobian RSVs. Image pairs use the selected test image and the next training-subset index.

Results: `measurements.csv` records cumulative squared projection energy by sample, vector type and RSV index. `profile.png` compares mean profiles; `run.json` records the settings.

Run all commands from the repository root after installation.

## Run

```bash
python experiments/14_gradient_alignment/run.py --checkpoint checkpoints/mnist_original_mlp/best_model_mnist.pth \
  --k 2 --directions 784 --output results/14_gradient_alignment/mnist
python experiments/14_gradient_alignment/run.py --pretrained resnet20 --data-path data/CIFAR10 \
  --k 2 --directions 3072 --output results/14_gradient_alignment/cifar10
```
