# Adversarial examples and ray coverage

Generate adversarial images and measure their training-to-target ray coverage.

**Paper:** Figure 5; Appendix I.3.

## Inputs and results

Train the MNIST MLP first. Pretrained CIFAR-10 and ImageNet models download their weights; ImageNette is downloaded separately.

The pixel-space workflow writes `coverage.csv`, `summary.csv`, `coverage.png`, and `run.json` for each dataset/attack combination. Settings: IFGSM epsilon 0.03, alpha 0.005, 10 iterations; PGD epsilon 0.03, alpha 0.005, 30 iterations; CW fixed c=0.01, Adam learning rate 0.01, 1000 iterations.

The source-compatible workflow runs PNG generation followed by PNG-based GI evaluation. It writes `class_<id>/sample_*.png`, sample metadata, attack reports, `correct_gi.csv`, `wrong_gi.csv`, and `run.json`. Use an empty output directory for each run. `--k` selects candidates per class; `--max-train` sets the anchor count per class. The two entry points select their respective preprocessing and attack implementations.

Run all commands from the repository root after installation.

## Pixel-space attacks

```bash
python -m src.training.mnist_train --config configs/training/mnist_mlp.yaml
for attack in cw ifgsm pgd; do
  python experiments/07_adversarial/run.py --checkpoint checkpoints/mnist_original_mlp/best_model_mnist.pth \
    --attack $attack --k 20 --pool-size 0 --output results/07_adversarial/pixel_space/mnist_$attack
  python experiments/07_adversarial/run.py --pretrained resnet20 --data-path data/CIFAR10 \
    --attack $attack --k 10 --pool-size 200 --output results/07_adversarial/pixel_space/cifar10_$attack
  python experiments/07_adversarial/run.py --pretrained vit_base --data-path data/imagenette2 \
    --attack $attack --k 5 --pool-size 100 --output results/07_adversarial/pixel_space/imagenette_$attack
done
```

## Source-compatible attack pipeline

```bash
python -m src.training.mnist_train --config configs/training/mnist_mlp.yaml
for attack in cw ifgsm pgd; do
  python experiments/07_adversarial/run_historical.py --dataset mnist \
    --model checkpoints/mnist_original_mlp/best_model_mnist.pth \
    --data-path data/MNIST-Original --attack $attack --k 20 --max-train 60000 \
    --output results/07_adversarial/historical/mnist_$attack
  python experiments/07_adversarial/run_historical.py --dataset cifar10 \
    --model resnet20 --data-path data/CIFAR10 --attack $attack --k 10 --max-train 200 \
    --output results/07_adversarial/historical/cifar10_$attack
  python experiments/07_adversarial/run_historical.py --dataset imagenette \
    --model ViT-Base --data-path data/imagenette2 --attack $attack --k 5 --max-train 100 \
    --output results/07_adversarial/historical/imagenette_$attack
done
```
