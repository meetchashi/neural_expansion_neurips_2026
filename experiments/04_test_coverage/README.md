# Test-to-train ray coverage

**Paper:** Figure 3; Appendix E and Table 2.

Evaluate rays anchored at training images and directed toward test images, using the test prediction as the reference class. Report the fraction with GI > 1 per test image and separate correct and incorrect predictions.

## Inputs and results

Train custom MNIST models first; pretrained model commands download their weights. Download ImageNette separately and point `--data-path` at its extraction.

Each run writes `coverage.csv`, `summary.csv`, `coverage.png`, and `run.json`. `coverage.csv` records the sample ID, label, prediction, pair count, mean GI, percentage of GIs above 1, and measurement status. Add `--save-pairs` to save individual GI arrays. MNIST commands use all test images and all anchors in the predicted-class training pool; CIFAR-10 uses 50 test images per class and 200 anchors; ImageNette uses 20 test images per class and 100 anchors.

Run all commands from the repository root after installation.

## Run

```bash
# MNIST: every test image, all anchors in its predicted class.
for model in mlp cnn vit; do
  python experiments/04_test_coverage/run.py --config configs/training/mnist_$model.yaml \
    --checkpoint checkpoints/mnist_original_$model/best_model_mnist.pth \
    --k 0 --pool-size 0 --batch-size 6000 --output results/04_test_coverage/mnist_$model
done
# CIFAR-10: 50 images per true class and 200 randomly sampled anchors.
for model in resnet20 vgg16_bn; do
  python experiments/04_test_coverage/run.py --pretrained $model --data-path data/CIFAR10 \
    --k 50 --pool-size 200 --batch-size 200 --output results/04_test_coverage/cifar10_$model
done
# ImageNette: 20 images/class, 100 anchors; retain the full 1000-class head.
for model in vgg16 resnet50 vit_base deit_small; do
  python experiments/04_test_coverage/run.py --pretrained $model --data-path data/imagenette2 \
    --k 20 --pool-size 100 --batch-size 100 --output results/04_test_coverage/imagenette_$model
done
```
