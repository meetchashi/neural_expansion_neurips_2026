# Untrained/trained × same/different-class controls

**Paper:** Table 4; Appendix F.1.

Evaluate 100 fixed test examples for each MNIST MLP, MNIST CNN and Fashion-MNIST CNN. Different-class rays hold the test prediction fixed while sampling three wrong-label anchor pools; this preserves the original control definition.

## Inputs and results

Train the MNIST MLP and CNN first. The commands select initialization and validation-best weights; the Fashion-MNIST training command is included below. Evaluation uses 100 test images per model and three wrong-class pools per test image.

Results: all-class GI CSVs, wrong-class coverage CSVs, four-cell summaries, plots and Mann–Whitney tests. All-class CSVs include global training IDs and supply the input for experiment 13.

Run all commands from the repository root after installation.

## Run

```bash
for model in mlp cnn; do
  python experiments/11_discriminative_baselines/run.py --config configs/training/mnist_$model.yaml \
    --untrained checkpoints/mnist_original_$model/model_epoch0_batch0.pth \
    --trained checkpoints/mnist_original_$model/best_model_mnist.pth \
    --data-path data/MNIST-Original --output results/11_discriminative_baselines/mnist_$model
done
python -m src.training.mnist_train --config configs/training/fashion_mnist_cnn.yaml
python experiments/11_discriminative_baselines/run.py --dataset fashionmnist \
  --config configs/training/fashion_mnist_cnn.yaml \
  --untrained checkpoints/fashion_mnist_cnn/model_epoch0_batch0.pth \
  --trained checkpoints/fashion_mnist_cnn/best_model_fashionmnist.pth \
  --data-path data/FashionMNIST-Original --output results/11_discriminative_baselines/fashionmnist_cnn
```
