# Orthogonal tube stability

**Paper:** Table 6; Appendix F.3.

Sample 100 previously covered rays on MNIST CNN and Fashion-MNIST CNN. Probe five positions along each training-to-test ray, using 20 random Gram–Schmidt orthogonal directions per position.

## Inputs and results

Run experiment 11 first to generate the trained all-class GI CSVs. Use the same trained weights for the tube commands. Each model selects 100 covered rays and probes five positions using 20 perpendicular directions per position. Settings: orthogonal scale 56, step 0.05, maximum epsilon 16.

Results: `ray_pool.csv` stores image-pair IDs and hop distances; `tube_gi.csv` stores directional measurements and `search_max_eps`; `analysis/` contains tube/hop statistics, CSV summaries and profiles.

Run all commands from the repository root after installation.

## Run

```bash
# First run experiment 11; its trained all-class CSV supplies the covered rays.
python experiments/13_orthogonal_tubes/run.py --config configs/training/mnist_cnn.yaml \
  --checkpoint checkpoints/mnist_original_cnn/best_model_mnist.pth \
  --data-path data/MNIST-Original \
  --gi-csv results/11_discriminative_baselines/mnist_cnn/trained/GI_all_classes_experiment_outputs_cnn/test_to_all_classes_gi_coverage.csv \
  --output results/13_orthogonal_tubes/mnist_cnn
python experiments/13_orthogonal_tubes/run.py --config configs/training/fashion_mnist_cnn.yaml \
  --checkpoint checkpoints/fashion_mnist_cnn/best_model_fashionmnist.pth \
  --data-path data/FashionMNIST-Original \
  --gi-csv results/11_discriminative_baselines/fashionmnist_cnn/trained/GI_all_classes_experiment_outputs_cnn/test_to_all_classes_gi_coverage.csv \
  --output results/13_orthogonal_tubes/fashionmnist_cnn
```
