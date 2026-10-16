# Experiment commands and results

Install the dependencies using the [main README](../README.md), then run commands from the repository root. Each linked guide lists prerequisites, commands and generated output files.

| Experiment | Commands and outputs |
| --- | --- |
| 01 | [GI evolution during training](../experiments/01_gi_evolution/README.md) |
| 02 | [Directional GI expansion](../experiments/02_directional_expansion/README.md) |
| 03 | [Linear approximation along RSVs](../experiments/03_linearity/README.md) |
| 04 | [Test-to-train ray coverage](../experiments/04_test_coverage/README.md) |
| 05 | [Random-label learning](../experiments/05_random_labels/README.md) |
| 06 | [Long-tailed learning](../experiments/06_long_tail/README.md) |
| 07 | [Adversarial examples and ray coverage](../experiments/07_adversarial/README.md) |
| 08 | [Out-of-distribution ray coverage](../experiments/08_ood/README.md) |
| 09 | [Class trajectories along train-to-test paths](../experiments/09_class_trajectories/README.md) |
| 10 | [Causal ray-coverage interventions](../experiments/10_interventions/README.md) |
| 11 | [Untrained/trained × same/different-class controls](../experiments/11_discriminative_baselines/README.md) |
| 12 | [Seeds and optimizer robustness](../experiments/12_seed_optimizer_robustness/README.md) |
| 13 | [Orthogonal tube stability](../experiments/13_orthogonal_tubes/README.md) |
| 14 | [Gradient alignment with Jacobian RSVs](../experiments/14_gradient_alignment/README.md) |

## Dataset and weight setup

MNIST, Fashion-MNIST and CIFAR-10 download automatically through torchvision. Download ImageNette from [fastai](https://github.com/fastai/imagenette) and supply its extraction path. Pretrained model commands download weights through Torch Hub, torchvision or timm. Custom-model training creates the checkpoint files used by subsequent analyses.

```bash
python -m src.training.mnist_train --config configs/training/mnist_mlp.yaml
python experiments/04_test_coverage/run.py \
  --config configs/training/mnist_mlp.yaml \
  --checkpoint checkpoints/mnist_original_mlp/best_model_mnist.pth \
  --k 1 --pool-size 20 --batch-size 20 --output results/mnist_mlp_coverage
```

Training writes model weights, accuracy curves, split indices and the effective config. The coverage command writes `coverage.csv`, `summary.csv`, `coverage.png` and `run.json` in `results/mnist_mlp_coverage/`.

## Output locations

| Files | Location |
| --- | --- |
| Downloaded images | `data/` or the supplied data path |
| Custom trained weights | `training.checkpoint_dir` in the selected YAML |
| Standalone analysis results | The `--output` directory |
| Intervention sweeps | Each model's `results/full_sweep/` and `results/analysis/` |
| Seed/optimizer sweep | `experiments/12_seed_optimizer_robustness/results/` |
| Saved Jacobians | Experiment config `output_dir` |
| Checkpoint GI measurements | Experiment config `gi_analysis.analysis_output_dir` |

## Run the checks

```bash
python -m unittest discover -s tests -v
```
