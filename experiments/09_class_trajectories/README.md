# Class trajectories along train-to-test paths

**Paper:** Figures 8–9; Appendix G.

Plot predicted class along x_train + epsilon (x_test − x_train), including extrapolation beyond the test point. The test image is reached at epsilon=1; uninterrupted class paths correspond to covered rays.

## Inputs and results

Train the MNIST MLP first. The command evaluates paths from same-label training anchors to selected test images, using epsilon=1 for the test endpoint.

Results: `trajectories.csv` records epsilon and predicted class for every sampled path position. `path_<sample>_<anchor>.png` plots each selected path.

Run all commands from the repository root after installation.

## Run

```bash
python experiments/09_class_trajectories/run.py --checkpoint checkpoints/mnist_original_mlp/best_model_mnist.pth \
  --k 2 --pool-size 5 --step-size 0.01 --max-eps 3 --output results/09_class_trajectories
```
