# Long-tailed learning

**Paper:** Figure 4(middle/right), Table 8; Appendix I.2.

Downsample the seeded MNIST training split to exactly the LT1/LT2 per-class counts from Table 8, train an MLP, and measure coverage and misclassification flows across frequency groups.

## Inputs and results

Run training and coverage for LT1 and LT2. The dataset builder samples the class counts below after the seeded training/validation split; selection indices are saved with the weights.

- LT1: `[4717,2866,2370,2015,1806,1590,1346,1034,886,730]`
- LT2: `[4716,2504,2026,1464,1096,761,383,218,93,60]`

Coverage writes `coverage.csv`, `summary.csv`, `coverage.png`, `misclassification_by_frequency.csv`, and `coverage_by_frequency.csv`. The main runner uses Table 8 frequency groups. The additional frequency-analysis commands use percentile groups and produce heatmaps, migration plots and detailed statistics in `source_frequency_analysis/`.

Run all commands from the repository root after installation.

## Run

```bash
for variant in lt1 lt2; do
  python -m src.training.mnist_train --config configs/training/mnist_${variant}_mlp.yaml
  python experiments/06_long_tail/run.py --config configs/training/mnist_${variant}_mlp.yaml \
    --checkpoint checkpoints/mnist_${variant}_mlp/best_model_mnist.pth \
    --k 0 --pool-size 0 --output results/06_long_tail/$variant
done
```

## Frequency plots

```bash
python - <<'PYTHON'
from pathlib import Path
import pandas as pd
for variant in ('lt1', 'lt2'):
    root = Path('results/06_long_tail') / variant
    frame = pd.read_csv(root / 'coverage.csv')
    folder = root / 'source_frequency_analysis'
    folder.mkdir(parents=True, exist_ok=True)
    frame.loc[frame.true_label != frame.predicted_class].rename(
        columns={'coverage': 'percent_gi_above_1'}
    ).to_csv(folder / 'misclassified_samples.csv', index=False)
PYTHON
for variant in lt1 lt2; do
  python experiments/06_long_tail/analyze_frequency.py \
    results/06_long_tail/$variant/source_frequency_analysis --distribution $variant
done
```
