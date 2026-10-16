"""Offline checks of recovered source behavior and its portable entry points."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]


def module_at(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedClass(torch.nn.Module):
    def __init__(self, label, classes=10):
        super().__init__()
        self.label, self.classes = label, classes
    def forward(self, x):
        logits = torch.zeros(len(x), self.classes, device=x.device)
        logits[:, self.label] = 1
        return logits


class RecoveredSourceTests(unittest.TestCase):
    def test_recovered_attack_functions(self):
        source = module_at('experiments/07_adversarial/historical/generate_adversarial.py')
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 10)).eval()
        x = torch.full((1, 1, 2, 2), .5)
        label = torch.tensor([1])
        for name in ['iterative_fgsm', 'pgd', 'cw_l2']:
            adv = getattr(source, name)(model, x, label, iters=3)
            self.assertTrue(torch.isfinite(adv).all())
            self.assertGreaterEqual(float(adv.min()), 0)
            self.assertLessEqual(float(adv.max()), 1)
            if name != 'cw_l2':
                self.assertLessEqual(float((adv - x).abs().max()), .030001)

    def test_recovered_png_coverage_and_imagenette_reference_convention(self):
        source = module_at('experiments/07_adversarial/historical/compute_gi_from_adversarials.py')
        for dataset, label, model in [('mnist', 0, FixedClass(0)),
                                      ('imagenette', 1, FixedClass(217, 1000))]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / f'class_{label}'
                folder.mkdir()
                channels = 1 if dataset == 'mnist' else 3
                save_image(torch.full((channels, 2, 2), .5), folder / 'sample_0.png')
                (folder / 'sample_0.json').write_text(json.dumps({'original': label, 'index': 0}))
                size = 2 if dataset == 'mnist' else 224
                train = TensorDataset(torch.full((1, channels, size, size), .5), torch.tensor([label]))
                train.targets = torch.tensor([label])  # Dataset fast path, as in MNIST/ImageFolder labels
                with contextlib.redirect_stdout(io.StringIO()):
                    correct, wrong = source.evaluate_gi_on_adversarials(model, train, temp, dataset, 'cpu', max_train=1)
                self.assertEqual(len(correct), 1)
                self.assertEqual(len(wrong), 0)
                # Preserve and expose the historical local-vs-ImageNet reference mismatch.
                self.assertEqual(correct[0]['percent_gi_above_1'], 100 if dataset == 'mnist' else 0)

    def test_historical_wrapper_passes_explicit_arguments_and_records_protocol(self):
        runner = module_at('experiments/07_adversarial/run_historical.py')
        with tempfile.TemporaryDirectory() as temp, patch.object(runner.subprocess, 'run') as run:
            out = Path(temp) / 'output'
            runner.main(['--dataset', 'mnist', '--model', 'my_weights.pth', '--data-path', 'data/MNIST-Original',
                         '--attack', 'pgd', '--k', '2', '--max-train', '10', '--output', str(out), '--device', 'cpu'])
            self.assertEqual(run.call_count, 2)
            self.assertIn('--seed', run.call_args_list[0].args[0])
            self.assertIn('--adv_dir', run.call_args_list[1].args[0])
            self.assertEqual(json.loads((out/'run.json').read_text())['protocol'], 'recovered_historical')

    def test_recovered_binary_search_for_exact_linear_features(self):
        source = module_at('experiments/03_linearity/historical_binary_search.py')
        class LinearFeatures(torch.nn.Module):
            def extract_features(self, x):
                features = x.flatten(1)
                return [x, features, features.sum(1, keepdim=True)]
        x = torch.ones(1, 1, 2, 2)
        jac = np.eye(4, dtype=np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            values = source.calculate_linear_approximation_thresholds_binary(
                LinearFeatures(), x, jac, np.eye(4, dtype=np.float32),
                max_alpha=5, batch_size=2, device='cpu')
        np.testing.assert_allclose(values, [5, 5, 5, 5])

    def test_recovered_svd_summary_reads_collector_format(self):
        source = module_at('experiments/01_gi_evolution/analyze_svd_evolution.py')
        with tempfile.TemporaryDirectory() as temp:
            for status in ['correct', 'incorrect']:
                folder = Path(temp) / 'class_0' / status / 'sample_0' / 'model_epoch0_batch0'
                folder.mkdir(parents=True)
                np.savez(folder/'svd_data.npz', S=np.array([3., 1.]))
            cfg = {'output_dir': temp, 'checkpoints': {'custom_checkpoints': ['model_epoch0_batch0.pth']},
                   'dataset': {'classes': [0]}, 'gi_analysis': {'num_rsvs': 4}}
            with contextlib.redirect_stdout(io.StringIO()):
                all_rows, correct, incorrect, _, _, summary = source.process_checkpoint_data(cfg)
            np.testing.assert_allclose(all_rows.iloc[0].to_numpy(dtype=float), [3, 1, 0, 0])
            self.assertTrue((Path(summary)/'average_singular_values_all.csv').exists())

    def test_tube_cap_fraction_uses_search_cap(self):
        source = module_at('experiments/13_orthogonal_tubes/analyze_tube_profile.py')
        rows = [{'ray_id': 0, 'probe_frac': .5, 'direction_idx': i,
                 'pos_eps': value, 'tube_radius_physical': value * 56,
                 'ray_physical_length': 6., 'ray_direction_norm': 3., 'ray_g_raw': 2.}
                for i, value in enumerate([1., 2.])]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tube.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            summary, _, _ = source.summarize_one(path, 'test')
            self.assertEqual(summary['censoring_rate'], 0.)
            frame = pd.DataFrame(rows)
            frame['search_max_eps'] = 2.
            frame.to_csv(path, index=False)
            summary, _, _ = source.summarize_one(path, 'test')
            self.assertEqual(summary['censoring_rate'], .5)

    def test_recovered_frequency_plot_handles_a_small_subset(self):
        source = module_at('experiments/06_long_tail/analyze_frequency.py')
        with tempfile.TemporaryDirectory() as temp:
            pd.DataFrame([{'true_label': 6, 'predicted_class': 0, 'percent_gi_above_1': 75.},
                          {'true_label': 6, 'predicted_class': 0, 'percent_gi_above_1': 80.}]).to_csv(
                              Path(temp)/'misclassified_samples.csv', index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                source.compute_misclassification_stats_by_class_frequency(temp, 'lt2')
            stats = pd.read_csv(Path(temp)/'lt2_gi_by_freq_stats.csv')
            self.assertEqual(stats.iloc[0]['true_freq'], 'medium')
            self.assertEqual(stats.iloc[0]['mean'], 77.5)
            self.assertTrue((Path(temp)/'lt2_misclass_by_freq.png').exists())


if __name__ == '__main__':
    unittest.main()
