"""Run binary-search feature-linearity analysis on experiment 01's saved Jacobians."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='results/01_gi_evolution/effective_config.yaml')
    p.add_argument('--checkpoints', help='Comma-separated checkpoint stems; default first and last collected')
    p.add_argument('--max-alpha', type=float, default=111.0)
    p.add_argument('--tolerance', type=float, default=0.01)
    p.add_argument('--device', default='auto')
    p.add_argument('--output-config', default='results/03_linearity/binary_search_config.yaml')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    collected = [Path(name).stem for name in cfg['checkpoints'].get('custom_checkpoints', [])]
    if not collected:
        p.error('Run experiment 01 first to collect explicit checkpoints and Jacobians')
    selected = a.checkpoints.split(',') if a.checkpoints else list(dict.fromkeys([collected[0], collected[-1]]))
    if any(name not in collected for name in selected):
        p.error('Requested checkpoint was not in the collected set')
    cfg['gi_analysis']['analyze_checkpoints'] = selected
    cfg['gi_analysis']['device'] = a.device
    cfg['linear_approximation'] = {'threshold': 0.95, 'max_alpha': a.max_alpha,
                                   'tolerance': a.tolerance, 'batch_size': 50,
                                   'max_iterations': 20, 'verbose': False}
    target = Path(a.output_config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([sys.executable, str(Path(__file__).with_name('historical_binary_search.py')),
                    '--config', str(target)], check=True)


if __name__ == '__main__':
    main()
