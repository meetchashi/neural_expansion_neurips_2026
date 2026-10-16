"""Run the source-compatible two-stage adversarial pipeline.

The historical normalized-coordinate clipping, PNG round trip, training-source
sampling and ImageNette local-label behavior are preserved. See README.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'imagenette'])
    p.add_argument('--model', required=True, help='MNIST checkpoint or historical pretrained model name')
    p.add_argument('--data-path', required=True)
    p.add_argument('--attack', choices=['cw', 'ifgsm', 'pgd'], required=True)
    p.add_argument('--k', type=int, required=True, help='Candidates per class, before filtering clean errors')
    p.add_argument('--max-train', type=int, required=True, help='Maximum anchors per class')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    a = p.parse_args(argv)
    if a.k < 1 or a.max_train < 1:
        p.error('--k and --max-train must be positive')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('Use an empty output directory so old samples cannot contaminate this run')
    (out / 'run.json').write_text(json.dumps({'protocol': 'recovered_historical', **vars(a)}, indent=2) + '\n')
    scripts = Path(__file__).parent / 'historical'
    shared = ['--model', a.model, '--dataset', a.dataset, '--data_path', a.data_path,
              '--seed', str(a.seed), '--device', a.device]
    subprocess.run([sys.executable, str(scripts / 'generate_adversarial.py'), *shared,
                    '--attack', a.attack, '--k', str(a.k), '--output', str(out)], check=True)
    subprocess.run([sys.executable, str(scripts / 'compute_gi_from_adversarials.py'), *shared,
                    '--adv_dir', str(out), '--max_train', str(a.max_train)], check=True)


if __name__ == '__main__':
    main()
