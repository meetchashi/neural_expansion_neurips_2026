"""Collect fixed-sample Jacobians and measure train/RSV/random GI evolution."""
import argparse
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',default='configs/experiments/mlp_mnist_exp2.yaml')
    p.add_argument('--checkpoints',type=int,default=11)
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    paths=list(Path(cfg['checkpoints']['base_dir']).glob('model_epoch*_batch*.pth'))
    paths.sort(key=lambda x:tuple(map(int,re.search(r'epoch(\d+)_batch(\d+)',x.name).groups())))
    if not paths:p.error('Train the model first; no checkpoints were found')
    chosen=[paths[i].name for i in np.unique(np.linspace(0,len(paths)-1,min(a.checkpoints,len(paths)),dtype=int))]
    cfg['checkpoints'].update(strategy='custom',custom_checkpoints=chosen)
    cfg['gi_analysis']['device']='cuda' if torch.cuda.is_available() else 'cpu'
    target=Path('results/01_gi_evolution/effective_config.yaml');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(yaml.safe_dump(cfg))
    for script in ['collect_jacobians_and_svd','get_gi_along_train_samples','get_gi_along_rsvs_v2','get_gi_along_random_dirs']:
        subprocess.run([sys.executable,f'experiments/gi_jacobian_svd/{script}.py','--config',str(target)],check=True)
    subprocess.run([sys.executable, 'experiments/01_gi_evolution/plot.py', '--input', cfg['gi_analysis']['analysis_output_dir'], '--output', 'results/01_gi_evolution/figures'], check=True)
if __name__=='__main__':main()
