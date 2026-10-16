"""Train and evaluate all 16 seed/optimizer runs, then aggregate."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def call(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--arch',choices=['mlp','cnn','all'],default='all');a=p.parse_args()
    root=Path('experiments/12_seed_optimizer_robustness')
    call(root/'gen_configs.py')
    for arch in (['mlp','cnn'] if a.arch=='all' else [a.arch]):
        for kind,ids in [('seed',list(range(5))),('optim',['sgd','adam','rmsprop'])]:
            for run_id in ids:
                cfg=root/'configs'/arch/f'{kind}_{run_id}.yaml'
                c=yaml.safe_load(cfg.read_text())
                call('src/training/mnist_train.py','--config',cfg)
                call(root/'same_class_gi_growth.py','--config',cfg,'--checkpoint_dir',c['training']['checkpoint_dir'],
                     '--data_path','data/MNIST-Original','--points','0,5,10,15,20' if arch=='mlp' else '0,10,20',
                     '--k','5','--arch',arch,'--run_type',kind,'--run_id',run_id,
                     '--output_dir',root/'results'/f'{arch}_{kind}'/str(run_id),'--summary_csv',root/'results/summary.csv',
                     '--step_size','0.01' if arch=='mlp' else '0.02','--max_eps','3' if arch=='mlp' else '1.5')
    call(root/'aggregate_multiseed.py','--summary_csv',root/'results/summary.csv','--results_dir',root/'results','--output_dir',root/'results/analysis')
if __name__=='__main__':main()
