"""Run all 2x2 control cells and generate their statistical comparison."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--untrained',required=True)
    p.add_argument('--trained',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--dataset',choices=['mnist','fashionmnist'],default='mnist')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=Path('experiments/gi_coverage/test_to_train')
    folder=Path('experiments/11_discriminative_baselines')
    allclasses=base/('fashion_mnist_test_to_all_classes_gi_coverage.py' if a.dataset=='fashionmnist' else 'mnist_test_to_all_classes_gi_coverage.py')
    diff=folder/('fashionmnist' if a.dataset=='fashionmnist' else 'mnist')/'compute_diff_class_rc.py'
    csvs={}
    def call(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
    for name,checkpoint in [('random',a.untrained),('trained',a.trained)]:
        dest=out/name
        call(allclasses,'--config',a.config,'--checkpoint',checkpoint,'--data_path',a.data_path,'--output_path',dest,'--k',10)
        csv=list(dest.rglob('test_to_all_classes_gi_coverage.csv'))
        if len(csv)!=1:raise RuntimeError(f'Expected one all-class coverage CSV in {dest}')
        csvs[name]=csv[0]
        call(diff,'--config',a.config,'--checkpoint',checkpoint,'--data_path',a.data_path,'--reference_csv',csv[0],'--output_csv',out/f'{name}_diff.csv')
    call(folder/'mnist/analyze_2x2.py','--random_csv',csvs['random'],'--trained_csv',csvs['trained'],
         '--random_diff_csv',out/'random_diff.csv','--trained_diff_csv',out/'trained_diff.csv','--output_dir',out/'analysis')
if __name__=='__main__':main()
