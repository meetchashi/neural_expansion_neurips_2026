"""Build 100 high-coverage rays, measure orthogonal tubes, and summarize."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-path',required=True);p.add_argument('--gi-csv',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);base=Path(__file__).parent
    def call(script,*args):subprocess.run([sys.executable,str(base/script),*map(str,args)],check=True)
    call('build_ray_pool.py','--gi_csv',a.gi_csv,'--config',a.config,'--data_path',a.data_path,'--n_rays',100,'--output_csv',out/'ray_pool.csv')
    call('compute_orthogonal_tube_gi.py','--config',a.config,'--checkpoint',a.checkpoint,'--data_path',a.data_path,'--ray_pool',out/'ray_pool.csv',
         '--n_directions',20,'--output_csv',out/'tube_gi.csv')
    call('analyze_tube_profile.py','--tube_csvs',out/'tube_gi.csv','--labels',Path(a.config).stem,'--output_dir',out/'analysis')
if __name__=='__main__':main()
