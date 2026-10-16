"""Summarize measured checkpoint CSVs and plot GI/coverage evolution."""
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for kind,suffix in [('train','gi_analysis.csv'),('rsv','rsv_gi_analysis.csv'),('random','random_gi_analysis.csv')]:
        for status in ['correct','incorrect']:
            for path in Path(a.input).glob(f'model_epoch*/{status}_{suffix}'):
                m=re.search(r'epoch(\d+)_batch(\d+)',str(path));epoch,batch=map(int,m.groups())
                try:df=pd.read_csv(path)
                except pd.errors.EmptyDataError:continue
                if df.empty:continue
                stats={'kind':kind,'status':status,'checkpoint':path.parent.name,'epoch':epoch,'batch':batch,'samples':len(df)}
                fields=['mean_gi','median_gi','percent_gi_above_1'] if kind=='train' else ['mean_pos_gi','mean_neg_gi','mean_total_gi','percent_pos_gi_above_1','percent_total_gi_above_1']
                for name in fields:
                    if name in df:stats[name]=float(df[name].mean())
                rows.append(stats)
    if not rows:p.error('No checkpoint analysis CSVs were found; run the collection pipeline first')
    summary=pd.DataFrame(rows).sort_values(['epoch','batch','kind','status'])
    summary.to_csv(out/'checkpoint_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,kind in zip(axes,['train','rsv','random']):
        group=summary[(summary.kind==kind)&(summary.status=='correct')].sort_values(['epoch','batch'])
        for field in (['mean_gi'] if kind=='train' else ['mean_pos_gi','mean_neg_gi','mean_total_gi']):
            if field in group:ax.plot(range(len(group)),group[field],marker='o',label=field)
        ax.set_title(kind);ax.set_xlabel('Checkpoint (chronological)');ax.set_ylabel('GI');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'gi_evolution.png',dpi=180);fig.savefig(out/'gi_evolution.pdf');plt.close(fig)
    for kind,group in summary[summary.status=='correct'].groupby('kind'):
        field='percent_gi_above_1' if kind=='train' else 'percent_pos_gi_above_1'
        if field in group:plt.plot(range(len(group)),group[field],marker='o',label=kind)
    plt.xlabel('Checkpoint (chronological)');plt.ylabel('Positive GI > 1 (%)');plt.legend();plt.tight_layout();plt.savefig(out/'coverage_evolution.png');plt.close()
if __name__=='__main__':main()
