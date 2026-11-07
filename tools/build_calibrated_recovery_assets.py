"""Rebuild extension figure and CSV tables from saved numerical outputs."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'results/calibrated_recovery_v1/analysis.json').read_text())
    semi=json.loads((ROOT/'results/semisynthetic_recovery_v2/analysis.json').read_text())
    measured=json.loads((ROOT/'results/extended_measured_nulls_v1/analysis.json').read_text())
    out=ROOT/'manuscript/calibrated_recovery';out.mkdir(exist_ok=True)
    for name,rows in [('calibrated_results',r['rows']),('semisynthetic_results',semi['rows']),('measured_nulls',measured['rows'])]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=list(r['thresholds']);names.remove('measured_high')
    def value(name,method,ratio=None):
        return next(x['correct'] for x in r['rows'] if x['name']==name and x['kind']=='evaluate' and x['method']==method and x['ratio']==ratio)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),gridspec_kw={'width_ratios':[1.,1.65]})
    x=np.arange(3);w=.36
    for method,label,offset,color in [('calibrated','Calibrated',-w/2,'#007395'),('censored','Censored',w/2,'#82888b')]:
        axes[0].bar(x+offset,[next(r['correct_secondary'] for r in semi['rows'] if r['method']==method and r['ratio']==q) for q in [.0625,.125,.25]],w,label=label,color=color)
        axes[1].bar(np.arange(9)+offset,[value(n,method) for n in names],w,color=color)
    axes[0].set(xticks=x,xticklabels=['6.25','12.5','25'],xlabel='Secondary / primary amplitude (%)',ylabel='Correct injected / 64',ylim=(0,64),title='(a) Measured-path injection')
    axes[0].legend(frameon=False,loc='upper left')
    axes[1].set(xticks=np.arange(9),xticklabels=['Nom.','d -5%','d +5%','e -2%','e +2%','j 0.1','j 0.5','ap 1%','ap 5%'],xlabel='Mismatch condition',ylabel='Correct / 32',ylim=(0,32),title='(b) Recovery and timing stress tests')
    axes[1].tick_params(axis='x',labelrotation=45)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout(pad=.8,w_pad=1.4);fig.savefig(out/'figure.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    print('Rebuilt calibrated recovery figure and both numerical CSV tables')


if __name__=='__main__':main()
