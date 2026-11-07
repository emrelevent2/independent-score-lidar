"""Deterministic final-stage figures and TeX tables from audited outcomes."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript';F=M/'figures'
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
NAMES=dict(local='Local fit',guarded='Guarded location',previous_routed='Previous router',routed='Previous router',disagreement='Disagreement ablation')
COLORS=dict(local='#7854a1',guarded='#176b94',previous_routed='#c75d2b',routed='#c75d2b',disagreement='#438a61')


def main():
    G=ROOT/'results/guarded_confirmation_v1';R=ROOT/'results/real_multiframe_test_v1'
    assert json.loads((G/'integrity.json').read_text())['status']=='PASS' and json.loads((R/'integrity.json').read_text())['status']=='PASS'
    a=json.loads((G/'analysis.json').read_text());outputs=json.loads((G/'endpoint_records.json').read_text())
    cases=json.loads((ROOT/'data/guarded_confirmation_v1/manifest.json').read_text())['cases'];methods=['local','guarded','previous_routed']
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.3),layout='constrained')
    for ax,g,title in [(axes[0,0],'coherent','(a) Spatially supported targets'),(axes[0,1],'singleton','(b) Isolated targets')]:
        cs=[c for c in cases if c['kind']=='positive' and c['support_family']==g]
        for method in methods:
            es=outputs[f'{method}_0.005'][g]
            values=[100*np.mean([e['correct'] for c,e in zip(cs,es) if c['secondary']==level]) for level in [3.,6.,10.]]
            ax.plot([3,6,10],values,'o-',color=COLORS[method],label=NAMES[method],markersize=3,linewidth=1.1)
        ax.set(xlabel='Incident secondary photons',ylabel='Correct localization (%)',ylim=(0,102),xticks=[3,6,10],title=title)
        ax.legend(fontsize=6.7,loc='lower right')
    families=['coherent','singleton','filament','crossing_step'];x=np.arange(4)
    for j,method in enumerate(methods):
        gs=a['methods'][f'{method}_0.005']['groups']
        axes[1,0].bar(x+(j-1)*.23,[gs[g]['wrong_depth']['count'] for g in families],.23,color=COLORS[method],label=NAMES[method])
        axes[1,1].bar(j,gs['null']['declared']['count'],.55,color=COLORS[method])
    axes[1,0].set(xticks=x,xticklabels=['Coherent','Isolated','Filament','Step'],ylabel='Wrong positive location (count)',title='(c) Localization errors');axes[1,0].legend(fontsize=6.5)
    axes[1,1].set(xticks=[0,1,2],xticklabels=['Local','Guarded','Router'],ylabel='Null declarations / 2000',title='(d) Distinct false-alarm endpoint')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(F/f'guarded_confirmation.{ext}',dpi=220)
    plt.close(fig)
    lines=[r'\begin{tabular}{lrrrrrrrrr}\toprule',r'Method & \multicolumn{2}{c}{Coherent /900} & \multicolumn{2}{c}{Isolated /600} & \multicolumn{2}{c}{Filament /300} & \multicolumn{2}{c}{Step /300} & Null /2000\\',r' & Correct & Wrong & Correct & Wrong & Correct & Wrong & Correct & Wrong & False\\\midrule']
    for method in methods:
        gs=a['methods'][f'{method}_0.005']['groups'];values=[]
        for g in families:values.extend([gs[g]['correct']['count'],gs[g]['wrong_depth']['count']])
        values.append(gs['null']['declared']['count']);lines.append(NAMES[method]+' & '+' & '.join(map(str,values))+r'\\')
    lines.extend([r'\bottomrule\end{tabular}']);(M/'guarded_table.tex').write_text('\n'.join(lines)+'\n')
    real=json.loads((R/'analysis.json').read_text());fig,axes=plt.subplots(2,2,figsize=(7.1,4.8),layout='constrained')
    table=[r'\begin{tabular}{lrrrrrr}\toprule',r' & \multicolumn{3}{c}{Two-interval pairs /500} & \multicolumn{3}{c}{Known-one unmatched /1000}\\',r'Frame, retention & Local & Router & Override & Local & Router & Override\\\midrule']
    for fi,name in enumerate(['Sequence17_first','Sequence3_middle']):
        for method in ['local','routed','disagreement']:
            values=[real['methods'][f'{name}_{i}_0.01_{method}']['groups'] for i in range(3)]
            axes[fi,0].plot([1,2.5,5],[100*v['two_interval']['pair']/500 for v in values],'o-',color=COLORS[method],label=NAMES[method],markersize=3,linewidth=1.1)
            axes[fi,1].plot([1,2.5,5],[100*v['known_one']['accepted_unmatched']/1000 for v in values],'o-',color=COLORS[method],label=NAMES[method],markersize=3,linewidth=1.1)
        for i,prob in enumerate([1,2.5,5]):
            gs=[real['methods'][f'{name}_{i}_0.01_{method}']['groups'] for method in ['local','routed','disagreement']]
            table.append(f'{"17 (primary)" if fi==0 else "3 (validation)"}, {prob:g}\\% & '+' & '.join(str(g['two_interval']['pair']) for g in gs)+' & '+' & '.join(str(g['known_one']['accepted_unmatched']) for g in gs)+r'\\')
        axes[fi,0].set(ylabel='Pair agreement (%)',xlabel='Retained recorded counts (%)',xticks=[1,2.5,5],ylim=(0,50),title=f'({chr(97+2*fi)}) Sequence{17 if fi==0 else 3}')
        axes[fi,1].set(ylabel='Accepted-unmatched (%)',xlabel='Retained recorded counts (%)',xticks=[1,2.5,5],ylim=(0,2.3),title=f'({chr(98+2*fi)}) Known-one intervals')
        axes[fi,0].legend(fontsize=6.5,loc='upper left');axes[fi,1].legend(fontsize=6.5,loc='upper left')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(F/f'real_transfer.{ext}',dpi=220)
    plt.close(fig);table.extend([r'\bottomrule\end{tabular}']);(M/'real_transfer_table.tex').write_text('\n'.join(table)+'\n')
    manifest=dict(guarded_analysis_sha256=hashlib.sha256((G/'analysis.json').read_bytes()).hexdigest(),real_analysis_sha256=hashlib.sha256((R/'analysis.json').read_bytes()).hexdigest())
    (M/'figures/improvement_assets.json').write_text(json.dumps(manifest,indent=2));print('Wrote new simulation/real figures and tables',flush=True)


if __name__=='__main__':main()
