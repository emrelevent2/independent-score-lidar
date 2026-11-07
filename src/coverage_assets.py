"""Render audited coverage results without choosing a new method or endpoint."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/coverage_confirmation_v1';M=ROOT/'manuscript'


def main():
    a=json.loads((D/'analysis.json').read_text())
    for name in ['integrity.json','supplemental_integrity.json']:
        assert json.loads((D/name).read_text())['status']=='PASS'
    groups=['coherent','singleton','filament','crossing_step'];pretty=['Coherent','Isolated','Filament','Crossing step']
    lines=[r'\begin{tabular}{lrrrrr}\toprule',r'Certified method & Coherent /900 & Isolated /600 & Filament /300 & Step /300 & Null /2000\\',r' & \multicolumn{4}{c}{Correct (wrong-depth)} & Declarations\\\midrule']
    for key,name in [('spatial_cert_0.01','Spatial-only'),('coverage_cert_0.01','Coverage')]:
        s=a['methods'][key]['groups'];vals=[f'{s[g]["correct"]["count"]} ({s[g]["wrong_depth"]["count"]})' for g in groups]
        lines.append(name+' & '+' & '.join(vals)+f' & {s["null"]["declared"]["count"]} '+r'\\')
    (M/'coverage_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
    lines=[r'\begin{tabular}{llrrrrr}\toprule',r'Method & Target (\%) & Coherent & Isolated & Filament & Step & Null\\\midrule']
    names=dict(spatial_fast='Spatial score',coverage_fast='Coverage score',local_fit='Local fit',guided_fit='Guided fit',previous_routed='Previous router')
    for key,s in a['methods'].items():
        if s['analytic']:continue
        gs=s['groups'];vals=[f'{gs[g]["correct"]["count"]} ({gs[g]["wrong_depth"]["count"]})' for g in groups]
        lines.append(names[s['method']]+f' & {100*s["alpha"]:g} & '+' & '.join(vals)+f' & {gs["null"]["declared"]["count"]} '+r'\\')
    (M/'coverage_empirical_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
    x=np.arange(4)
    for j,(key,name,color) in enumerate([('spatial_cert_0.01','Spatial-only','#82939d'),('coverage_cert_0.01','Coverage','#176b94')]):
        s=a['methods'][key]['groups'];v=np.array([s[g]['correct']['rate']*100 for g in groups])
        axes[0].bar(x+(j-.5)*.36,v,width=.36,label=name,color=color)
        for i,value in enumerate(v): axes[0].text(i+(j-.5)*.36,value+1,f'{s[groups[i]]["correct"]["count"]}',ha='center',fontsize=7)
        err=[s['null']['declared']['count'],sum(s[g]['wrong_depth']['count'] for g in groups)]
        axes[1].bar(np.arange(2)+(j-.5)*.36,err,width=.36,color=color)
        for i,value in enumerate(err): axes[1].text(i+(j-.5)*.36,value+.15,str(value),ha='center',fontsize=8)
    axes[0].set(xticks=x,xticklabels=pretty,ylabel='Correct secondary localization (%)',ylim=(0,100),title='(a) Matched certified alternatives')
    axes[0].tick_params(axis='x',labelsize=7);axes[0].legend(fontsize=7,loc='upper left')
    axes[1].set(xticks=[0,1],xticklabels=['Null\n(n = 2000)','Wrong positive\n(n = 2100)'],ylabel='Observed declarations',title='(b) Separate error endpoints')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(M/f'figures/coverage_confirmation.{ext}',dpi=220)
    plt.close(fig)
    (M/'figures/coverage_assets.json').write_text(json.dumps(dict(analysis_sha256=hashlib.sha256((D/'analysis.json').read_bytes()).hexdigest(),protocol_sha256=a['protocol_sha256']),indent=2))
    print('Audited coverage tables and figure generated',flush=True)


if __name__=='__main__':main()
