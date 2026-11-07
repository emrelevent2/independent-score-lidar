"""Current-method comparison tables and sparse-kernel figure from audits."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript';D=ROOT/'results'


def main():
    for folder in ['coverage_external_v1','real_coverage_development_v1','real_primary_bank_development_v1']:
        assert json.loads((D/folder/'integrity.json').read_text())['status']=='PASS'
    a=json.loads((D/'coverage_external_v1/analysis.json').read_text());lines=[r'\begin{tabular}{lrrr}\toprule',r'Method & Correct & Wrong & Null\\\midrule']
    methods=[('coverage_empirical','Coverage score'),('local_likelihood_cal0.005','Local fit'),('complete_guided_likelihood_cal0.005','Strong guided fit'),
        ('unrolled_raw_cal0.005','Unrolling (raw)'),('unrolled_coates_cal0.005','Unrolling (Coates)'),('caspi_cal0.005','CASPI + reporter')]
    for key,name in methods:
        s=a['methods'][key];lines.append(f'{name} & {s["correct"]} & {s["wrong_depth"]} & {s["null_false"]} '+r'\\')
    s=a['methods']['coverage_analytic'];lines.extend([r'\midrule',f'Coverage (analytic 1\%) & {s["correct"]} & {s["wrong_depth"]} & {s["null_false"]} '+r'\\',r'\bottomrule\end{tabular}'])
    (M/'external_current_table.tex').write_text('\n'.join(lines)+'\n')
    dev=json.loads((D/'real_coverage_development_v1/analysis.json').read_text());bank=json.loads((D/'real_primary_bank_development_v1/analysis.json').read_text())
    lines=[r'\begin{tabular}{lrrr}\toprule',r'Method & Retained (\%) & Pair /200 & Known-one unmatched /100\\\midrule']
    for method,name,source in [('local','Local fit',dev),('coverage','Single-anchor coverage',dev),('coverage','Three-anchor coverage',bank),('routed','Earlier router',dev)]:
        for i,prob in enumerate([.01,.025,.05]):
            s=source['methods'][f'{i}_0.01_{method}']['groups'];lines.append(f'{name} & {100*prob:g} & {s["evaluation_two"]["pair"]} & {s["evaluation_one"]["accepted_unmatched"]} '+r'\\')
    (M/'real_coverage_development_table.tex').write_text('\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n')
    b=json.loads((D/'coverage_sparse_benchmark_v1/analysis.json').read_text());records=b['records']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5),layout='constrained')
    for j,(key,color) in enumerate([('sparse','#176b94'),('full','#82939d')]):
        v=np.array([r['seconds'][key]*1000 for r in records]);median=np.median(v)
        axes[0].bar(j,median,width=.48,color=color)
        axes[0].errorbar(j,median,yerr=[[median-v.min()],[v.max()-median]],fmt='none',ecolor='#303b42',capsize=3,linewidth=.7)
        axes[0].text(j,v.max()*1.13,f'{median:.2f} ms',ha='center',fontsize=8)
    axes[0].set(xticks=[0,1],xticklabels=['Sparse: 20','Full: 1396'],yscale='log',ylim=(2,650),ylabel='Numerator lower-bound kernel (ms)',title='(a) Fixed 18-query computation audit')
    gaps=np.array([r['log_probability_gap_interval'][1] for r in records]);axes[1].plot(np.arange(1,19),gaps,'o',ms=3.5,color='#176b94')
    axes[1].axhline(np.median(gaps),color='#82939d',lw=.8,ls='--',label='Median upper gap')
    axes[1].set(xlabel='Fixed query index',ylabel='Full-minus-sparse log mass',title='(b) Explicit retained-mass cost',ylim=(-.04,1.15));axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['pdf','png']:fig.savefig(M/f'figures/sparse_computation_v5.{ext}',dpi=220)
    plt.close(fig)
    files=['coverage_external_v1/analysis.json','real_coverage_development_v1/analysis.json','real_primary_bank_development_v1/analysis.json','coverage_sparse_benchmark_v1/analysis.json']
    (M/'figures/v5_assets.json').write_text(json.dumps({name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in files},indent=2))
    print('Generated audited current-comparison assets',flush=True)


if __name__=='__main__':main()
