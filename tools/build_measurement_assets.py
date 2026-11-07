"""Render completed measured-data models on untouched evaluation segments."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'manuscript/ded_style_article_v2';OUT=ROOT/'results/measured_recovery_v1';CL=ROOT/'results/measured_cluster_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    analysis=json.loads((OUT/'analysis.json').read_text());predictive=json.loads((OUT/'predictive.json').read_text());audit=json.loads((OUT/'formula_audit.json').read_text())
    result_audit=json.loads((OUT/'results_audit.json').read_text());cluster=json.loads((CL/'analysis.json').read_text());ca=json.loads((CL/'audit.json').read_text())
    assert analysis['status']==predictive['status']==cluster['status']=='COMPLETE'
    assert audit['status']==result_audit['status']==ca['status']=='PASS'
    assert predictive['analysis_sha256']==result_audit['analysis_sha256']==sha(OUT/'analysis.json')
    assert ca['analysis_sha256']==sha(CL/'analysis.json')
    assert audit['model_source_sha256']==sha(ROOT/'tools/measurement_recovery_model.py')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'mathtext.fontset':'stix'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.8),layout='constrained');body=[]
    inputs=[OUT/n for n in ['analysis.json','predictive.json','formula_audit.json','results_audit.json']]+[CL/'analysis.json',CL/'audit.json']
    inputs.extend(ROOT/n for n in ['MEASURED_RECOVERY_PROTOCOL_20261107.json','MEASURED_CLUSTER_PROTOCOL_20261107.json'])
    inputs.extend(ROOT/'tools'/n for n in ['measurement_recovery_model.py','measurement_recovery_study.py','measurement_recovery_predictive.py','audit_measurement_recovery.py','audit_measured_results.py','measured_cluster_extension.py','audit_measured_cluster.py','build_measurement_assets.py'])
    rows=[]
    for ax,row,fresh in zip(axes,analysis['rows'],cluster['rows']):
        assert row['file']==fresh['file']
        stem=Path(row['file']).stem
        inputs.extend([OUT/(stem+'_heldout.npz'),OUT/(stem+'_predictive.npz'),CL/(stem+'.npz')])
        assert sha(CL/(stem+'.npz'))==fresh['data_sha256']
        z=np.load(CL/(stem+'.npz'));x=np.linspace(72,160,440)
        for gaps,label,color,style in [(z['gaps'],'Measured','#4d5155','-'),(z['baseline_predicted'],'Recovery only','#b34d24','--'),(z['predicted'],'Recovery + clusters','#126a84','-')]:
            y=np.searchsorted(np.sort(gaps),x,side='right')/len(gaps)*100
            ax.plot(x,y,label=label,color=color,ls=style,lw=1.1)
        od=int(row['metadata']['optical_density']);p=row['parameters'];cp=fresh['parameters'];label='Low flux' if od else 'High flux'
        ax.axvline(80,color='#909090',ls=':',lw=.8)
        ax.set(xlabel='Recorded gap (ns)',ylabel='Cumulative fraction (%)',xlim=(72,160),title=f'({"a" if od else "b"}) {label}, OD {od}');ax.legend(frameon=False)
        gain=fresh['mean_log_score_gain'];gain_text=f'{gain:+.5f}' if abs(gain)>=.000005 else '0.00000'
        negligible=cp['cluster_fraction']<=1.00001e-6
        fraction=r'$10^{-6}$' if negligible else f'{cp["cluster_fraction"]:.4f}'
        timescale='---' if negligible else f'{cp["cluster_exponential_mean_ns"]:.2f}'
        body.append(f'{od} & {p["recovery_location_ns"]:.2f} & {p["recovery_scale_ns"]:.2f} & {cp["flux_scale"]:.4f} & {fraction} & {timescale} & {fresh["fixed_cutoff_zero_densities"]} & {gain_text} & {fresh["baseline_cdf_distance"]:.4f}; {fresh["cdf_distance"]:.4f}')
        rows.append(dict(file=row['file'],optical_density=od,parameters=p,cluster_parameters=cp,evaluation=row['evaluation'],fresh_evaluation=fresh,all_gaps=row['metadata']['pairs'],all_gaps_below50=row['random_zero_density_all']))
    for fmt in ['pdf','png']:
        kw={'metadata':{'CreationDate':None,'ModDate':None}} if fmt=='pdf' else {'dpi':180}
        fig.savefig(D/'figures'/('fig06_measurements.'+fmt),bbox_inches='tight',pad_inches=.05,**kw)
    plt.close(fig)
    (D/'table_measured.tex').write_text('\n'.join([r'\begin{table*}[t]\centering',r'\caption{Measured-data recovery and clustering fits. Each acquisition uses disjoint 20000-gap calibration, development and evaluation segments. Recovery location and scale are in ns. The fixed-cutoff column counts evaluation gaps assigned zero density by a cutoff estimated from the calibration minimum. Log-score gain is relative to recovery-only fitting, in nats per transition. CDF distances use the full evaluation gap range. At the cluster-fraction lower bound, the cluster time scale is not reported.}',r'\label{tab:measured}',r'\begin{tabular}{rrrrrrrrr}\toprule',r'OD & $\mu$ (ns) & $s_R$ (ns) & $q$ & $\pi$ & $\tau_b$ (ns) & \shortstack{Fixed-cutoff\\zero densities} & \shortstack{Log-score\\gain} & \shortstack{CDF distance\\recovery; mixture}\\\midrule',*[z+r'\\' for z in body],r'\bottomrule\end{tabular}\end{table*}'])+'\n')
    total=sum(r['all_gaps'] for r in rows);assert all(r['all_gaps_below50']==0 for r in rows)
    text=r"""We analyze the public 48-ns-holdoff measurements at optical densities 3 and 0, corresponding to low and high incident flux. Each acquisition supplies three disjoint segments of 20000 consecutive gaps: calibration, development and evaluation. The development segment exposes temporal clustering in low flux and motivates the additional transition component. Its form and the evaluation segment are fixed before fitting that component. Both fitting stages use calibration data only. Each segment retains its recorded timestamps.

A fixed-cutoff comparator estimates electronics recovery as the minimum calibration gap minus one raw timing-resolution unit (0.008 ns). It gives 77.745 ns in low flux and 76.809 ns in high flux. Each cutoff assigns zero density to one gap in its evaluation segment. Learning a single cutoff therefore leaves a short-gap prediction failure.

"""
    text+=f'The variable-recovery and clustered-transition models assign finite density to all 40000 evaluation gaps. Their support contains all {total} gaps in the two complete acquisitions. Table~\\ref{{tab:measured}} reports fitted parameters and evaluation results. The low-flux recovery scale reaches the 0.15-ns optimization bound; this segment does not resolve a narrower recovery distribution.\n\n'
    text+=r"""A cumulative distribution function (CDF) gives the fraction of gaps no longer than a specified duration. Its largest absolute prediction error summarizes the difference between measured and predicted gap distributions. Figure~\ref{fig:measured} uses 100000 next-gap predictions per model and acquisition, conditioned on the measured preceding phases. In low flux, the clustered-transition component reduces the whole-distribution CDF distance from 0.2449 to 0.0366 and improves mean log density by 0.7522 nats per transition. In high flux, its fitted fraction reaches the $10^{-6}$ lower bound; log density is unchanged to five decimal places, and CDF distances are 0.0310 and 0.0313. The extra component is useful for the low-flux clustering and adds no predictive gain in the high-flux segment.

\input{table_measured.tex}
\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig06_measurements.pdf}
\caption{Measured gap distributions and calibrated transition-model predictions. Each empirical curve uses 20000 evaluation gaps disjoint from calibration and development. Each model curve uses 100000 next-gap predictions conditioned on the measured preceding phases. The displayed range emphasizes short gaps; reported CDF distances use the whole gap distribution. The dotted vertical line marks the nominal 80-ns electronics setting. In high flux the model predictions nearly coincide.}
\label{fig:measured}\end{figure*}
"""
    (D/'measured_results.tex').write_text(text)
    result=dict(status='COMPLETE',rows=rows,total_measured_gaps=total,input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},builder_sha256=sha(__file__))
    (D/'measured_data.json').write_text(json.dumps(result,indent=2));print('Measured assets complete',flush=True);return inputs


if __name__=='__main__':build()
