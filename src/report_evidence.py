"""Generate tables, figures and manuscript text from completed evidence jobs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'results'/'likelihood_evidence_screen.json').read_text())
    measured=json.loads((ROOT/'results'/'measured_likelihood_evidence.json').read_text())
    assert len(r['details'])==8 and all(not d['in_progress'] for d in r['details'].values())
    rows=[]
    for name,d in r['details'].items():
        s=d['summary'];lo,hi=s['declaration_rate_ci95']
        rows.append((name,s['n'],s['declared'],s['declaration_rate'],lo,hi))
    lines=['# Likelihood evidence prototype findings', '',
        'This report concerns a NEW likelihood method. Results for the old',
        'empirically calibrated residual score must not be attributed to it.', '',
        '## Pre-specified model verification', '',
        'Known 5-bin FWHM, integer-delay, first-detection model; N=1000,',
        'T=256. Every integer primary delay is bounded in the composite null.',
        'All alternative parameters use a 9x9 neighborhood excluding the tested',
        'pixel. A fixed E>=100 rule corresponds to alpha=.01 under the analytic',
        'model argument. No empirical threshold calibration is performed.', '',
        '| Context | Acquisitions | Declarations | Rate | Exact 95% interval |',
        '|---|---:|---:|---:|---|']
    for name,n,k,rate,lo,hi in rows:
        lines.append(f'| {name} | {n} | {k} | {rate:.4f} | [{lo:.4f}, {hi:.4f}] |')
    maxgap=max(d['summary']['global_gap_max'] for d in r['details'].values())
    exponent=int(np.floor(np.log10(maxgap)))
    gap_tex=f'{maxgap/10**exponent:.2f}'+r'\times10^{'+str(exponent)+'}'
    lines+=['', 'Every alternative declaration was within two bins of the true',
        'secondary in these fixed-geometry trials. Seeds are paired across',
        'contexts: do not pool five null contexts into 750 independent trials.',
        'For each null context, zero out of 150 has a two-sided 95% upper limit',
        'of 2.43%, so simulation alone does not demonstrate a 1% upper bound.', '',
        f'Maximum global numerical bound gap: {maxgap:.8g} log-likelihood units.',
        'No conservative zero-bound fallbacks occurred. Optimization can report',
        'failure while a separately checked concave tangent is still feasible.',
        'Recorded runtimes are about 0.26--0.37 seconds per tested pixel for the',
        'null grid bound, not real-time image processing or FPGA timings.', '',
        '## Measured count experiment', '',
        'The fixed central ROI of the public 3.2 ms camouflage file used the',
        'supplied measured response and a separate independent-Poisson model.',
        'All 550 possible integer primary delays were considered. Out of 196',
        'scored pixels, 50 exceeded E=100 and 38 exceeded E=3920, the Bonferroni',
        'threshold for alpha=.05 over these 196 pixels under the model.',
        'These are declarations, not 50 verified correct returns. No ground-truth',
        'return map is supplied. The known-IRF, uniform-background, integer-delay',
        'and independence assumptions require additional validation.', '',
        '## Remaining scientific gaps', '',
        'Fractional primary delay and IRF uncertainty need coverage in the null.',
        'Numerical certificates need rigorous rounding-error bounds. Strong',
        'likelihood and state-of-the-art multi-return baselines are missing for',
        'this NEW method. No untouched heterogeneous-scene benchmark exists.',
        'Novelty beyond established universal inference and J-invariant',
        'prediction is not established. The paper remains a working draft.']
    (ROOT/'EVIDENCE_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    tex=[r'\subsection*{Prototype likelihood evidence results}',
        r'A pre-specified development verification used the known Gaussian',
        r'first-detection model with integer primary delay, 1000 cycles and 256',
        r'bins. The decision threshold was fixed at $E\geq100$, without empirical',
        r'calibration. Every alternative parameter was obtained from the',
        r'neighbor-only nine-by-nine proposal. The composite null covered every',
        r'integer primary delay and nonnegative signal and background amplitudes.',
        r'\begin{table}[ht]\centering',
        r'\begin{tabular}{lrrr}\toprule',
        r'Context & $n$ & Declarations & Rate (95\% interval) \\\midrule']
    pretty={'nominal':'Nominal single return','high_background':'High background',
            'bright_primary':'Bright primary','pileup':'Pileup',
            'neighbor_surface':'Secondary at donors only',
            'alternative_3':'Secondary count 3','alternative_6':'Secondary count 6',
            'alternative_10':'Secondary count 10'}
    for name,n,k,rate,lo,hi in rows:
        tex.append(f'{pretty[name]} & {n} & {k} & {rate:.3f} ({lo:.3f}, {hi:.3f}) '+r'\\')
    tex+=[r'\bottomrule\end{tabular}',
        r'\caption{Prototype verification at one tested pixel per acquisition.',
        r'Intervals are exact two-sided binomial 95\% intervals within a context.',
        r'Contexts share seeds and must not be pooled as independent trials.}',
        r'\end{table}',
        r'All declarations in the three alternative conditions were within two',
        r'bins of the true secondary. Zero declarations in 150 acquisitions has',
        r'a 95\% upper bound of 0.0243; these finite trials alone therefore do not',
        r'demonstrate a 0.01 false-positive upper bound. The analytic argument is',
        r'separate and depends on the stated null and numerical assumptions.',
        f'The largest numerical upper--lower likelihood gap was ${gap_tex}$;',
        r'no conservative zero-bound fallback was used. The mean null-grid',
        r'computation time ranged from 0.260 to 0.369 seconds per pixel across',
        r'conditions. These measurements do not establish real-time performance.',
        r'For the measured-data prototype, the central $24\times24$ crop used',
        r'rows 64--87 and columns 27--50 in zero-based coordinates. Excluding a',
        r'five-pixel border left 196 scored pixels. The known-response Poisson',
        r'null covered all 550 integer delays. Fifty pixels exceeded $E=100$;',
        r'38 exceeded $E=3920$, corresponding to Bonferroni level 0.05 across',
        r'those 196 pixels under the model. These are model-based declarations,',
        r'not verified true returns. Unknown response errors, nonuniform background,',
        r'fractional delay and detector correlations may invalidate the null.',
        r'\begin{figure}[ht]\centering',
        r'\includegraphics[width=\linewidth]{../results/figures/likelihood_evidence_audit.pdf}',
        r'\caption{Prototype verification at the fixed evidence threshold.',
        r'Left, localized detection probability versus incident secondary photons',
        r'per acquisition, with exact 95\% binomial intervals over 120 acquisitions.',
        r'Right, evidence in the fixed measured-data ROI. The outlined declarations',
        r'exceed $E=100$ under the measured-response Poisson model. No ground-truth',
        r'accuracy is inferred from this map.}\end{figure}']
    (ROOT/'manuscript'/'evidence_results.tex').write_text('\n'.join(tex)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    vals=[r['details'][f'alternative_{n}']['summary'] for n in [3,6,10]]
    means=np.array([s['localized_rate'] for s in vals])
    cis=np.array([s['localized_rate_ci95'] for s in vals])
    axes[0].errorbar([3,6,10],means,yerr=np.stack([means-cis[:,0],cis[:,1]-means]),
                     marker='o',color='#2563eb',capsize=4)
    axes[0].set(xlabel='Incident secondary photons / acquisition',ylabel='Localized detection probability',
                ylim=(0,1.05),title='Known-IRF first-detection model')
    log_e=np.empty((14,14));declared=np.zeros((14,14),bool)
    for record in measured['records']:
        yy=record['row']-69;xx=record['column']-32
        log_e[yy,xx]=record['log_e'];declared[yy,xx]=record['log_e']>=np.log(100)
    im=axes[1].imshow(log_e,cmap='coolwarm',vmin=-10,vmax=10,origin='upper')
    yy,xx=np.where(declared);axes[1].scatter(xx,yy,s=34,facecolors='none',edgecolors='black',linewidths=.7)
    axes[1].set(xlabel='ROI column',ylabel='ROI row',title='Measured counts: log evidence')
    fig.colorbar(im,ax=axes[1],label='log(E), color clipped at -10 and 10')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf']:fig.savefig(ROOT/'results'/'figures'/f'likelihood_evidence_audit.{ext}',dpi=180)
    print('Reports, manuscript table and evidence figure saved')


if __name__=='__main__':main()
