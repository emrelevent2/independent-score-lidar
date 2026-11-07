from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
x=json.loads((root/'results/context_shift_screen.json').read_text())
methods=['raw','box3','depth_guided_3','depth_guided_7','neighbor_box_7','neighbor_box_9']
lines=['# Calibration transfer development audit','',
'Thresholds were frozen from the flat geometry calibration, seeds 5000–5019.',
'The 100 new acquisitions per scenario use seeds 7000–7099. These are development',
'experiments; the scenarios were designed after inspecting earlier results.', '',
'A 2x2 or 4x4 region lacks the weak secondary surface while surrounding pixels',
'contain it. The true primary surface and background are unchanged. Pixel noise',
'is independent under the simulator. Optical spatial blur and crosstalk are absent.', '',
'| Scenario | Method | Weak recall | False return at absent pixels |',
'|---|---|---:|---:|']
for scenario,record in x['details'].items():
    for method in methods:
        m=record['results'][method]['metrics']
        recall=f"{m['recall']:.4f}" if 'recall' in m else 'not applicable'
        fp=f"{m['off_target_fp']:.4f}" if 'off_target_fp' in m else 'not applicable'
        lines.append(f'| {scenario} | {method} | {recall} | {fp} |')
lines+=['', 'The neighborhood-only method trades dense-region sensitivity for greater',
'stability at unsupported locations. These experiments do not establish optimality',
'or a finite-sample error guarantee. Larger calibration sets, untouched geometry',
'families and real measurements are required. Wavelet shrinkage has not shown',
'a consistent benefit over the plain neighbor-box proposal and should not be',
'claimed as a required innovation.', '',
'Intervals below use paired acquisition bootstrap, seed 113, 10000 replicates.',
'They are conditional on the fixed scene geometry and exclude calibration uncertainty.']
rng=np.random.default_rng(113)
for scenario in ['hole2','hole4']:
    records=x['details'][scenario]['results']
    for method in ['box3','depth_guided_7']:
        a={v['seed']:v['off_target_fp'] for v in records[method]['scene_metrics']}
        b={v['seed']:v['off_target_fp'] for v in records['neighbor_box_9']['scene_metrics']}
        d=np.array([a[s]-b[s] for s in sorted(a)])
        boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(-1)
        lo,hi=np.quantile(boot,[.025,.975])
        lines.append(f'- {scenario}: {method} minus neighbor_box_9 false return difference {d.mean():.4f} [{lo:.4f}, {hi:.4f}].')
(root/'CONTEXT_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
pos=np.arange(len(methods)); width=.36
for shift,scenario in [(-width/2,'hole2'),(width/2,'hole4')]:
    rec=x['details'][scenario]['results']
    axes[0].bar(pos+shift,[rec[m]['metrics']['off_target_fp'] for m in methods],width,label=scenario)
    axes[1].bar(pos+shift,[rec[m]['metrics']['recall'] for m in methods],width,label=scenario)
axes[0].set_yscale('log'); axes[0].axhline(.01,color='k',ls='--',lw=1)
axes[0].set_ylabel('False return probability at absent pixels (log scale)')
axes[1].set_ylabel('Weak return recall')
for ax in axes:
    ax.set_xticks(pos,methods,rotation=25,ha='right'); ax.grid(axis='y',alpha=.2); ax.legend()
fig.suptitle('Calibration transfer to unobserved occupancy patterns — development experiment')
fig.savefig(root/'results/figures/context_audit.png',dpi=180)
fig.savefig(root/'results/figures/context_audit.pdf')
print('Updated context report and figure')
