from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
res=root/'results'
v2=json.loads((res/'geometry_screen_v2.json').read_text())
v3_path=res/'geometry_screen_v3.json'
v3=json.loads(v3_path.read_text()) if v3_path.exists() else None
methods=['box3','depth_guided_3','depth_guided_7','wavelet_crossfit_9_fused']
records=dict(v2['details'])
if v3:
    methods+=['neighbor_wavelet_3','neighbor_box_3','neighbor_wavelet_9','neighbor_box_9']
    for key,value in v3['details'].items():
        records[key]={'results':{**records[key]['results'],**value['results']}}
lines=['# Geometry and calibration development audit','',
'All experiments in this report are development data. No final held-out result',
'has been established. Calibration targets 1% separately in globally null scenes',
'and absent pixels in mixed scenes, taking the larger empirical threshold.',
'Screen false-positive rates can differ from that target and are reported.', '',
'| Geometry | Method | Null FP | Weak recall | Off-target FP |',
'|---|---|---:|---:|---:|']
for geometry,record in records.items():
    for method in methods:
        if method not in record['results']: continue
        m=record['results'][method]['metrics']
        lines.append(f"| {geometry} | {method} | {m['null_fpr']:.4f} | {m['weak_recall']:.4f} | {m['off_target_fp']:.4f} |")
lines += ['', '## Interpretation', '',
'Cycle splitting reduces return-transfer artifacts but loses power by dividing',
'the photons. On the flat geometry, several simple pooling methods recover more',
'weak returns. The apparent advantage over plain box3 near a primary-depth step',
'does not survive comparison with the depth-guided pooling control. It would be',
'misleading to claim superiority from the plain-box comparison alone.', '',
'Neighbor-only proposals are a further candidate: the tested pixel is removed',
'from the proposal statistics and all of its photons remain available for',
'validation. The neighborhood contribution must be assessed against a no-wavelet',
'ablation and against depth-guided pooling, not just the original CASPI.', '',
'Calibration and photon model uncertainty, unrelated scene geometries, spatial',
'crosstalk, pulse mismatch, true real-data validation and hardware validation',
'remain outstanding. Neighbor-only noise independence is conditional on the',
'scene/rates and assumes independent detector pixels. It is not a new principle',
'in general denoising; relevant J-invariant/Noise2Self literature must be cited.']
(root/'GEOMETRY_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
geometries=list(records)
for method in methods:
    values=[records[g]['results'].get(method) for g in geometries]
    recall=[v['metrics']['weak_recall'] if v else np.nan for v in values]
    fp=[v['metrics']['off_target_fp'] if v else np.nan for v in values]
    axes[0].plot(geometries,recall,'o-',label=method)
    axes[1].plot(geometries,fp,'o-',label=method)
axes[0].set_ylabel('Weak return recall')
axes[1].set_ylabel('False secondary detections outside weak support')
axes[1].axhline(.01,color='k',ls='--',label='Calibration target')
for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7); ax.tick_params(axis='x',rotation=15)
fig.suptitle('Development audit with stronger pooling controls — not a final test set')
fig.savefig(res/'figures/geometry_audit.png',dpi=180)
fig.savefig(res/'figures/geometry_audit.pdf')
print('Updated geometry report and figure; neighbor-only coverage:',list(v3['details']) if v3 else [])
