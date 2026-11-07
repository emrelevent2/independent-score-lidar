from pathlib import Path
import csv
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
res=root/'results'
v1=json.loads((res/'development_screen.json').read_text())
v2=json.loads((res/'development_screen_v2.json').read_text())
caspi=json.loads((res/'caspi_screen_analysis.json').read_text())
rng=np.random.default_rng(901)

def paired_ci(entries_a,entries_b,metric):
    a={r['seed']:r[metric] for r in entries_a if metric in r}
    b={r['seed']:r[metric] for r in entries_b if metric in r}
    assert set(a)==set(b)
    d=np.array([a[s]-b[s] for s in sorted(a)])
    boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(-1)
    return d.mean(),*np.quantile(boot,[.025,.975])

text=['# Development experiment findings', '',
'These data are for method development. They are not an untouched test set.',
'All comparisons use paired simulated acquisitions across methods. Calibration',
'uses 20 single-return scenes (seeds 3000–3019); screening uses 30 single-return',
'and 30 two-return scenes (seeds 4000–4029) per condition. The same seeds are',
'reused across conditions. Uncertainty below is conditional on this geometry',
'and photon model; it does not measure generalization across scenes.', '',
'## Results that determine the next action', '',
'The first heteroscedastic wavelet candidate does not establish an improvement',
'over a 3x3 box average. A lower waveform MSE is not sufficient evidence of',
'weak-return preservation. Revision 2 (local energy Wiener shrinkage without',
'coefficient pooling) also fails to establish an advantage in this screen.', '',
'| Condition | Method | Null FP per pixel | Weak return recall | Off-target FP per pixel |',
'|---|---|---:|---:|---:|']
for condition in v1['details']:
    rows=list(csv.DictReader((res/'development_screen.csv').open()))
    rows2=list(csv.DictReader((res/'development_screen_v2.csv').open()))
    keep=[r for r in rows if r['condition']==condition and r['method'] in ['box3','candidate','candidate_constant_variance']]
    keep += [r for r in rows2 if r['condition']==condition and r['method']=='block_wiener_v2']
    if condition in caspi['details']: keep+=[caspi['details'][condition]['metrics']]
    for r in keep:
        text.append(f"| {condition} | {r['method']} | {float(r['null_fpr']):.4f} | {float(r['weak_recall']):.4f} | {float(r['off_target_fp']):.4f} |")
text += ['', '## Paired uncertainty on completed CASPI batches', '',
'Differences below are original CASPI minus box3. Intervals use 10000 paired',
'scene-bootstrap replicates, percentile 95% intervals, seed 901. These are',
'exploratory intervals, without correction for repeated model development.', '']
for condition,record in caspi['details'].items():
    for metric in ['weak_recall','off_target_fp','null_fpr']:
        mean,lo,hi=paired_ci(record['scene_metrics'],v1['details'][condition]['scene_metrics']['box3'],metric)
        text.append(f'- {condition}, {metric}: {mean:.4f} [{lo:.4f}, {hi:.4f}].')
text += ['', '## Interpretation and limitations', '',
'The nominal upstream CASPI result is consistent with spatial spreading of weak',
'returns: successful localization in true weak pixels co-occurs with secondary',
'detections elsewhere. This is a controlled failure mode to investigate, not a',
'claim that CASPI generally fails. The common detector fits one primary pulse,',
'then searches residual matched-filter local maxima at least 8 bins away.',
'That exclusion distance limits the 6-bin close-return condition; it must be',
'replaced or supplemented by a joint two-return fit for the final experiment.', '',
'The next candidate should preserve independent evidence for a weak return',
'while preventing transfer of unsupported returns to other pixels. Photon-cycle',
'splitting with a separate validation statistic is a possible direction, but',
'its power cost and relation to existing signal-unmixing methods need testing.', '',
f"Completed CASPI conditions: {', '.join(caspi['completed_conditions'])}.",
f"Pending CASPI conditions: {', '.join(caspi['missing_conditions'])}.", '',
'No FPGA synthesis, measured hardware throughput, held-out superiority result',
'or submission-ready manuscript is available yet.']
(root/'DEVELOPMENT_FINDINGS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print('Updated DEVELOPMENT_FINDINGS.md')
