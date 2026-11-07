"""Regenerate a development report, preserving the detector fairness failure."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    a=json.loads((ROOT/'results'/'heterogeneous_screen_v1.json').read_text())
    b=json.loads((ROOT/'results'/'fair_detector_audit.json').read_text())
    rows=[]
    for scene,detail in a['details'].items():
        for m,r in detail['methods'].items():
            rows.append(dict(scene=scene,method=m,audit='initial',threshold=r['threshold'],**r['metrics']))
    for scene,detail in b['details'].items():
        for m,r in detail.items():
            rows.append(dict(scene=scene,method=m,audit='matched_exclusion',threshold=r['threshold'],**r['metrics']))
    with (ROOT/'results'/'heterogeneous_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Single-report heterogeneous development audit', '',
           'Eight simulated families; 20 calibration and 40 evaluation acquisitions',
           'per family, each with null and mixed observations (960 observations).',
           'Seeds are paired across families and methods. These are development',
           'data, not 960 independent scenes or an untouched final test.', '',
           'Each method can report at most one secondary per pixel. A reported',
           'delay farther than two bins from truth is a wrong-depth error.',
           'Spatial metrics exclude five border pixels; every 9x9 donor',
           'neighborhood for a scored pixel stays inside the image.', '',
           '## Important failed superiority explanation', '',
           'With 6.5-bin true pulse FWHM and a 5-bin detector, joint3 box3 recall',
           'was zero, versus 0.3554 for neighborhood multiscale. After matching',
           'the minimum exclusion to eight bins, joint8 box3 recall was 0.5833',
           'and depth-guided7 recall was 0.6318. Thus the apparent advantage',
           'does NOT establish pulse-mismatch robustness of the proposal method.',
           'The detector exclusion was a major confound. Retain the original',
           'results for audit, but do not use them as a superiority claim.', '',
           '## Matched-exclusion results', '',
           'Thresholds are calibrated separately for each method and family:',
           'maximum of null/mixed 99th-percentile negative scores. Observed',
           'false-positive rates differ and are not claimed exactly matched.', '',
           '| Family | Method | Recall | Wrong depth | Absent-pixel FP | Null FP |',
           '|---|---|---:|---:|---:|---:|']
    for scene in b['details']:
        for m in ['box3_joint8','depth_guided_7_joint8','neighbor_multiscale']:
            r=(a['details'][scene]['methods'][m] if m=='neighbor_multiscale'
               else b['details'][scene][m])['metrics']
            lines.append(f"| {scene} | {m} | {r['recall']:.4f} | {r['wrong_depth_fp']:.4f} | {r['absent_pixel_fp']:.4f} | {r['null_fp']:.4f} |")
    lines+=['', 'No general sensitivity advantage is supported. Close six-bin',
            'returns remain a strong failure case for methods with eight-bin',
            'proposal exclusion. Wavelets do not consistently improve results.',
            'A separate exact-model likelihood-evidence prototype is under',
            'verification; these empirical-score results do not transfer to it.']
    (ROOT/'HETEROGENEOUS_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    scenes=['slope','pulse_mismatch','hole','close']
    labels=['Sloped primary','Pulse mismatch','Hole in secondary','Close returns']
    colors=['#2563eb','#15803d','#b45309']
    names=['Box 3, joint8','Depth guided 7, joint8','Neighbor multiscale']
    for j,m in enumerate(['box3_joint8','depth_guided_7_joint8','neighbor_multiscale']):
        data=[(a['details'][s]['methods'][m] if m=='neighbor_multiscale' else b['details'][s][m])['metrics'] for s in scenes]
        for ax,key in zip(axes,['recall','absent_pixel_fp']):
            ax.bar(np.arange(4)+(j-1)*.25,[r[key] for r in data],width=.24,color=colors[j],label=names[j])
    for ax in axes:
        ax.set_xticks(range(4),labels,rotation=22,ha='right');ax.spines[['top','right']].set_visible(False)
    axes[0].set(ylabel='Localized secondary recall',ylim=(0,1),title='Sensitivity at calibrated thresholds')
    axes[1].set(ylabel='False reports / absent pixel',ylim=(0,.03),title='Measured false reports in mixed scenes')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Development audit: matched 8-bin exclusion; one report per pixel',fontsize=11)
    folder=ROOT/'results'/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'heterogeneous_fairness.{ext}',dpi=180)
    print('Saved report, CSV, figure')


if __name__=='__main__':main()
