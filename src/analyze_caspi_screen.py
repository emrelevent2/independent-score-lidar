"""Analyze completed upstream batches with the same detector and observations."""
from pathlib import Path
import json
import csv
import hashlib
import argparse
import numpy as np
import h5py
from scipy.io import loadmat
from pilot import simulate
from screen import detect_scores, CONDITIONS

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--detector',choices=['residual','joint'],default='residual')
args=parser.parse_args()
if args.detector=='joint':
    from joint_detector import secondary_scores
    score_function=secondary_scores
else: score_function=detect_scores
tag='caspi_screen_analysis' if args.detector=='residual' else 'caspi_screen_joint'
rows=[]; outputs={}; missing=[]
for condition,cfg in CONDITIONS.items():
    path=root/'results'/f'screen_caspi_{condition}.mat'
    if not path.exists(): missing.append(condition); continue
    manifest=json.loads((root/'data'/f'screen_{condition}.json').read_text())
    observations=loadmat(root/'data'/f'screen_{condition}.mat')['observations']
    h,_,_=simulate(3000,**{**cfg,'secondary':0.})
    assert np.array_equal(h,observations[...,0])
    with h5py.File(path) as f:
        fluxes=np.asarray(f['fluxes']).transpose(3,2,1,0)
        runtimes=np.asarray(f['runtimes']).ravel()
    assert fluxes.shape==observations.shape and np.isfinite(fluxes).all()
    nullscores=[]; scene=[]
    for index,sample in enumerate(manifest['samples']):
        f=fluxes[...,index]; scores=score_function(f)
        if sample['split']=='calibration':
            nullscores.extend(np.max(scores[2:-2,2:-2],axis=-1).ravel())
    threshold=float(np.quantile(nullscores,.99,method='higher'))
    for index,sample in enumerate(manifest['samples']):
        if sample['split']=='calibration': continue
        h,truth,mask=simulate(sample['seed'],**({**cfg,'secondary':0.} if sample['null'] else cfg))
        assert np.array_equal(h,observations[...,index])
        scores=score_function(fluxes[...,index]); pred=np.max(scores,axis=-1)>threshold
        interior=np.zeros(mask.shape,bool); interior[2:-2,2:-2]=True
        if sample['null']:
            entry={'seed':sample['seed'],'null_fpr':float(pred[interior].mean())}
        else:
            near=np.abs(np.arange(f.shape[-1])-(80+cfg['separation']))<=2
            localized=np.max(scores[...,near],axis=-1)>threshold
            roi=mask&interior
            entry={'seed':sample['seed'],'weak_recall':float(localized[roi].mean()),
                   'off_target_fp':float(pred[interior&~mask].mean()),
                   'mse':float(np.mean((fluxes[...,index]-truth)**2))}
        scene.append(entry)
    row={'condition':condition,'method':'original_CASPI','threshold':threshold,
         'mean_runtime_seconds':float(runtimes.mean())}
    for metric in ['null_fpr','weak_recall','off_target_fp','mse']:
        row[metric]=float(np.mean([s[metric] for s in scene if metric in s]))
    rows.append(row)
    outputs[condition]={'metrics':row,'scene_metrics':scene,
                        'input_sha256':hashlib.sha256((root/'data'/f'screen_{condition}.mat').read_bytes()).hexdigest()}
dest=root/'results'
(dest/f'{tag}.json').write_text(json.dumps({'status':'development comparison only','detector':args.detector,
 'completed_conditions':list(outputs),'missing_conditions':missing,'details':outputs},indent=2))
if rows:
    with (dest/f'{tag}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps({'completed':rows,'missing':missing},indent=2))
