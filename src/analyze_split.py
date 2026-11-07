from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import h5py
from scipy.io import loadmat
from crossfit import crossfit_scores
from pilot import simulate

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--condition',default='nominal')
args=parser.parse_args(); condition=args.condition
manifest=json.loads((root/'data'/f'split_{condition}.json').read_text())
data=loadmat(root/'data'/f'split_{condition}.mat')
path=root/'results'/f'split_caspi_{condition}.mat'
with h5py.File(path) as f:
    a=np.asarray(f['flux_left']).transpose(3,2,1,0)
    b=np.asarray(f['flux_right']).transpose(3,2,1,0)
    actual_cycles=np.asarray(f['cycles']).ravel()
assert np.array_equal(actual_cycles,[500,500])
scores=[]; delays=[]; calibration=[]
for i,sample in enumerate(manifest['samples']):
    z,d=crossfit_scores(data['left'][...,i],data['right'][...,i],a[...,i],b[...,i])
    scores.append(z); delays.append(d)
    if sample['split']=='calibration': calibration.extend(z[2:-2,2:-2].max(-1).ravel())
threshold=float(np.quantile(calibration,.99,method='higher'))
entries=[]
for i,sample in enumerate(manifest['samples']):
    if sample['split']=='calibration': continue
    z=scores[i]; d=delays[i]
    _,_,mask=simulate(sample['seed'],**manifest['config'])
    interior=np.zeros(mask.shape,bool); interior[2:-2,2:-2]=True
    hit=z>threshold; prediction=hit.any(-1)
    entry={'seed':sample['seed']}
    if sample['null']: entry['null_fpr']=float(prediction[interior].mean())
    else:
        target=80+manifest['config']['separation']
        localized=(hit&(np.abs(d-target)<=2)).any(-1)
        entry.update(weak_recall=float(localized[mask&interior].mean()),
                     off_target_fp=float(prediction[interior&~mask].mean()))
    entries.append(entry)
metrics={k:float(np.mean([s[k] for s in entries if k in s])) for k in ['null_fpr','weak_recall','off_target_fp']}
output={'status':'development only; no guarantee or held-out result',
 'method':'bidirectional independent-cycle CASPI candidate validation',
 'condition':condition,'threshold':threshold,'metrics':metrics,'scene_metrics':entries,
 'source_sha256':hashlib.sha256((root/'src/crossfit.py').read_bytes()).hexdigest()}
(root/'results'/f'split_analysis_{condition}.json').write_text(json.dumps(output,indent=2))
print(json.dumps({k:v for k,v in output.items() if k!='scene_metrics'},indent=2))
