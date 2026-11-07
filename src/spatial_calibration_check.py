"""Development audit: calibrate on absent pixels in mixed-return scenes.

Earlier calibration contains no secondary surface anywhere, whereas screening
negative pixels can have neighbors containing one. This audit tests that shift.
The entire 4000-4029 seed set is development data already inspected. Reusing
parts for this audit does not create a fresh held-out test set.
"""
from pathlib import Path
import json
import numpy as np
import h5py
from scipy.io import loadmat
from pilot import simulate
from methods import all_methods,pooled_wavelet_proposer
from joint_detector import secondary_scores
from crossfit import crossfit_scores,propose,validate

root=Path(__file__).resolve().parents[1]
condition='nominal'; target=92
data=loadmat(root/'data/screen_nominal.mat')['observations']
meta=json.loads((root/'data/screen_nominal.json').read_text())
split=loadmat(root/'data/split_nominal.mat')
with h5py.File(root/'results/screen_caspi_nominal.mat') as f:
    original=np.asarray(f['fluxes']).transpose(3,2,1,0)
with h5py.File(root/'results/split_caspi_nominal.mat') as f:
    left=np.asarray(f['flux_left']).transpose(3,2,1,0)
    right=np.asarray(f['flux_right']).transpose(3,2,1,0)
_,_,mask=simulate(601)
interior=np.zeros(mask.shape,bool); interior[2:-2,2:-2]=True
positive=mask&interior; negative=~mask&interior
near=np.abs(np.arange(256)-target)<=2
calibration={}; records={}
for i,sample in enumerate(meta['samples']):
    if sample['split']=='calibration': continue
    h=data[...,i]
    output=all_methods(h); output['original_CASPI']=original[...,i]
    scoremaps={}
    for name,flux in output.items():
        z=secondary_scores(flux)
        scoremaps[name]=(np.max(z,axis=-1),np.max(z[...,near],axis=-1))
    z,delays=crossfit_scores(split['left'][...,i],split['right'][...,i],left[...,i],right[...,i])
    scoremaps['crossfit']=(z.max(-1),np.max(np.where(np.abs(delays-target)<=2,z,-np.inf),axis=-1))
    p,s,valid=propose(original[...,i])
    z_full=validate(h,1000,p,s,valid)
    scoremaps['full_data_gate']=(z_full,np.where(np.abs(s-target)<=2,z_full,-np.inf))
    for size in [3,7,9,15]:
        a=pooled_wavelet_proposer(split['left'][...,i],spatial_size=size)
        b=pooled_wavelet_proposer(split['right'][...,i],spatial_size=size)
        z,d=crossfit_scores(split['left'][...,i],split['right'][...,i],a,b)
        scoremaps[f'wavelet_crossfit_{size}']=(z.max(-1),np.max(np.where(np.abs(d-target)<=2,z,-np.inf),axis=-1))
    if not sample['null'] and sample['seed']<4015:
        for name,(allscore,localized) in scoremaps.items():
            calibration.setdefault(name,[]).extend(allscore[negative].ravel())
    else:
        for name,(allscore,localized) in scoremaps.items():
            records.setdefault(name,[]).append({'seed':sample['seed'],'null':sample['null'],
                 'allscore':allscore,'localized':localized})
results={}
for name,entries in records.items():
    threshold=float(np.quantile(calibration[name],.99,method='higher'))
    scene=[]
    for entry in entries:
        if entry['null']:
            scene.append({'seed':entry['seed'],'null_fpr':float((entry['allscore'][interior]>threshold).mean())})
        else:
            scene.append({'seed':entry['seed'],
                'weak_recall':float((entry['localized'][positive]>threshold).mean()),
                'off_target_fp':float((entry['allscore'][negative]>threshold).mean())})
    metrics={k:float(np.mean([s[k] for s in scene if k in s])) for k in ['null_fpr','weak_recall','off_target_fp']}
    results[name]={'threshold':threshold,'metrics':metrics,'scene_metrics':scene}
output={'status':'development recalibration audit only',
 'calibration':'absent pixels in mixed-return scenes, seeds 4000-4014',
 'evaluation':'mixed scenes 4015-4029; null scenes 4000-4029',
 'detector':'joint nuisance projection for baseline methods; independent-cycle proposed-delay score for crossfit',
 'details':results}
(root/'results/spatial_calibration_check.json').write_text(json.dumps(output,indent=2))
print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))
