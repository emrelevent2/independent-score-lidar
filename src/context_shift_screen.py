"""Development audit of calibration transfer to unseen occupancy contexts.

Thresholds are copied from the flat geometry calibration, not refitted here.
The hole scenario distinguishes spatial borrowing from local photon evidence.
This experiment is exploratory, not the final held-out publication evaluation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import uniform_filter
from pilot import simulate,coates
from methods import depth_guided_pool,pooled_wavelet_proposer
from joint_detector import secondary_scores
from crossfit import propose,validate

root=Path(__file__).resolve().parents[1]
cal=json.loads((root/'results/geometry_screen_v3.json').read_text())['details']['flat']['results']
methods=['raw','box3','depth_guided_3','depth_guided_7',
         'neighbor_wavelet_7','neighbor_wavelet_9','neighbor_box_7','neighbor_box_9']
thresholds={k:cal[k]['threshold'] for k in methods}
out={}
for scenario in ['hole2','hole4','dense','empty']:
    mask=np.ones((16,16),bool)
    if scenario=='hole2': mask[7:9,7:9]=False
    elif scenario=='hole4': mask[6:10,6:10]=False
    elif scenario=='empty': mask[:]=False
    interior=np.zeros_like(mask); interior[2:-2,2:-2]=True
    records={k:[] for k in methods}
    for seed in range(7000,7100):
        h,_,_=simulate(seed,secondary_mask=mask)
        raw=coates(h)
        waveforms={'raw':raw,'box3':uniform_filter(raw,(3,3,1),mode='wrap'),
                   'depth_guided_3':depth_guided_pool(h,spatial_size=3),
                   'depth_guided_7':depth_guided_pool(h,spatial_size=7)}
        scoremaps={}
        for method,f in waveforms.items():
            z=secondary_scores(f)
            scoremaps[method]=(np.max(z,axis=-1),np.max(z[...,90:95],axis=-1))
        for method in methods[4:]:
            size=int(method.rsplit('_',1)[1]); wavelet='wavelet' in method
            f=pooled_wavelet_proposer(h,cycles=1000,spatial_size=size,
                    exclude_center=True,z=1.5 if wavelet else 0)
            p,s,valid=propose(f)
            z=validate(h,1000,p,s,valid,primary_consistency=True)
            scoremaps[method]=(z,np.where(np.abs(s-92)<=2,z,-np.inf))
        for method,(z,local) in scoremaps.items():
            entry={'seed':seed}
            if np.any(mask&interior): entry['recall']=float((local[mask&interior]>thresholds[method]).mean())
            if np.any(~mask&interior): entry['off_target_fp']=float((z[~mask&interior]>thresholds[method]).mean())
            records[method].append(entry)
    summary={}
    for method,entries in records.items():
        metrics={k:float(np.mean([x[k] for x in entries if k in x])) for k in ['recall','off_target_fp'] if any(k in x for x in entries)}
        summary[method]={'metrics':metrics,'scene_metrics':entries}
    out[scenario]={'mask':mask.tolist(),'results':summary}
    (root/'results/context_shift_screen.json').write_text(json.dumps({'status':'development audit only',
       'calibration_source':'geometry_screen_v3 flat, seeds 5000-5019',
       'thresholds':thresholds,'screen_seeds':[7000,7099],'details':out},indent=2))
    print(scenario,{k:v['metrics'] for k,v in summary.items()},flush=True)
