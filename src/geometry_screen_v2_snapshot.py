"""New development geometries, mixed-scene and empty-scene calibration.

Geometry is supplied only to the simulator and evaluation, not the methods.
"""
from pathlib import Path
import json
import hashlib
import time
import argparse
import numpy as np
from scipy.ndimage import uniform_filter
from pilot import simulate,coates
from methods import pooled_wavelet_proposer,depth_guided_pool
from photon_split import split_cycles
from crossfit import crossfit_scores,propose,validate,fuse_agreeing
from joint_detector import secondary_scores

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--revision2',action='store_true')
args=parser.parse_args()
tag='geometry_screen_v2' if args.revision2 else 'geometry_screen'


def geometry(name):
    center=np.full((16,16),80.)
    mask=np.zeros((16,16),bool)
    if name=='flat': mask[4:12,4:12]=True
    elif name=='depth_step': center[:,8:]=92; mask[4:12,4:8]=True
    elif name=='thin': mask[3:13,7:9]=True
    elif name=='point': mask[7:9,7:9]=True
    else: raise ValueError(name)
    return center,mask


def process(h,split_seed):
    raw=coates(h)
    out={}
    for name,flux in [('raw',raw),('box3',uniform_filter(raw,(3,3,1),mode='wrap'))]:
        z=secondary_scores(flux)
        # One score per possible reported delay, preserving the detector output.
        out[name]=(z,np.broadcast_to(np.arange(256),z.shape))
    if args.revision2:
        for size in (3,7):
            f=depth_guided_pool(h,spatial_size=size)
            z=secondary_scores(f)
            out[f'depth_guided_{size}']=(z,np.broadcast_to(np.arange(256),z.shape))
    a,b,_,_=split_cycles(h,1000,split_seed)
    for size in (3,7,9,15):
        fa=pooled_wavelet_proposer(a,spatial_size=size)
        fb=pooled_wavelet_proposer(b,spatial_size=size)
        for check in (False,True):
            z,d=crossfit_scores(a,b,fa,fb,primary_consistency=check)
            out[f'wavelet_crossfit_{size}'+('_consistent' if check else '')]=(z,d)
            if args.revision2 and check:
                out[f'wavelet_crossfit_{size}_fused']=fuse_agreeing(z,d)
    return out


all_results={}
for name in ['flat','depth_step','thin','point']:
    center,mask=geometry(name); target=center+12
    interior=np.zeros((16,16),bool); interior[2:-2,2:-2]=True
    pos=mask&interior; neg=~mask&interior
    calibration={}; scenes={}
    for split,first,n in [('calibration',5000,20),('screen',6000,30)]:
        for seed in range(first,first+n):
            for null in (True,False):
                h,_,_=simulate(seed,secondary=0. if null else 3.,primary_delay_map=center,secondary_mask=mask)
                outputs=process(h,200000+seed*2+int(null))
                for method,(z,d) in outputs.items():
                    anyscore=np.max(z,axis=-1)
                    near=np.abs(d-target[...,None])<=2
                    localized=np.max(np.where(near,z,-np.inf),axis=-1)
                    if split=='calibration':
                        # Pool null and off-target mixed calibration observations.
                        # This does not guarantee uniform contextual validity.
                        context='null' if null else 'mixed'
                        calibration.setdefault(method,{'null':[],'mixed':[]})[context].extend(anyscore[interior if null else neg].ravel())
                    else:
                        scenes.setdefault(method,[]).append({'seed':seed,'null':null,
                          'allscore':anyscore,'localized':localized})
        print(name,split,'complete',flush=True)
    results={}
    for method,records in scenes.items():
        if args.revision2:
            threshold=max(float(np.quantile(v,.99,method='higher')) for v in calibration[method].values())
        else:
            threshold=float(np.quantile(calibration[method]['null']+calibration[method]['mixed'],.99,method='higher'))
        entries=[]
        for record in records:
            entry={'seed':record['seed']}
            if record['null']: entry['null_fpr']=float((record['allscore'][interior]>threshold).mean())
            else:
                entry['weak_recall']=float((record['localized'][pos]>threshold).mean())
                entry['off_target_fp']=float((record['allscore'][neg]>threshold).mean())
            entries.append(entry)
        results[method]={'threshold':threshold,'scene_metrics':entries,
            'metrics':{k:float(np.mean([x[k] for x in entries if k in x])) for k in ['null_fpr','weak_recall','off_target_fp']}}
    all_results[name]={'primary_centers':center.tolist(),'weak_mask':mask.tolist(),'results':results}
    hashes={f:hashlib.sha256((root/'src'/f).read_bytes()).hexdigest() for f in ['geometry_screen.py','crossfit.py','methods.py','pilot.py','joint_detector.py']}
    (root/'results'/f'{tag}.json').write_text(json.dumps({'status':'development only',
        'calibration_rule':'max of per-context 99th percentiles' if args.revision2 else 'pooled context percentile',
        'calibration_seeds':[5000,5019],'screen_seeds':[6000,6029],
        'code_sha256':hashes,'calibration_target':.01,'details':all_results},indent=2))
    print(name,{k:v['metrics'] for k,v in results.items()},flush=True)
