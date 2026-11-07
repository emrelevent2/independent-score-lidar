"""Development-only controls for detector exclusion-width confounding.

Neither original detector nor the proposed validator is changed here. Both
eight-bin-exclusion controls are calibrated separately at the acquisition level.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import uniform_filter
from pilot import simulate, coates
from methods import depth_guided_pool
from screen import detect_scores
from joint_detector import secondary_scores
from detector_suite import evaluate_report
from heterogeneous_screen import scene_spec

ROOT=Path(__file__).resolve().parents[1]


def process(h):
    x=coates(h)
    inputs={'raw':x, 'box3':uniform_filter(x,(3,3,1),mode='wrap'),
            'depth_guided_3':depth_guided_pool(h,spatial_size=3),
            'depth_guided_7':depth_guided_pool(h,spatial_size=7)}
    out={}
    for name,flux in inputs.items():
        for detector in ('joint8','residual8'):
            z=secondary_scores(flux,min_separation=8.) if detector=='joint8' else detect_scores(flux)
            delay=np.argmax(z,axis=-1)
            score=np.take_along_axis(z,delay[...,None],axis=-1)[...,0]
            out[f'{name}_{detector}']=(score,delay)
    return out


def main():
    result={'status':'development detector fairness audit', 'minimum_separation_bins':8,
            'calibration_seeds':[8000,8019], 'evaluation_seeds':[9000,9039],
            'code_sha256':{n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()
                           for n in ('fair_detector_audit.py','heterogeneous_screen.py',
                                     'pilot.py','methods.py','screen.py','joint_detector.py')},
            'details':{}}
    for scene in ('slope','pulse_mismatch','hole','close'):
        cfg,mask,target,interior=scene_spec(scene)
        cal={}; records={}
        for split,start,count in [('calibration',8000,20),('evaluation',9000,40)]:
            for seed in range(start,start+count):
                for null in (True,False):
                    pars=dict(cfg)
                    if null: pars['secondary']=0.
                    h,_,_=simulate(seed,**pars)
                    for m,(z,d) in process(h).items():
                        if split=='calibration':
                            cal.setdefault(m,{'null':[],'mixed':[]})['null' if null else 'mixed'].extend(
                                z[interior if null else ~mask&interior])
                        else:
                            records.setdefault(m,[]).append((seed,null,z,d))
            print(scene,split,'complete',flush=True)
        details={}
        for m,recs in records.items():
            threshold=max(float(np.sort(v)[int(np.ceil(.99*(len(v)-1)))]) for v in cal[m].values())
            entries=[]
            for seed,null,z,d in recs:
                metrics=evaluate_report(z,d,threshold,np.zeros_like(mask) if null else mask,target,interior)
                entries.append(dict(seed=seed,null=null,**metrics))
            keys=('recall','wrong_depth_fp','absent_pixel_fp')
            metrics={k:float(np.mean([e[k] for e in entries if not e['null'] and k in e]))
                     for k in keys}
            metrics['null_fp']=float(np.mean([e['absent_pixel_fp'] for e in entries if e['null']]))
            details[m]=dict(threshold=threshold,metrics=metrics,scene_metrics=entries)
        result['details'][scene]=details
        (ROOT/'results'/'fair_detector_audit.json').write_text(json.dumps(result,indent=2))
        print(scene,{m:r['metrics'] for m,r in details.items()},flush=True)


if __name__=='__main__':main()
