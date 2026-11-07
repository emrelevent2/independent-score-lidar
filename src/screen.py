"""Development screening, not final test data. Common detector for all methods.

The reported false-positive rate is per queried pixel across all searched delay
bins, not per time bin. Interpixel correlation requires scene-level uncertainty.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import hashlib
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks
from pilot import simulate
from methods import all_methods,block_wiener_candidate

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS={
 'nominal':dict(primary=30.,secondary=3.,background=20.,separation=12.),
 'bright_primary':dict(primary=100.,secondary=3.,background=20.,separation=12.),
 'high_background':dict(primary=30.,secondary=3.,background=200.,separation=12.),
 'weak_secondary':dict(primary=30.,secondary=1.,background=20.,separation=12.),
 'close_returns':dict(primary=30.,secondary=3.,background=20.,separation=6.),
 'pileup':dict(primary=600.,secondary=30.,background=100.,separation=12.)}


def detect_scores(flux):
    """Fit primary response, then search residual matched-filter local maxima.

    No true target delay is supplied. A minimum 8-bin separation is a detector
    limitation and intentionally makes the 6-bin condition an adverse case.
    """
    t=np.arange(-10,11)
    sigma=5/(2*np.sqrt(2*np.log(2)))
    g=np.exp(-.5*(t/sigma)**2); g/=g.sum()
    background=np.median(flux,axis=-1,keepdims=True)
    y=flux-background
    mf=convolve1d(y,g,axis=-1,mode='constant')/np.sum(g*g)
    primary=np.argmax(mf,axis=-1)
    amp=np.take_along_axis(mf,primary[...,None],axis=-1)
    dt=np.arange(flux.shape[-1])-primary[...,None]
    pulse=np.exp(-.5*(dt/sigma)**2)/np.sum(np.exp(-.5*(t/sigma)**2))
    residual=y-amp*pulse
    score=convolve1d(residual,g,axis=-1,mode='constant')/np.sum(g*g)
    valid=(np.abs(dt)>=8)&(np.arange(flux.shape[-1])>=10)&(np.arange(flux.shape[-1])<flux.shape[-1]-10)
    maxima=(score>=np.roll(score,1,axis=-1))&(score>np.roll(score,-1,axis=-1))
    score=np.where(valid&maxima,score,-np.inf)
    return score


def run(calibration=20,evaluation=30,conditions=None,tag='development_screen',revision2=False,detector='residual'):
    rows=[]; details={}
    if detector=='joint':
        from joint_detector import secondary_scores
        score_function=secondary_scores
    elif detector=='residual': score_function=detect_scores
    else: raise ValueError(detector)
    def process(h):
        outputs=all_methods(h)
        if revision2:
            outputs['block_wiener_v2']=block_wiener_candidate(h)
        return outputs
    selected=CONDITIONS if conditions is None else {k:CONDITIONS[k] for k in conditions}
    for condition,kw in selected.items():
        print('START',condition,flush=True)
        calibration_scores={}
        for i in range(calibration):
            nullkw={**kw,'secondary':0.}
            h,_,_=simulate(3000+i,**nullkw)
            for name,f in process(h).items():
                # Ignore spatial border in summary, shared by all methods.
                scores=score_function(f)[2:-2,2:-2]
                calibration_scores.setdefault(name,[]).extend(np.max(scores,axis=-1).ravel())
        thresholds={k:float(np.quantile(v,.99,method='higher')) for k,v in calibration_scores.items()}
        aggregate={k:[] for k in thresholds}
        for i in range(evaluation):
            for negative in (True,False):
                cfg={**kw,'secondary':0.} if negative else kw
                # Paired across methods; null/alternative are not assumed paired
                # photons across changed probability vectors.
                h,truth,mask=simulate(4000+i,**cfg)
                for name,f in process(h).items():
                    scores=score_function(f)
                    pred=np.max(scores,axis=-1)>thresholds[name]
                    target=80+kw['separation']
                    near=np.abs(np.arange(f.shape[-1])-target)<=2
                    localized=np.max(scores[...,near],axis=-1)>thresholds[name]
                    roi=mask.copy(); roi[:2]=False; roi[-2:]=False; roi[:,:2]=False; roi[:,-2:]=False
                    interior=np.zeros(mask.shape,bool); interior[2:-2,2:-2]=True
                    if negative:
                        entry={'seed':4000+i,'null_fpr':float(pred[interior].mean())}
                    else:
                        entry={'seed':4000+i,'weak_recall':float(localized[roi].mean()),
                               'off_target_fp':float(pred[interior&~mask].mean()),
                               'mse':float(np.mean((f-truth)**2))}
                    aggregate[name].append(entry)
        for name,entries in aggregate.items():
            row={'condition':condition,'method':name,'threshold':thresholds[name]}
            for metric in ('null_fpr','weak_recall','off_target_fp','mse'):
                values=[v[metric] for v in entries if metric in v]
                row[metric]=float(np.mean(values))
            rows.append(row)
        details[condition]={'configuration':kw,'thresholds':thresholds,'scene_metrics':aggregate}
        print('DONE',condition,flush=True)
        dest=ROOT/'results'; dest.mkdir(exist_ok=True)
        code_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['methods.py','pilot.py','screen.py','joint_detector.py']}
        (dest/f'{tag}.json').write_text(json.dumps({'status':'development screening only','detector':detector,'code_sha256':code_hashes,
            'calibration_seeds':[3000,3000+calibration-1],'screen_seeds':[4000,4000+evaluation-1],
            'target_calibration_pixel_fpr':.01,'details':details},indent=2))
        with (dest/f'{tag}.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--calibration',type=int,default=20)
    parser.add_argument('--evaluation',type=int,default=30)
    parser.add_argument('--tag',default='development_screen')
    parser.add_argument('--revision2',action='store_true')
    parser.add_argument('--detector',choices=['residual','joint'],default='residual')
    args=parser.parse_args()
    start=time.perf_counter(); run(args.calibration,args.evaluation,tag=args.tag,revision2=args.revision2,detector=args.detector)
    print('Elapsed',time.perf_counter()-start,flush=True)
