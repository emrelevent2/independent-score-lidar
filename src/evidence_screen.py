"""Pre-specified prototype verification, NOT a final publication benchmark."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.stats import beta
from pilot import simulate
from likelihood_evidence import neighbor_alternative,evaluate_evidence

ROOT=Path(__file__).resolve().parents[1]
ALPHA=.01


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def configuration(name):
    cfg=dict(size=16,secondary=0.,primary=30.,background=20.)
    if name=='high_background':cfg['background']=200.
    elif name=='bright_primary':cfg['primary']=100.
    elif name=='pileup':cfg.update(primary=600.,background=100.)
    elif name=='neighbor_surface':
        mask=np.ones((16,16),bool);mask[8,8]=False
        cfg.update(secondary=10.,secondary_mask=mask)
    elif name.startswith('alternative_'):cfg['secondary']=float(name.split('_')[1])
    elif name!='nominal':raise ValueError(name)
    return cfg


def main():
    names=['nominal','high_background','bright_primary','pileup','neighbor_surface',
           'alternative_3','alternative_6','alternative_10']
    manifest={'status':'prototype verification; development only',
              'alpha':ALPHA,'decision_rule':'log_e >= log(1/alpha)',
              'known_fwhm':5.,'null_delay_grid':'all integer bins 0 to 255',
              'tested_pixel_zero_based':[8,8], 'neighbor_size':9,
              'source_sha256':{n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()
                  for n in ('evidence_screen.py','likelihood_evidence.py','pilot.py',
                            'crossfit.py','screen.py')},'details':{}}
    start=time.perf_counter()
    for name in names:
        cfg=configuration(name)
        first,count=(11000,150) if not name.startswith('alternative_') else (12000,120)
        records=[]
        for seed in range(first,first+count):
            h,_,_=simulate(seed,**cfg)
            alternative,p,s,valid=neighbor_alternative(h,size=9)
            t=time.perf_counter()
            r=evaluate_evidence(h[8,8],alternative[8,8])
            r.update(seed=seed,primary_proposal=int(p[8,8]),secondary_proposal=int(s[8,8]),
                     valid_proposal=bool(valid[8,8]),seconds=time.perf_counter()-t)
            r['declared']=bool(valid[8,8] and r['log_e']>=np.log(1/ALPHA))
            r['localized']=bool(r['declared'] and abs(r['secondary_proposal']-92)<=2)
            records.append(r)
            if len(records)%30==0:
                manifest['details'][name]={'records':records,'in_progress':True}
                (ROOT/'results'/'likelihood_evidence_screen.json').write_text(json.dumps(manifest,indent=2))
                print(name,len(records),'of',count,flush=True)
        detected=sum(r['declared'] for r in records)
        localized=sum(r['localized'] for r in records)
        summary=dict(n=count,declared=detected,declaration_rate=detected/count,
                     declaration_rate_ci95=interval(detected,count),
                     localized=localized,localized_rate=localized/count,
                     localized_rate_ci95=interval(localized,count),
                     global_gap_max=max(r['global_gap'] for r in records),
                     fallback_count=sum(r['fallback_count'] for r in records),
                     mean_bound_seconds=float(np.mean([r['seconds'] for r in records])))
        manifest['details'][name]={'records':records,'summary':summary,'in_progress':False}
        manifest['elapsed_seconds']=time.perf_counter()-start
        (ROOT/'results'/'likelihood_evidence_screen.json').write_text(json.dumps(manifest,indent=2))
        print(name,summary,flush=True)


if __name__=='__main__':main()
