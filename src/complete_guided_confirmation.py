from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,datetime,time,argparse
import numpy as np
from guided_alternatives import guided_pool,fit_pool
from complete_guided_likelihood import complete_guided_likelihood
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'guided_confirmation_v1'
OUT=ROOT/'results'/'guided_confirmation_v1'/'complete_guided';P=ROOT/'COMPLETE_GUIDED_COMPARATOR_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['complete_guided_confirmation.py','complete_guided_likelihood.py','guided_alternatives.py','adaptive_evidence.py','likelihood_baseline.py']
def freeze():
    if P.exists():raise RuntimeError('Comparator already frozen')
    p=dict(status='Additional strong comparator frozen before final accuracy inspection',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_protocol_sha256=sha(ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json'),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        notes=['Close nuisance-start loophole: primary full range0--255; starts at donor primary, best null and three queried primary peaks.',
            'Secondary refines within2bins of same guided donor proposal; three secondary starts.',
            'Same1999 calibration cases and0.005/0.01 order-statistic rules as every full-study calibrated control.',
            'Earlier simpler guided control retained; do not select the weaker baseline for claims.',
            'No test accuracy inspected; source settings will not be tuned on these data.'])
    P.write_text(json.dumps(p,indent=2))
def worker(item,digest):
    path=OUT/(item['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    h=np.load(DATA/(item['id']+'.npz'))['histogram'];pool,_=guided_pool(h,8,8);parameters=fit_pool(pool)
    report=complete_guided_likelihood(h[8,8],parameters['primary'],parameters['secondary'])
    out=dict(id=item['id'],protocol_sha256=digest,report=report);path.write_text(json.dumps(out,indent=2,allow_nan=False));return out
def run(workers):
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    cases=json.loads((DATA/'manifest.json').read_text())['cases'];OUT.mkdir(exist_ok=True);start=time.perf_counter()
    for split in ['calibration','evaluation']:
        subset=[c for c in cases if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,sha(P)) for c in subset]
            for i,j in enumerate(as_completed(jobs),1):
                j.result()
                if i%500==0 or i==len(jobs):print('Complete guided',split,i,'/',len(subset),'seconds',round(time.perf_counter()-start,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(cases=len(cases),protocol_sha256=sha(P),elapsed_seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    freeze() if args.action=='freeze' else run(args.workers)
