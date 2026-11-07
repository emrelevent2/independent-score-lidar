"""Original-generator censored control, including full-model support failures.

The known-active criterion can remain computable when a gap violates the full
model's nominal recovery support. Do not discard that comparator advantage.
"""
import json,sqlite3,sys,hashlib,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import revision_sensitivity as r
P=r.ROOT/'REVISION_SAFE_ROBUSTNESS_PROTOCOL_20261107.json'
OUT=r.ROOT/'results/revision_safe_robustness_v1'

def check():
    p=json.loads(P.read_text());assert p['source_sha256']==r.sha(__file__);assert p['parent_protocol_sha256']==r.sha(r.P);return r.check()

def freeze():
    if P.exists():raise FileExistsError(P)
    r.check();p=dict(status='FROZEN_BEFORE_ORIGINAL_GENERATOR_ROBUSTNESS_CONTROL',source_sha256=r.sha(__file__),parent_protocol_sha256=r.sha(r.P),
      design='All 864 fixed mismatch acquisitions, same hashes and truth. The original known-active criterion uses its own null fit as the bootstrap generator, nominal Td/Te and 199 complete paths, same paired seed formula. Repeat safe fit and quarter-grid score in every draw; observed-only safe localizer. This control remains applicable to below-nominal-support gaps. Only timestamps outside [0,H) make its acquisition domain incompatible. Main full/common-generator records remain unchanged.',
      reason='A support-invalid full likelihood also prevents common-generator calibration. A censored criterion with its original generator can still operate and must be reported, rather than inheriting the full model support failure. This companion was specified from model structure before inspecting its efficacy outcomes.',workers=2)
    P.write_text(json.dumps(p,indent=2));print('Frozen',r.sha(P),flush=True)

def worker(case):
    p=check();g=p['regime'];obs,truth,theta=r.acquisition(case,p)
    row=dict(case=case,truth=truth,data_sha256=hashlib.sha256(obs.tobytes()).hexdigest())
    if len(obs) and (obs[0]<0 or obs[-1]>=64000):return dict(row,status='TIMESTAMP_DOMAIN_INCOMPATIBLE')
    lag=r.outward_safe_lag(31.7,80.);s,e=r.numeric_statistics(obs,1000,64,lag);fit=r.fit_null(s,e,1.5)
    v=r.fine_scan(s,e,1.5,fit=fit);loc=r.localize(s,e,1.5,fit,v['delay']);rate,_=r.rates_and_design(fit['theta'],64,1.5)
    boots=[];warnings=[]
    for j in range(199):
        bo,_=r.simulate_two_dead_times(rate,1000,31.7,80.,p['bootstrap_seed_base']+1000003*case['index']+1009*j)
        bs,be=r.numeric_statistics(bo,1000,64,lag);bf=r.refit_from_start(bs,be,fit['theta'],1.5,'duration');boots.append(r.fine_scan(bs,be,1.5,fit=bf)['score'])
        if not bf['optimizer_success']:warnings.append(j)
    sample=dict(score=v['score'],primary=loc['primary'],delay=loc['delay'],pvalue=r.pvalue(v['score'],boots),bootstrap_scores=boots)
    return dict(row,status='EVALUATED',samples={'safe_original':sample},null_fit=r.d.json_fit(fit),localizer=loc,warning_draws=warnings)

def run():
    p=check();OUT.mkdir(exist_ok=True);expected=list(r.cases(p,'robust'));start=time.perf_counter()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');done={a[0] for a in db.execute('SELECT id FROM records')}
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i,v in enumerate(pool.map(worker,[c for c in expected if c['id'] not in done])):
                db.execute('INSERT INTO records VALUES(?,?)',(v['case']['id'],json.dumps(v)));db.commit()
                if (len(done)+i+1)%64==0:print('Original censored robustness',len(done)+i+1,'/864',flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(status='COMPLETE',cases=864,records_sha256=r.sha(OUT/'records.sqlite3'),protocol_sha256=r.sha(P),seconds=time.perf_counter()-start),indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
