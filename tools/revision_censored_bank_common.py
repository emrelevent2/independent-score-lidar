"""Common-generator companion to the frozen known-active-bank ablation."""
import sys,json,hashlib,sqlite3,time,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import revision_censored_bank as ab
P=ab.ROOT/'REVISION_CENSORED_BANK_COMMON_PROTOCOL_20261107.json'
OUT=ab.ROOT/'results/revision_censored_bank_common_v1'

def check():
    p=json.loads(P.read_text());assert p['source_sha256']==ab.sha(__file__)
    assert p['original_ablation_protocol_sha256']==ab.sha(ab.P);ab.check();return p

def freeze():
    if P.exists():raise FileExistsError(P)
    ab.check();p=dict(status='FROZEN_BEFORE_COMMON_GENERATOR_ABLATION',source_sha256=ab.sha(__file__),original_ablation_protocol_sha256=ab.sha(ab.P),bank_sha256=ab.sha(ab.OUT/'bank.npz'),
       design='Same observed censored-bank statistic and safe localizer as original-generator ablation. Replace only calibration generator and bootstrap warm start by original common full-null fit. Regenerate all 199 common paths; verify exact path hashes and original-common safe scores against frozen rows. All 640 rows, no-hidden exact copies. This closes the generator-matched factorial comparison; mechanistic reanalysis, not independent confirmation.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',ab.sha(P),flush=True)

def worker(row):
    p=ab.study.check();case=row['case'];g=p['regimes'][case['regime']]
    if g['td']>=g['te']:
        return dict(case=case,data_sha256=row['data_sha256'],sample=row['samples']['localized_safe_score_common'],mode='EXACT_NOHIDDEN_COPY')
    obs,_,_,_,_,digest=ab.study.d.previous.old.base.data(case,p);assert digest==row['data_sha256']
    lag=ab.outward_safe_lag(g['td'],g['te']);s,e=ab.numeric_statistics(obs,g['cycles'],64,lag)
    v,flags=ab.statistic(s,e,row['safe_null'],case['regime']);loc=ab.localize(s,e,1.5,row['safe_null'],v['delay'])
    theta=row['calibration_null']['theta'];rate,_=ab.rates_and_design(theta,64,1.5);boots=[];h=hashlib.sha256();warnings=[]
    for draw in range(199):
        bo,_=ab.simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);h.update(bo.tobytes())
        bs,be=ab.numeric_statistics(bo,g['cycles'],64,lag);bf=ab.refit_from_start(bs,be,theta,1.5,'duration')
        z,_=ab.statistic(bs,be,bf,case['regime']);boots.append(z['score'])
        assert abs(ab.fine_scan(bs,be,1.5,fit=bf)['score']-row['samples']['safe_score_common']['bootstrap_scores'][draw])<1e-9
        if not bf['optimizer_success']:warnings.append(draw)
    assert h.hexdigest()==row['bootstrap_path_sha256']['common']
    v.update(primary=loc['primary'],delay=loc['delay'],pvalue=ab.pvalue(v['score'],boots),bootstrap_scores=boots)
    return dict(case=case,data_sha256=digest,sample=v,localization=loc,bank_flags=flags,warning_draws=warnings,bootstrap_path_sha256=h.hexdigest())

def run():
    check();OUT.mkdir(exist_ok=True);start=time.perf_counter();source=ab.study.read()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={v[0] for v in db.execute('SELECT id FROM records')};todo=[r for r in source if r['case']['id'] not in done]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,r in enumerate(pool.map(worker,todo)):
                db.execute('INSERT INTO records VALUES(?,?)',(r['case']['id'],json.dumps(r)));db.commit()
                if (len(done)+i+1)%32==0:print('Common censored bank',len(done)+i+1,'/640',flush=True)
    with sqlite3.connect(OUT/'records.sqlite3') as db:rows=[json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]
    assert len(rows)==640;totals={};groups={}
    for alpha in [.005,.01,.02]:
        t=dict(correct=0,wrong=0,false=0,positive=0,null=0);regs={str(i):dict(t) for i in range(4)}
        for a,b in zip(source,rows):
            assert a['case']==b['case'];z=b['sample'];assert z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200
            ep=ab.study.d.previous.old.endpoint(dict(a,samples={'new':z}),'new',alpha)
            for k,v in ep.items():t[k]+=v;regs[str(a['case']['regime'])][k]+=v
        totals[str(alpha)]=t;groups[str(alpha)]=regs
    result=dict(status='COMPLETE',cases=640,totals=totals,by_regime=groups,protocol_sha256=ab.sha(P),records_sha256=ab.sha(OUT/'records.sqlite3'),seconds=time.perf_counter()-start)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
