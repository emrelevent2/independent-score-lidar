"""Known-active control that interpolates exposure, retaining exact 1/lambda.

Predefined after auditing the first bank's interpolation error, before looking
at any efficacy result from this revised construction. Reports both versions.
"""
import sys,json,sqlite3,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
import revision_censored_bank as ab
import revision_censored_bank_common as common
P=ab.ROOT/'REVISION_EXPOSURE_BANK_PROTOCOL_20261107.json'
OUT=ab.ROOT/'results/revision_exposure_bank_v1'

def freeze():
    if P.exists():raise FileExistsError(P)
    ab.check();common.check()
    p=dict(status='FROZEN_BEFORE_REVISED_INTERPOLATION_EFFICACY',source_sha256=ab.sha(__file__),first_bank_protocol_sha256=ab.sha(ab.P),common_protocol_sha256=ab.sha(common.P),
      motivation='First known-active bank held-out abs log variance error median .0634, p95 .3202, max .6699. Exact division by lambda is available analytically and should not be approximated by grid interpolation.',
      construction='Same 40000 training paths, cells, seeds and averages. At each cell recover mean E from mean(E/lambda)*known lambda. Convex interpolation/rotation of mean exposure, then divide by exact current null rate lambda. Same phase-rate nuisance projection and safe localizer. Original and common calibration branches on all 640 records. Same 199 refits/lookup/scan repeated. Frozen first-bank results retained.',
      validation='Before efficacy run, held-out 32 profiles and 6400 independent paths used for first-bank audit. Report all 7232 variance directions. Predeclared engineering requirement: p95 abs log variance error below .3202, no nonpositive variances. If not met, report validation and do not substitute this version as more accurate.',
      audit_sha256=ab.sha(ab.ROOT/'results/revision_censored_bank_audit.json'))
    P.write_text(json.dumps(p,indent=2));print('Frozen',ab.sha(P),flush=True)

def check():
    p=json.loads(P.read_text());assert p['source_sha256']==ab.sha(__file__);ab.check();common.check();return p

class ExposureBank(ab.NullInformationBank):
    def lookup(self,theta,regime):
        h,flags=super().lookup(theta,regime);rate,_=ab.rates_and_design(theta,64,1.5)
        return np.diag(np.diag(h)/rate),flags

@lru_cache(1)
def bank():
    return ExposureBank.load(OUT/'exposure_bank.npz')

def train_validate():
    check();OUT.mkdir(exist_ok=True);bp=ab.oldbank.check();matrices=np.zeros((2,5,5,4,64,64))
    for c in ab.oldbank.cells(bp):
        row=json.loads((ab.OUT/'cells'/(c['id']+'.json')).read_text());rate,_=ab.rates_and_design(c['theta'],64,1.5)
        matrices[c['ri'],c['ai'],c['bi'],c['fi']]=np.diag(np.array(row['diagonal'])*rate)
    np.savez_compressed(OUT/'exposure_bank.npz',information=matrices,amplitudes=np.array([[g['a']*m for m in bp['grid_multipliers']] for g in bp['regimes']]),backgrounds=np.array([[g['b']*m for m in bp['grid_multipliers']] for g in bp['regimes']]),fractions=bp['fractional_phases'],regimes=[g['index'] for g in bp['regimes']],base_phase_integer=bp['base_phase_integer'])
    from scipy.linalg import qr
    from ct_full_score import templates
    import hashlib
    errors=[]
    for c in ab.oldbank.validation_cases(bp):
        g=bp['regimes'][c['ri']];rate,j=ab.rates_and_design(c['theta'],64,1.5);summed=np.zeros(64);digest=hashlib.sha256()
        for i in range(200):
            obs,_=ab.simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],c['seed']+1009*i);digest.update(obs.tobytes())
            _,e=ab.numeric_statistics(obs,g['cycles'],64,ab.outward_safe_lag(g['td'],g['te']));summed+=e/rate
        with np.load(ab.oldbank.OUT/'validation'/(str(c['index'])+'.npz'),allow_pickle=False) as f:assert digest.hexdigest()==json.loads(str(f['metadata']))['observed_arrays_sha256']
        h=np.diag(summed/200);hb,_=bank().lookup(c['theta'],g['index']);grid,t=templates(64,1.5)
        def variance(matrix):
            roots=np.sqrt(np.diag(matrix));q,_=qr(roots[:,None]*j,mode='economic');r=t*roots-(t*roots)@q@q.T;return np.sum(r*r,axis=1)
        v,vb=variance(h),variance(hb);allowed=np.abs((grid-c['theta'][1]+32)%64-32)>=3.75
        assert np.all(v[allowed]>0) and np.all(vb[allowed]>0);errors.extend(np.abs(np.log(vb[allowed]/v[allowed])).tolist())
    qs=np.quantile(errors,[.5,.95,1]);result=dict(status='PASS' if qs[1]<.3202 else 'APPROXIMATION_REQUIREMENT_FAILED',profiles=32,records=6400,directions=len(errors),absolute_log_variance_ratio_quantiles=dict(zip(['median','p95','max'],qs.tolist())),bank_sha256=ab.sha(OUT/'exposure_bank.npz'),protocol_sha256=ab.sha(P))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

def execute(item):
    branch,row=item;ab.bank=bank
    result=(ab.worker if branch=='original' else common.worker)(row)
    return branch,result

def run():
    check();assert json.loads((OUT/'validation.json').read_text())['status']=='PASS';start=time.perf_counter();source=ab.study.read()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={v[0] for v in db.execute('SELECT id FROM records')}
        todo=[(branch,r) for branch in ['original','common'] for r in source if branch+'_'+r['case']['id'] not in done]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(branch,r) in enumerate(pool.map(execute,todo)):
                db.execute('INSERT INTO records VALUES(?,?)',(branch+'_'+r['case']['id'],json.dumps(dict(branch=branch,**r))));db.commit()
                if (len(done)+i+1)%64==0:print('Exposure bank',len(done)+i+1,'/1280',flush=True)
    with sqlite3.connect(OUT/'records.sqlite3') as db:rows=[json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]
    assert len(rows)==1280;totals={}
    for branch in ['original','common']:
        rr=[r for r in rows if r['branch']==branch];assert [r['case'] for r in rr]==[r['case'] for r in source]
        totals[branch]={}
        for alpha in [.005,.01,.02]:
            t=dict(correct=0,wrong=0,false=0,positive=0,null=0)
            for a,b in zip(source,rr):
                z=b['sample'];assert z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200
                for k,v in ab.study.d.previous.old.endpoint(dict(a,samples={'new':z}),'new',alpha).items():t[k]+=v
            totals[branch][str(alpha)]=t
    result=dict(status='COMPLETE',cases=1280,totals=totals,records_sha256=ab.sha(OUT/'records.sqlite3'),protocol_sha256=ab.sha(P),seconds=time.perf_counter()-start)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':{'freeze':freeze,'validate':train_validate,'run':run}[sys.argv[1]]()
