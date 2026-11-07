"""Independent known-active information ablation, isolated from frozen studies.

Training uses exactly the old bank's null paths, verified by their saved hashes.
It estimates E[E_r/lambda_r], the information of the known-active score, rather
than borrowing the full-record information matrix. All 199 adaptive safe-null
refits and bank lookups are repeated in each query's own calibration law.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import sys, json, hashlib, sqlite3, time, os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import train_null_information_bank as oldbank
import score_peak_localization_development as study
from null_information_bank import NullInformationBank
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from deadtime_path_bootstrap import refit_from_start, pvalue, fine_scan
from score_peak_localization import localize
from ct_full_score import templates
P=ROOT/'REVISION_CENSORED_BANK_PROTOCOL_20261107.json'
OUT=ROOT/'results/revision_censored_bank_v1'
sha=oldbank.sha

def freeze():
    if P.exists(): raise FileExistsError(P)
    study.check(); oldbank.check()
    p=dict(status='FROZEN_BEFORE_NEW_ABLATION',source_sha256=sha(__file__),
      original_study_protocol_sha256=sha(study.P),original_bank_protocol_sha256=sha(oldbank.P),
      original_data_sha256=sha(study.OUT/'records.sqlite3'),
      training='Same 40000 independent null paths/cells/seeds as full bank. Verify all 200 original path hashes. Average diagonal E_r/lambda_r, not full OPG.',
      query='All 640 original queries. Same known-active pilot, score gradient, quarter-grid, exclusion, 1e-12 nuisance pseudoinverse and safe localizer as original localized safe control. Replace only information weights by independent known-active bank. No-hidden controls copied exactly.',
      calibration='199 original-safe-null paths per query, original seeds. Repeat safe-null fit, bank lookup and score scan on every draw. Observed-only safe localization.',
      inference='Paired mechanistic ablation on existing data, not an independent confirmation.',
      workers=4,threads={k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    P.write_text(json.dumps(p,indent=2)); print('Frozen',sha(P),flush=True)

def check():
    p=json.loads(P.read_text()); assert p['source_sha256']==sha(__file__)
    assert p['original_data_sha256']==sha(study.OUT/'records.sqlite3')
    study.check(); oldbank.check(); return p

def train_cell(cell):
    bp=oldbank.check(); g=bp['regimes'][cell['ri']]
    rate,_=rates_and_design(cell['theta'],64,1.5)
    total=np.zeros(64); h=hashlib.sha256()
    for i in range(bp['records_per_cell']):
        obs,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],bp['training_seed_base']+1000003*cell['index']+1009*i)
        h.update(obs.tobytes()); _,exposure=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']))
        total+=exposure/rate
    with np.load(oldbank.OUT/'cells'/(cell['id']+'.npz'),allow_pickle=False) as f:
        assert h.hexdigest()==json.loads(str(f['metadata']))['observed_arrays_sha256']
    assert np.all(total>0)
    return dict(cell=cell,diagonal=(total/bp['records_per_cell']).tolist(),path_sha256=h.hexdigest())

def train():
    p=check(); OUT.mkdir(exist_ok=True); bp=oldbank.check(); start=time.perf_counter()
    expected=list(oldbank.cells(bp)); folder=OUT/'cells'; folder.mkdir(exist_ok=True)
    todo=[c for c in expected if not (folder/(c['id']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for i,r in enumerate(pool.map(train_cell,todo)):
            (folder/(r['cell']['id']+'.json')).write_text(json.dumps(r));
            if (i+1)%20==0: print('Safe bank cells',len(expected)-len(todo)+i+1,'/200',flush=True)
    matrices=np.zeros((2,5,5,4,64,64))
    for c in expected:
        r=json.loads((folder/(c['id']+'.json')).read_text()); assert r['cell']==c
        matrices[c['ri'],c['ai'],c['bi'],c['fi']]=np.diag(r['diagonal'])
    np.savez_compressed(OUT/'bank.npz',information=matrices,
      amplitudes=np.array([[g['a']*m for m in bp['grid_multipliers']] for g in bp['regimes']]),
      backgrounds=np.array([[g['b']*m for m in bp['grid_multipliers']] for g in bp['regimes']]),
      fractions=bp['fractional_phases'],regimes=[g['index'] for g in bp['regimes']],base_phase_integer=bp['base_phase_integer'])
    (OUT/'training.json').write_text(json.dumps(dict(status='PASS',cells=200,null_paths=40000,path_hash_checks=200,bank_sha256=sha(OUT/'bank.npz'),protocol_sha256=sha(P),seconds=time.perf_counter()-start),indent=2))

@lru_cache(1)
def bank(): return NullInformationBank.load(OUT/'bank.npz')

def statistic(s,e,fit,regime):
    theta=np.asarray(fit['theta']); rate,j=rates_and_design(theta,64,1.5)
    h,flags=bank().lookup(theta,regime); w=np.diag(h)
    u=s/rate-e; grid,b=templates(64,1.5)
    # Exact original-safe projection convention; only w is replaced.
    cross=b@(w[:,None]*j); inv=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12)
    raw=b@u-cross@inv@(j.T@u)
    v=(b*b)@w-np.einsum('mi,ij,mj->m',cross,inv,cross)
    good=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(v>1e-8)
    scores=np.where(good,np.maximum(raw,0)**2/np.maximum(v,1e-8),-np.inf)
    index=int(np.argmax(scores))
    return dict(score=float(scores[index]),primary=float(theta[1]),delay=float(grid[index])),flags

def worker(row):
    p=study.check(); case=row['case']; g=p['regimes'][case['regime']]; start=time.perf_counter()
    if g['td']>=g['te']:
        return dict(case=case,data_sha256=row['data_sha256'],sample=row['samples']['localized_safe_score_original'],mode='EXACT_NOHIDDEN_COPY')
    obs,_,_,_,_,digest=study.d.previous.old.base.data(case,p); assert digest==row['data_sha256']
    lag=outward_safe_lag(g['td'],g['te']); s,e=numeric_statistics(obs,g['cycles'],64,lag)
    fit=row['safe_null']; value,flags=statistic(s,e,fit,case['regime'])
    loc=localize(s,e,1.5,fit,value['delay']); boots=[]; bh=hashlib.sha256(); warnings=[]
    rate,_=rates_and_design(fit['theta'],64,1.5)
    for draw in range(199):
        bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw)
        bh.update(bo.tobytes()); bs,be=numeric_statistics(bo,g['cycles'],64,lag)
        bf=refit_from_start(bs,be,fit['theta'],1.5,'duration'); bv,_=statistic(bs,be,bf,case['regime']); boots.append(bv['score'])
        baseline=fine_scan(bs,be,1.5,fit=bf)
        assert abs(baseline['score']-row['samples']['safe_score_original']['bootstrap_scores'][draw])<1e-9
        if not bf['optimizer_success']: warnings.append(draw)
    assert bh.hexdigest()==row['bootstrap_path_sha256']['original']
    value.update(primary=loc['primary'],delay=loc['delay'],pvalue=pvalue(value['score'],boots),bootstrap_scores=boots)
    return dict(case=case,data_sha256=digest,sample=value,mode='INDEPENDENT_KNOWN_ACTIVE_BANK',bank_flags=flags,localization=loc,bootstrap_path_sha256=bh.hexdigest(),warning_draws=warnings,seconds=time.perf_counter()-start)

def run():
    p=check(); assert json.loads((OUT/'training.json').read_text())['bank_sha256']==sha(OUT/'bank.npz')
    assert all(os.environ.get(k)==v for k,v in p['threads'].items())
    start=time.perf_counter(); rows=study.read()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={r[0] for r in db.execute('SELECT id FROM records')}; todo=[r for r in rows if r['case']['id'] not in done]
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            for i,r in enumerate(pool.map(worker,todo)):
                db.execute('INSERT INTO records VALUES(?,?)',(r['case']['id'],json.dumps(r))); db.commit()
                if (len(done)+i+1)%32==0: print('Censored bank',len(done)+i+1,'/640',flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(status='COMPLETE',cases=640,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),seconds=time.perf_counter()-start),indent=2))
    analyze()

def analyze():
    check(); source=study.read()
    with sqlite3.connect(OUT/'records.sqlite3') as db: rows=[json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]
    assert len(rows)==640
    groups={}; totals={}
    for alpha in [.005,.01,.02]:
        t=dict(correct=0,wrong=0,false=0,positive=0,null=0); reg={str(i):dict(t) for i in range(4)}
        for a,b in zip(source,rows):
            assert a['case']==b['case'] and a['data_sha256']==b['data_sha256']
            z=b['sample']; assert z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200
            ep=study.d.previous.old.endpoint(dict(a,samples={'new':z}),'new',alpha)
            for k,v in ep.items(): t[k]+=v; reg[str(a['case']['regime'])][k]+=v
        totals[str(alpha)]=t; groups[str(alpha)]=reg
    result=dict(status='COMPLETE',totals=totals,by_regime=groups,records_sha256=sha(OUT/'records.sqlite3'),bank_sha256=sha(OUT/'bank.npz'))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': {'freeze':freeze,'train':train,'run':run,'analyze':analyze}[sys.argv[1]]()
