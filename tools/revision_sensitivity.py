"""Prospectively specified bright-hidden power and model-mismatch simulations.

Original nominal method is unchanged. Bootstrap samples are generated from
the fitted nominal null; perturbations affect acquisitions only. Hard-support
violations are reported as model incompatibilities, never removed or repaired.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,hashlib,sqlite3,time,os,heapq
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import score_peak_localization_development as study
import null_bank_development as d
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from clock_pulse_likelihood import rates_and_design
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import fine_scan,refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from null_bank_score import primary_statistic
from ct_full_score import fit_full_null
from score_peak_localization import localize
import ct_dual_fast as kernel
P=ROOT/'REVISION_SENSITIVITY_PROTOCOL_20261107.json'
OUT=ROOT/'results/revision_sensitivity_v1'
sha=d.sha

def freeze():
    if P.exists():raise FileExistsError(P)
    study.check()
    p=dict(status='FROZEN_BEFORE_NEW_POWER_AND_MISMATCH_RESULTS',source_sha256=sha(__file__),original_protocol_sha256=sha(study.P),
      regime=dict(index=2,a=2.4,b=.8,cycles=1000,td=31.7,te=80.,sigma=1.5,bins=64),
      power=dict(ratios=[.03125,.0625,.125],separations=[6.,10.,16.,24.],positives_per_cell=32,nulls=96),
      robustness=dict(positives_per_condition=32,nulls_per_condition=64,ratio=.0625,separation=10.,
        conditions=[dict(name='nominal',td_scale=1.,te_scale=1.,jitter=0.,afterpulse=0.),
         dict(name='detector_minus5',td_scale=.95,te_scale=1.,jitter=0.,afterpulse=0.),
         dict(name='detector_plus5',td_scale=1.05,te_scale=1.,jitter=0.,afterpulse=0.),
         dict(name='electronics_minus2',td_scale=1.,te_scale=.98,jitter=0.,afterpulse=0.),
         dict(name='electronics_plus2',td_scale=1.,te_scale=1.02,jitter=0.,afterpulse=0.),
         dict(name='jitter_0p1',td_scale=1.,te_scale=1.,jitter=.1,afterpulse=0.),
         dict(name='jitter_0p5',td_scale=1.,te_scale=1.,jitter=.5,afterpulse=0.),
         dict(name='afterpulse_1pct',td_scale=1.,te_scale=1.,jitter=0.,afterpulse=.01),
         dict(name='afterpulse_5pct',td_scale=1.,te_scale=1.,jitter=0.,afterpulse=.05)]),
      design='New independent query seeds. Primary phase uniform [14.15,16.45], amplitude .7--1.3 times nominal, background .75--1.25, secondary ratio fixed per cell and separation with uniform +/-.35 jitter. No donors needed for compared local methods. Null power samples shared across all amplitude/separation cells.',
      methods='Unchanged localized full-bank primary and matched localized known-active score. Both use common three-start nominal full-null bootstrap generator. The censored original-generator result is separate in original 640 comparison. Every draw repeats pilot, full-null refit, independent-bank lookup and quarter-grid score; observed-only own likelihood localizers.',
      mismatch='True detector/electronics recovery perturbed; inference retains nominal values. Timing jitter adds independent Gaussian timestamp noise, followed by chronological readout order. Invalid initial/horizon/support observations recorded as incompatible, full method makes no decision. Afterpulses: each avalanche independently schedules one at td+exponential(mean=5 bins), probability .01 or .05; candidate suppressed if detector dead; accepted hidden/recorded avalanches restart detector and can generate descendants. Timing electronics filters all avalanches normally. Bootstrap retains nominal no-jitter/no-afterpulse model.',
      reporting='All cells and conditions, integer numerators/denominators, Wilson intervals, correct/wrong/null separately, and incompatible records. Conditional valid-record rates separately from coverage; incompatible records are not counted as successful null rejections or misses. No threshold tuning or selective scenario dropping.',
      seed_base=75019862401,bootstrap_seed_base=950000000000,bootstrap_replicates=199,workers=4,
      threads={k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)

def check():
    p=json.loads(P.read_text()); assert p['source_sha256']==sha(__file__); study.check(); return p

def cases(p,kind):
    index=0
    if kind=='power':
        for ratio in p['power']['ratios']:
            for sep in p['power']['separations']:
                for j in range(p['power']['positives_per_cell']):
                    yield dict(id=f'power_{index:04}',index=index,positive=True,ratio=ratio,separation=sep,condition=dict(name='nominal',td_scale=1.,te_scale=1.,jitter=0.,afterpulse=0.));index+=1
        for j in range(p['power']['nulls']):
            yield dict(id=f'power_{index:04}',index=index,positive=False,ratio=0.,separation=10.,condition=dict(name='nominal',td_scale=1.,te_scale=1.,jitter=0.,afterpulse=0.));index+=1
    else:
        cfg=p['robustness']
        for cond in cfg['conditions']:
            for pos,num in [(True,cfg['positives_per_condition']),(False,cfg['nulls_per_condition'])]:
                for j in range(num):
                    yield dict(id=f'robust_{index:04}',index=index+10000,positive=pos,ratio=cfg['ratio'] if pos else 0.,separation=cfg['separation'],condition=cond);index+=1

def with_afterpulses(rate,cycles,td,te,seed,prob):
    rng=np.random.default_rng(seed); prefix=np.r_[0.,np.cumsum(rate)]; mass=prefix[-1]; horizon=cycles*64.
    available=0.; next_electronics=0.; scheduled=[]; obs=[]; avalanche=[]
    while available<horizon:
        cycle=np.floor(available/64); phase=available-cycle*64; j=min(int(phase),63)
        value=cycle*mass+prefix[j]+rate[j]*(phase-j)+rng.exponential()
        c=np.floor(value/mass); v=value-c*mass; k=min(int(np.searchsorted(prefix,v,side='right')-1),63)
        incident=c*64+k+(v-prefix[k])/rate[k]
        while scheduled and scheduled[0]<available:heapq.heappop(scheduled)
        if scheduled and scheduled[0]<incident: t=heapq.heappop(scheduled)
        else:t=incident
        if t>=horizon:break
        avalanche.append(t)
        if t>=next_electronics:obs.append(t);next_electronics=t+te
        if rng.uniform()<prob:heapq.heappush(scheduled,t+td+rng.exponential(5.))
        available=t+td
    return np.array(obs),np.array(avalanche)

def acquisition(case,p):
    g=p['regime'];rng=np.random.default_rng(p['seed_base']+1009*case['index'])
    a=g['a']*rng.uniform(.7,1.3);tau=rng.uniform(14.15,16.45);b=g['b']*rng.uniform(.75,1.25)
    delta=(tau+case['separation']+rng.uniform(-.35,.35))%64
    rate=a*pulse(tau,64,1.5)[0]+b/64+case['ratio']*a*pulse(delta,64,1.5)[0]
    q=case['condition'];seed=p['seed_base']+1000003*(case['index']+1)
    if q['afterpulse']:
        obs,latent=with_afterpulses(rate,g['cycles'],g['td']*q['td_scale'],g['te']*q['te_scale'],seed,q['afterpulse'])
    else:obs,latent=simulate_two_dead_times(rate,g['cycles'],g['td']*q['td_scale'],g['te']*q['te_scale'],seed)
    if q['jitter']:obs=np.sort(obs+np.random.default_rng(seed+777).normal(0,q['jitter'],len(obs)))
    return obs,dict(primary=tau,secondary=delta),[a,tau,b,case['ratio']*a,delta]

def worker(case):
    p=check();g=p['regime'];start=time.perf_counter();obs,truth,theta=acquisition(case,p)
    result=dict(case=case,truth=truth,generating_parameters=theta,observed_events=len(obs),data_sha256=hashlib.sha256(obs.tobytes()).hexdigest())
    incompatible=bool(np.any(np.diff(obs)<max(g['td'],g['te'])) or (len(obs) and (obs[0]<0 or obs[-1]>=g['cycles']*64)))
    if incompatible:return dict(result,status='HARD_SUPPORT_INCOMPATIBLE',seconds=time.perf_counter()-start)
    lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);sf=fit_null(s,e,1.5)
    geo=kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te'])
    fv,fn,flags=primary_statistic(s,e,1.5,sf,geo,d.bank(),2);sv=fine_scan(s,e,1.5,fit=sf)
    calibration=fit_full_null(s,e,1.5,sf,geo);rate,_=rates_and_design(calibration['theta'],64,1.5)
    fl=localize(s,e,1.5,fn,fv['delay'],geo);sl=localize(s,e,1.5,sf,sv['delay'])
    boots={'full':[],'safe':[]};warnings=[];h=hashlib.sha256()
    for j in range(199):
        bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],p['bootstrap_seed_base']+1000003*case['index']+1009*j);h.update(bo.tobytes())
        bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,calibration['theta'],1.5,'duration')
        bg=kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);bv,bfn,_=primary_statistic(bs,be,1.5,bf,bg,d.bank(),2)
        boots['full'].append(bv['score']);boots['safe'].append(fine_scan(bs,be,1.5,fit=bf)['score'])
        if not bf['optimizer_success'] or not bfn['optimizer_success']:warnings.append(j)
    samples={}
    for m,v,loc in [('full',fv,fl),('safe',sv,sl)]:
        samples[m]=dict(score=v['score'],primary=loc['primary'],delay=loc['delay'],pvalue=pvalue(v['score'],boots[m]),bootstrap_scores=boots[m])
    return dict(result,status='EVALUATED',samples=samples,bank_flags=flags,localizers={'full':fl,'safe':sl},nulls={'safe':d.json_fit(sf),'full':d.json_fit(fn),'calibration':d.json_fit(calibration)},bootstrap_path_sha256=h.hexdigest(),warning_draws=warnings,seconds=time.perf_counter()-start)

def run(kind):
    p=check(); OUT.mkdir(exist_ok=True); assert all(os.environ.get(k)==v for k,v in p['threads'].items())
    start=time.perf_counter();expected=list(cases(p,kind));path=OUT/(kind+'.sqlite3')
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={v[0] for v in db.execute('SELECT id FROM records')};todo=[c for c in expected if c['id'] not in done]
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            for i,r in enumerate(pool.map(worker,todo)):
                db.execute('INSERT INTO records VALUES(?,?)',(r['case']['id'],json.dumps(r)));db.commit()
                if (len(done)+i+1)%16==0:print(kind,len(done)+i+1,'/',len(expected),'seconds',round(time.perf_counter()-start,1),flush=True)
    (OUT/(kind+'_completion.json')).write_text(json.dumps(dict(status='COMPLETE',cases=len(expected),records_sha256=sha(path),protocol_sha256=sha(P),seconds=time.perf_counter()-start),indent=2))

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run(sys.argv[1])
