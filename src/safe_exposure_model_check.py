"""Point-null model check under two dead times,not second-surface efficacy."""
from pathlib import Path
import sys,json,time,hashlib,platform
import numpy as np
import numba
from deadtime_exposure import sufficient_statistics
from safe_deadtime_exposure import safe_lag,safe_statistics,simulate_two_dead_times
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'SAFE_EXPOSURE_MODEL_CHECK_PROTOCOL.json';OUT=ROOT/'results/safe_exposure_model_check_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Protocol frozen')
    sources=['safe_exposure_model_check.py','safe_deadtime_exposure.py','test_safe_deadtime_exposure.py','deadtime_path_bootstrap.py','deadtime_exposure.py','deadtime_scan.py']
    p=dict(status='FROZEN_BEFORE_TWO_DEAD_TIME_POINT_MODEL_CHECK',sources={n:sha(ROOT/'src'/n) for n in sources},environment=dict(python=platform.python_version(),numpy=np.__version__,numba=numba.__version__),settings=dict(bins=16,cycles=200,replicates_per_regime=2048,seed_base=81764019,alternative_ratio=.9),
           regimes=[dict(name='no_hidden',td=19.3,te=8.,rate=.04),dict(name='moderate_hidden',td=50.3,te=80.7,rate=.04),dict(name='intermediate_hidden',td=23.7,te=61.2,rate=.04),dict(name='many_hidden',td=5.3,te=61.2,rate=.15)],
           scope='Implementation/point-null martingale model check ONLY. Homogeneous known incident rates;alternative0.9xtrue fixed before simulation,not a fitted secondary-surface detector. No composite-null,physical-data or novelty/efficacy claim. Safe lag known from Optica2021Tmin;hidden avalanches simulated explicitly. Naive max(td,te) comparison deliberately misspecified when hidden avalanches matter;not a strong detection baseline.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text())
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,numba=numba.__version__)
    return p


def case_data(gi,j,p):
    g=p['regimes'][gi];c=p['settings'];k=c['bins'];obs,latent=simulate_two_dead_times(np.full(k,g['rate']),c['cycles'],g['td'],g['te'],c['seed_base']+1000003*gi+1009*j)
    s,e,info=safe_statistics(obs,c['cycles'],k,safe_lag(g['td'],g['te']));ns,ne=sufficient_statistics(obs,c['cycles'],k,max(g['td'],g['te']))
    # Independent truth check: no safe opening may overlap a hidden avalanche's
    # dead interval. No hidden avalanches can occur inside a safe opening.
    for lo,hi in zip(info['open_starts'],info['open_ends']):
        t=np.searchsorted(latent,lo,side='right')-1
        if t>=0:assert lo>=latent[t]+g['td']-1e-8
        assert np.searchsorted(latent,hi-1e-8)==np.searchsorted(latent,lo+1e-8)
    h=hashlib.sha256()
    for v in [obs,latent,s,e,ns,ne]:h.update(v.tobytes())
    q=g['rate']*c['alternative_ratio'];r=dict(regime=gi,index=j,observed=len(obs),latent=len(latent),safe_count=int(s.sum()),safe_time=float(e.sum()),naive_count=int(ns.sum()),naive_time=float(ne.sum()),safe_log_ratio=float(s.sum()*np.log(q/g['rate'])-(q-g['rate'])*e.sum()),naive_log_ratio=float(ns.sum()*np.log(q/g['rate'])-(q-g['rate'])*ne.sum()),path_sha256=h.hexdigest())
    return r


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'completion.json').exists();start=time.perf_counter();rows=[]
    for gi,g in enumerate(p['regimes']):
        for j in range(p['settings']['replicates_per_regime']):rows.append(case_data(gi,j,p))
        print('Verified safe state for',g['name'],p['settings']['replicates_per_regime'],'paths',flush=True)
    (OUT/'records.json').write_text(json.dumps(rows,separators=(',',':')))
    (OUT/'completion.json').write_text(json.dumps(dict(status='COMPLETE',protocol_sha256=sha(P),records_sha256=sha(OUT/'records.json'),cases=len(rows),seconds=time.perf_counter()-start),indent=2));analyze()


def analyze():
    p=check();rows=json.loads((OUT/'records.json').read_text());groups=[]
    for gi,g in enumerate(p['regimes']):
        a=[v for v in rows if v['regime']==gi];entry=dict(regime=g['name'],true_rate=g['rate'],paths=len(a),mean_hidden_avalanches=float(np.mean([v['latent']-v['observed'] for v in a])),methods={})
        for name in ['safe','naive']:
            values=np.exp([v[f'{name}_log_ratio'] for v in a]);rate=sum(v[f'{name}_count'] for v in a)/sum(v[f'{name}_time'] for v in a)
            entry['methods'][name]=dict(pooled_rate_estimate=rate,relative_bias=rate/g['rate']-1,mean_point_likelihood_ratio=float(np.mean(values)),ratio_mean_monte_carlo_se=float(np.std(values,ddof=1)/np.sqrt(len(values))))
        groups.append(entry)
    result=dict(status='POINT_MODEL_DIAGNOSTIC_ONLY',per_regime=groups,scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
