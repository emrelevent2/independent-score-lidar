"""Fit a random-recovery transition law on calibration; score disjoint measurements."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.optimize import minimize
from measurement_recovery_model import Pulse,GapLikelihood,raw_pairs
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'MEASURED_RECOVERY_PROTOCOL_20261107.json';OUT=ROOT/'results/measured_recovery_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES=['FGS_td048_2019_01_28_acq1.mat','FGS_td048_2019_01_28_acq2.mat']


def freeze():
    assert not P.exists()
    p=dict(status='FIXED_BEFORE_FITTED_MEASUREMENT_RESULTS',files=FILES,inputs={str((ROOT/'data/highflux_raw/2019_01_28'/n).relative_to(ROOT)):sha(ROOT/'data/highflux_raw/2019_01_28'/n) for n in FILES+['laser_calib_100.mat']},sources={n:sha(ROOT/'tools'/n) for n in ['measurement_recovery_model.py','measurement_recovery_study.py']},
      calibration_pair_range=[100,20100],evaluation_pair_range=[20200,40200],bins=128,detector_recovery_ns=50.,integration_step_ns=.125,refinement_step_ns=.0625,
      starts=[[78.,1.],[82.,3.]],mu_bounds_ns=[60.,95.],sd_bounds_ns=[.15,8.],
      fitting='Conditional likelihood of all consecutive gaps. Independent supplied measured pulse shape; a, delay and B fit jointly with recovery mean-location and scale. Two fixed starts; select by calibration likelihood only. Recovery normal truncated to [Td,2Td], independent at each recorded avalanche. Td=holdoff+2ns follows source convention and remains fixed.',
      comparator='Fixed recovery set to calibration minimum gap minus one raw time-resolution unit. Rate nuisance refitted on calibration only; no test-data parameter tuning. Nominal 80ns zero-density coverage also reported. No gap removal, timestamp clipping or invented second surface.',
      endpoints='All-file model coverage; predictive log density per heldout transition; all starts and mesh refinement. Real records have one surface and no independent two-surface labels. No measured secondary-detection accuracy claim.',
      scope='Exploratory measured-data model extension. This study does not retroactively change frozen simulation bank or its 199-draw results.')
    P.write_text(json.dumps(p,indent=2));print('Frozen measured model',sha(P),flush=True)


def fit(lik,pulse,starts,p,hard_te=None):
    counts=np.bincount(np.floor(lik.phase/pulse.width).astype(int),minlength=pulse.bins)
    centre=(np.argmax(counts)+.5)*pulse.width
    duration=np.mean(lik.gaps)
    flux=max(.01,pulse.period/max(duration-lik.td,1.))
    candidates=[]
    for mu,sd in starts:
        x=np.array([np.log(flux*.8),centre,np.log(flux*.2),mu,np.log(sd)])
        bounds=[(np.log(1e-6),np.log(20)),(centre-5,centre+5),(np.log(1e-6),np.log(30)),tuple(p['mu_bounds_ns']),tuple(np.log(p['sd_bounds_ns']))]
        if hard_te is not None:x=x[:3];bounds=bounds[:3]
        def objective(z):
            rate=pulse.rate(z)
            values=lik.logdensity(rate,z[3] if hard_te is None else hard_te,np.exp(z[4]) if hard_te is None else 1.,hard=hard_te is not None)
            return -float(np.mean(values))
        start=time.perf_counter()
        result=minimize(objective,x,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=180,ftol=2e-11,gtol=1e-6,maxls=30,finite_diff_rel_step=1e-5))
        candidates.append(dict(x=result.x.tolist(),negative_mean_logdensity=float(result.fun),success=bool(result.success),message=str(result.message),evaluations=int(result.nfev),seconds=time.perf_counter()-start))
        print('Fit',mu,sd,candidates[-1],flush=True)
    return min(candidates,key=lambda z:z['negative_mean_logdensity']),candidates


def run():
    p=json.loads(P.read_text())
    for n,d in p['sources'].items():assert sha(ROOT/'tools'/n)==d
    for n,d in p['inputs'].items():assert sha(ROOT/n)==d
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'analysis.json').exists()
    output=[]
    for name in p['files']:
        phase,gaps,period,meta=raw_pairs(ROOT/'data/highflux_raw/2019_01_28'/name)
        assert meta['holdoff_ns']==48 and len(gaps)>p['evaluation_pair_range'][1]
        a,b=p['calibration_pair_range'];c,e=p['evaluation_pair_range'];d=p['detector_recovery_ns']
        pulse=Pulse(ROOT/'data/highflux_raw/2019_01_28/laser_calib_100.mat',period,p['bins'])
        train=GapLikelihood(phase[a:b],gaps[a:b],period,d,p['integration_step_ns'])
        best,starts=fit(train,pulse,p['starts'],p)
        hard_te=float(gaps[a:b].min()-meta['raw_resolution_ns'])
        hard,hard_starts=fit(train,pulse,[p['starts'][0]],p,hard_te)
        x=np.array(best['x']);hx=np.array(hard['x']);rate=pulse.rate(x);hrate=pulse.rate(hx)
        del train
        test=GapLikelihood(phase[c:e],gaps[c:e],period,d,p['integration_step_ns'])
        ll=test.logdensity(rate,x[3],np.exp(x[4]));hl=test.logdensity(hrate,hard_te,1.,hard=True)
        refined=GapLikelihood(phase[c:e],gaps[c:e],period,d,p['refinement_step_ns']).logdensity(rate,x[3],np.exp(x[4]))
        row=dict(file=name,metadata=meta,period_ns=period,calibration_n=b-a,evaluation_n=e-c,calibration_pair_range=[a,b],evaluation_pair_range=[c,e],best=best,starts=starts,hard_fit=hard,hard_starts=hard_starts,
          parameters=dict(amplitude=float(np.exp(x[0])),delay_ns=float(x[1]%period),background=float(np.exp(x[2])),recovery_location_ns=float(x[3]),recovery_scale_ns=float(np.exp(x[4])),detector_ns=d,hard_recovery_ns=hard_te),
          nominal80_zero_density_all=int(np.count_nonzero(gaps<80)),hard_zero_density_all=int(np.count_nonzero(gaps<max(d,hard_te))),random_zero_density_all=int(np.count_nonzero(gaps<=d)),minimum_gap_all_ns=float(gaps.min()),
          evaluation=dict(random_finite=int(np.isfinite(ll).sum()),hard_finite=int(np.isfinite(hl).sum()),random_mean_logdensity=float(ll.mean()),hard_mean_logdensity=float(hl.mean()) if np.all(np.isfinite(hl)) else None,random_minus_hard_mean=float(np.mean(ll-hl)) if np.all(np.isfinite(hl)) else None,refinement_max_abs_logdensity=float(np.max(np.abs(ll-refined))),refinement_mean_abs_logdensity=float(np.mean(np.abs(ll-refined)))))
        np.savez_compressed(OUT/(Path(name).stem+'_heldout.npz'),phase=phase[c:e],gaps=gaps[c:e],logdensity=ll,hard_logdensity=hl,refined_logdensity=refined,rate=rate,hard_rate=hrate)
        output.append(row);print('Measurement complete',json.dumps(row),flush=True)
    result=dict(status='COMPLETE',protocol_sha256=sha(P),rows=output)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print('All measured fits complete',flush=True)


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
