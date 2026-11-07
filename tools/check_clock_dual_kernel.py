"""Conditional periodic-clock kernel validation against independent event draws."""
from pathlib import Path
import sys,json,time,math,hashlib,platform
import numpy as np
import scipy,numba
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from deadtime_path_bootstrap import njit
from clock_dual_transition import transition_kernel,survival_after
from clock_pulse_likelihood import rates_and_design
P=ROOT/'CLOCK_DUAL_KERNEL_VALIDATION_PROTOCOL.json';OUT=ROOT/'results/clock_dual_kernel_validation_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


@njit(cache=True)
def sample_conditional_gaps(draws,hazard_prefix,phase,dd,de,count):
    k=len(hazard_prefix)-1;period=hazard_prefix[-1];out=np.empty(count,dtype=np.int64);draw_index=0
    for trial in range(count):
        # A recorded event is forced at0. Future input trials are independent.
        available=dd+1
        while True:
            absolute=phase+available;target=(absolute//k)*period+hazard_prefix[absolute%k]+draws[draw_index];draw_index+=1;cycle=int(np.floor(target/period));value=target-cycle*period;lo=0;hi=k
            while lo<hi:
                mid=(lo+hi)//2
                if hazard_prefix[mid+1]<=value:lo=mid+1
                else:hi=mid
            event=cycle*k+min(lo,k-1)-phase
            if event>=de+1:out[trial]=event;break
            available=event+dd+1
    return out


def freeze():
    if P.exists():raise FileExistsError('Frozen validation exists')
    sources=['src/clock_dual_transition.py','src/clock_dual_gradient.py','src/clock_pulse_likelihood.py','src/test_clock_dual_transition.py','src/test_clock_dual_gradient.py','src/test_clock_pulse_likelihood.py','src/homogeneous_dual_deadtime.py','src/deadtime_path_bootstrap.py','src/deadtime_exposure.py','src/deadtime_scan.py','src/safe_deadtime_exposure.py','tools/check_clock_dual_kernel.py']
    p=dict(status='FROZEN_BEFORE_NEW_PERIODIC_CLOCK_GAP_SAMPLES',sources={n:sha(ROOT/n) for n in sources},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),bins=64,sigma=1.5,phases=[0,13,31,47],samples_per_case_phase=100000,seed_base=6810942731,familywise_DKW_failure_probability=.001,cases=[dict(name='detector_dominant',theta=[.8,15.3,.3,.065,27.3],dd=83,de=39),dict(name='single_return_hidden',theta=[2.4,15.3,.8],dd=31,de=79),dict(name='two_returns_hidden',theta=[2.4,15.3,.8,.15,25.3],dd=31,de=79),dict(name='many_hidden_renewals',theta=[4.,15.3,.2,.3,28.3],dd=7,de=79)],criterion='16fixed known-parameter conditional gap distributions,each100000independent resets at given recorded phase. Compare empirical and modelCDF at BOTHsides of discrete jump locations(tieshandled). DKWunionbound=.001;epsilon=sqrt(log(2*16/.001)/(2*100000)). All64phase short-mass+tail normalization error<=1e-10.',scope='Exact finite-clock model validation,not a continuous-time detector/efficacy result. Independent inverse-hazard Bernoulli event simulation checks transient table summation with latent detector resets.12unit checks separately cover exhaustive small paths,full derivatives and limited homogeneous refinement to continuous time. Does not prove improved precision/recall,novelty,Q1 or submission readiness.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text())
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__);return p


def statistic(gaps,table,phase):
    x,counts=np.unique(gaps,return_counts=True);cum=np.cumsum(counts);n=len(gaps);right=cum/n;left=(cum-counts)/n;model_right=1-survival_after(table,phase,x);model_left=1-survival_after(table,phase,x-1);ks=max(float(np.max(np.abs(right-model_right))),float(np.max(np.abs(left-model_left))))
    return dict(empirical_CDF_max_error=ks,mean_gap=float(np.mean(gaps)),minimum_gap=int(np.min(gaps)),maximum_gap=int(np.max(gaps)))


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);n=p['samples_per_case_phase'];situations=len(p['cases'])*len(p['phases']);bound=float(np.sqrt(np.log(2*situations/p['familywise_DKW_failure_probability'])/(2*n)));rows=[];start=time.perf_counter();index=0
    for case in p['cases']:
        rates,_=rates_and_design(case['theta'],p['bins'],p['sigma']);probabilities=-np.expm1(-rates);table=transition_kernel(probabilities,case['dd'],case['de']);normalization=float(np.max(np.abs(table['short'].sum(axis=1)+table['live_at_tail']-1)));prefix=np.r_[0.,np.cumsum(-np.log1p(-probabilities))]
        for phase in p['phases']:
            name=f"{case['name']}_{phase}_gaps.npy";file=OUT/name
            if file.exists():raise FileExistsError('Samples exist;use replay')
            capacity=(n+1)*(math.ceil((case['de']+1)/(case['dd']+1))+1)+16;draws=np.random.default_rng(p['seed_base']+1009*index).exponential(size=capacity);gaps=sample_conditional_gaps(draws,prefix,phase,case['dd'],case['de'],n);np.save(file,gaps);s=statistic(gaps,table,phase);row=dict(case=case['name'],phase=phase,samples=n,file=name,sha256=sha(file),statistics=s,normalization_error=normalization,passes=s['empirical_CDF_max_error']<=bound and normalization<=1e-10);rows.append(row);index+=1;print(case['name'],phase,'CDFerror',s['empirical_CDF_max_error'],flush=True)
    result=dict(status='PASS' if all(x['passes'] for x in rows) else 'MODEL_CHECK_FAILED',protocol_sha256=sha(P),samples_total=situations*n,familywise_DKW_bound=bound,cases=rows,seconds=time.perf_counter()-start,scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)


def replay():
    p=check();a=json.loads((OUT/'analysis.json').read_text());assert a['protocol_sha256']==sha(P)
    for row in a['cases']:
        case=next(c for c in p['cases'] if c['name']==row['case']);rates,_=rates_and_design(case['theta'],p['bins'],p['sigma']);table=transition_kernel(-np.expm1(-rates),case['dd'],case['de']);file=OUT/row['file'];assert sha(file)==row['sha256'];gaps=np.load(file);assert len(gaps)==p['samples_per_case_phase'] and statistic(gaps,table,row['phase'])==row['statistics']
    result=dict(status='PASS',all16conditional_gap_arrays_checked=True,samples=a['samples_total'],all_CDF_statistics_recomputed=True,analysis_sha256=sha(OUT/'analysis.json'),scope=p['scope']);(OUT/'replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':{'freeze':freeze,'run':run,'replay':replay}[sys.argv[1]]()
