"""Frozen continuous observation-model check on original four recovery regimes."""
from pathlib import Path
import sys,json,time,math,hashlib,platform
import numpy as np
import scipy,numba
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import ct_dual_fast as ct
from deadtime_path_bootstrap import njit
from clock_pulse_likelihood import rates_and_design
import check_clock_dual_kernel as clock_reference
import check_homogeneous_dual_kernel as homogeneous_reference
P=ROOT/'CT_DUAL_KERNEL_VALIDATION_PROTOCOL.json';OUT=ROOT/'results/ct_dual_kernel_validation_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


@njit(cache=True)
def sequential_conditional_gaps(draws,rates,prefix,phase,td,te,count):
    k=len(rates);period=prefix[-1];index=0;out=np.empty(count)
    for trial in range(count):
        available=td
        while True:
            absolute=phase+available;cycle=math.floor(absolute/k);v=absolute-cycle*k;j=min(int(math.floor(v)),k-1);target=cycle*period+prefix[j]+rates[j]*(v-j)+draws[index];index+=1;cc=math.floor(target/period);value=target-cc*period;lo=0;hi=k
            while lo<hi:
                mid=(lo+hi)//2
                if prefix[mid+1]<=value:lo=mid+1
                else:hi=mid
            j=min(lo,k-1);event=cc*k+j+(value-prefix[j])/rates[j]-phase
            if event>=te:out[trial]=event;break
            available=event+td
    return out


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    a=clock_reference.check();homogeneous_reference.check();names=['src/ct_dual_reference.py','src/ct_dual_outer_reference.py','src/ct_dual_fast.py','src/test_ct_dual_reference.py','src/test_ct_dual_fast.py','tools/check_ct_dual_kernel.py'];p=dict(status='FROZEN_BEFORE_NEW_CONTINUOUS_GAP_SAMPLES',sources={name:sha(ROOT/name) for name in names},clock_reference_protocol_sha256=sha(clock_reference.P),homogeneous_reference_protocol_sha256=sha(homogeneous_reference.P),environment=a['environment'],bins=64,sigma=1.5,phases=[0.,.37,15.31,42.1],samples_per_case_phase=100000,seed_base=8917402639,familywise_DKW_failure_probability=.001,cases=[dict(name='moderate',theta=[.8,15.3,.3,.065,27.3],td=83.7,te=40.),dict(name='near',theta=[.8,15.3,1.,.09,21.3],td=126.7,te=80.),dict(name='bright_hidden',theta=[2.4,15.3,.8,.15,25.3],td=31.7,te=80.),dict(name='weak_hidden',theta=[.25,15.3,.1,.035,35.3],td=83.7,te=100.)],criterion='16known-parameter conditionalCDFchecks,100000independentcontinuousinverse-hazardphoton paths each. DKWunionbound.001 and normalization atfirstobservablegap within1e-8. OriginalperiodicpiecewiseconstantGaussianbin-rate model,originalfractionalrecoveryvalues. No parameter/depthfitting or efficacy endpoint.',scope='Continuous-time full-observation model validation covering ALLoriginal4recoveryregimes and up to2hiddenavalanches. Not a secondary-return detection/improvement result,not finiteunknown-null guarantee or novelty/Q1 evidence.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());clock_reference.check();homogeneous_reference.check()
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
    assert sha(clock_reference.P)==p['clock_reference_protocol_sha256'] and sha(homogeneous_reference.P)==p['homogeneous_reference_protocol_sha256'];return p


def statistic(gaps,profile):
    n=len(gaps);x=np.sort(gaps);logsf,_=ct.conditional_from_profile(profile,x,True);cdf=-np.expm1(logsf);assert np.all(cdf>=-1e-9) and np.all(cdf<=1+1e-9);ks=max(float(np.max(np.arange(1,n+1)/n-cdf)),float(np.max(cdf-np.arange(n)/n)));minimum=max(profile['geometry']['td'],profile['geometry']['te']);v,_=ct.conditional_from_profile(profile,[minimum],True)
    return dict(empirical_CDF_max_error=ks,survival_at_first_possible_gap=float(np.exp(v[0])),sample_mean=float(np.mean(gaps)),sample_minimum=float(np.min(gaps)),sample_maximum=float(np.max(gaps)))


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);n=p['samples_per_case_phase'];bound=float(np.sqrt(np.log(2*len(p['cases'])*len(p['phases'])/p['familywise_DKW_failure_probability'])/(2*n)));rows=[];start=time.perf_counter();index=0
    for case in p['cases']:
        rates,_=rates_and_design(case['theta'],64,1.5);prefix=np.r_[0.,np.cumsum(rates)]
        for phase in p['phases']:
            file=OUT/f"{case['name']}_{index}_gaps.npy"
            if file.exists():raise FileExistsError('Saved gaps exist;use replay')
            draws=np.random.default_rng(p['seed_base']+1009*index).exponential(size=(n+1)*(math.ceil(case['te']/case['td'])+1)+16);gaps=sequential_conditional_gaps(draws,rates,prefix,phase,case['td'],case['te'],n);np.save(file,gaps);profile=ct.cached_profile(rates,np.zeros((64,0)),phase,case['td'],case['te']);s=statistic(gaps,profile);rows.append(dict(case=case['name'],phase=phase,file=file.name,sha256=sha(file),statistics=s,passes=s['empirical_CDF_max_error']<=bound and abs(s['survival_at_first_possible_gap']-1)<1e-8));index+=1;print(case['name'],phase,json.dumps(s),flush=True)
    r=dict(status='PASS' if all(row['passes'] for row in rows) else 'MODEL_CHECK_FAILED',protocol_sha256=sha(P),samples_total=n*len(rows),familywise_DKW_bound=bound,cases=rows,seconds=time.perf_counter()-start,scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2),flush=True)


def replay():
    p=check();a=json.loads((OUT/'analysis.json').read_text());assert len(a['cases'])==16;assert {(x['case'],x['phase']) for x in a['cases']}=={(c['name'],phase) for c in p['cases'] for phase in p['phases']};assert a['protocol_sha256']==sha(P)
    for row in a['cases']:
        case=next(c for c in p['cases'] if c['name']==row['case']);rates,_=rates_and_design(case['theta'],64,1.5);profile=ct.cached_profile(rates,np.zeros((64,0)),row['phase'],case['td'],case['te']);file=OUT/row['file'];assert sha(file)==row['sha256'];gaps=np.load(file);assert len(gaps)==p['samples_per_case_phase'] and statistic(gaps,profile)==row['statistics']
    r=dict(status='PASS',all16continuous_case_keys_and_data_arrays_verified=True,all_CDF_statistics_recomputed=True,samples=a['samples_total'],analysis_sha256=sha(OUT/'analysis.json'),scope=p['scope']);(OUT/'replay.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))


if __name__=='__main__':{'freeze':freeze,'run':run,'replay':replay}[sys.argv[1]]()
