"""Frozen physical-kernel validation,not a lidar efficacy experiment."""
from pathlib import Path
import sys,json,time,math,hashlib,platform
import numpy as np
import scipy,numba
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from homogeneous_dual_deadtime import log_gap_density,gap_survival
from deadtime_path_bootstrap import njit
P=ROOT/'HOMOGENEOUS_DUAL_KERNEL_VALIDATION_PROTOCOL.json';OUT=ROOT/'results/homogeneous_dual_kernel_validation_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


@njit(cache=True)
def sequential_gaps(draws,rate,td,te,count):
    # Sequential two-component physical mechanism;no CDF/kernel evaluation.
    available=0.;electronics=0.;last=0.;have_first=False;n=0;out=np.empty(count)
    for draw in draws:
        event=available+draw/rate;available=event+td
        if event<electronics:continue
        electronics=event+te
        if have_first:
            out[n]=event-last;n+=1
            if n==count:return out
        else:have_first=True
        last=event
    raise ValueError('Insufficient sequential draws')


def freeze():
    if P.exists():raise FileExistsError('Frozen validation protocol exists')
    sources=['src/homogeneous_dual_deadtime.py','src/test_homogeneous_dual_deadtime.py','src/deadtime_path_bootstrap.py','src/deadtime_exposure.py','src/deadtime_scan.py','tools/check_homogeneous_dual_kernel.py']
    p=dict(status='FROZEN_BEFORE_NEW_HOMOGENEOUS_GAP_SAMPLES',sources={n:sha(ROOT/n) for n in sources},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),samples_per_regime=100000,seed_base=4617927001,familywise_DKW_failure_probability=.001,cases=[dict(name='detector_dominant',td=83.7,te=40.,rate=.02),dict(name='bright_hidden',td=31.7,te=80.,rate=.05),dict(name='many_hidden',td=7.,te=50.,rate=.3),dict(name='weak_hidden',td=83.7,te=100.,rate=.005),dict(name='three_recovery_window',td=10.,te=35.,rate=.2)],criterion='For5fixed known-rate regenerative i.i.d.gap distributions,each empiricalCDF must bewithin sqrt(log(2*5/.001)/(2*100000)) of its analyticalCDF. Union DKW bound=.001 under the specified homogeneous law. Numerical normalization integral+exact exponential tail must bewithin1e-9 of1. No parameter fitting or outcome-based selection.',scope='Homogeneous renewal-overshoot model check including multiple unrecorded avalanches. Standard renewal reasoning,not a priority claim. NOT a periodic pulse model,depth estimator,improved detector,Q1 result,or efficacy confirmation.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text())
    for name,value in p['sources'].items():assert sha(ROOT/name)==value,name
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__)
    return p


def statistics(gaps,c):
    n=len(gaps);td,te,rate=[c[k] for k in ['td','te','rate']];x=np.sort(gaps);cdf=1-gap_survival(x,rate,td,te);ks=max(float(np.max(np.arange(1,n+1)/n-cdf)),float(np.max(cdf-np.arange(n)/n)));end=te+td;breaks=sorted(set([te,end]+[i*td for i in range(1,int(end/td)+1) if te<i*td<end]));mass=sum(quad(lambda t:np.exp(log_gap_density(t,rate,td,te)),a,b,epsabs=1e-11,epsrel=1e-11)[0] for a,b in zip(breaks[:-1],breaks[1:]));mass+=np.exp(log_gap_density(end,rate,td,te))/rate
    return dict(empirical_CDF_max_error=ks,density_total_mass=float(mass),gap_mean=float(np.mean(gaps)),minimum_gap=float(np.min(gaps)),scored_gap_fraction=float(np.mean(gaps>=(td if td>=te else td+te))))


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);n=p['samples_per_regime'];bound=float(np.sqrt(np.log(2*len(p['cases'])/p['familywise_DKW_failure_probability'])/(2*n)));rows=[];start=time.perf_counter()
    for i,c in enumerate(p['cases']):
        file=OUT/f"{c['name']}_gaps.npy"
        if file.exists():raise FileExistsError('Validation samples exist;use replay')
        draws=np.random.default_rng(p['seed_base']+1009*i).exponential(size=(n+1)*(math.ceil(c['te']/c['td'])+1)+16);gaps=sequential_gaps(draws,c['rate'],c['td'],c['te'],n);np.save(file,gaps);result=statistics(gaps,c);row=dict(case=c,samples=n,data_file=file.name,data_sha256=sha(file),statistics=result,passes_CDF_bound=result['empirical_CDF_max_error']<=bound,passes_normalization=abs(result['density_total_mass']-1)<1e-9);rows.append(row);print(c['name'],json.dumps(result),flush=True)
    result=dict(status='PASS' if all(x['passes_CDF_bound'] and x['passes_normalization'] for x in rows) else 'MODEL_CHECK_FAILED',protocol_sha256=sha(P),samples_total=n*len(rows),familywise_DKW_bound=bound,cases=rows,seconds=time.perf_counter()-start,scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)


def replay():
    p=check();saved=json.loads((OUT/'analysis.json').read_text());assert saved['protocol_sha256']==sha(P)
    for row,c in zip(saved['cases'],p['cases']):
        file=OUT/row['data_file'];assert row['case']==c and sha(file)==row['data_sha256'];gaps=np.load(file);assert len(gaps)==p['samples_per_regime'];assert statistics(gaps,c)==row['statistics']
    result=dict(status='PASS',all_saved_gap_arrays_verified=True,samples=saved['samples_total'],all_CDF_and_normalization_statistics_recomputed=True,analysis_sha256=sha(OUT/'analysis.json'),scope=p['scope']);(OUT/'replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':{'freeze':freeze,'run':run,'replay':replay}[sys.argv[1]]()
