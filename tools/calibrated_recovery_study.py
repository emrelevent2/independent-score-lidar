"""M2/M3 extension: calibration, matched noisy-path testing, measured nulls.

This is a separate cohort-calibrated extension. The original 199-draw,
query-specific results remain unchanged. Previously evaluated mismatch
records are reused openly, with independent sensor-calibration records.
"""
from pathlib import Path
import os, sys, json, hashlib, sqlite3, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.stats import beta
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import calibrated_recovery as model
import revision_sensitivity as previous
from safe_deadtime_exposure import safe_statistics
from null_information_bank import NullInformationBank

OUT = ROOT/'results/calibrated_recovery_v1'
P = ROOT/'CALIBRATED_RECOVERY_PROTOCOL_20261107.json'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setting(name):
    if name == 'measured_high':
        original = json.loads((ROOT/'results/measured_recovery_v1/analysis.json').read_text())['rows'][1]
        par = original['parameters']; unit = original['period_ns']/64
        return dict(name=name, td=50/unit, nominal_te=80/unit,
            mu=par['recovery_location_ns']/unit, sd=par['recovery_scale_ns']/unit,
            jitter=0., a=par['amplitude'], b=par['background'], cycles=1000,
            unit_ns=unit, measured_template=True, acquisition='measured_random_recovery')
    protocol = json.loads(previous.P.read_text())
    condition = next(c for c in protocol['robustness']['conditions'] if c['name']==name)
    return dict(name=name, td=31.7, nominal_te=80., mu=80., sd=.1,
        jitter=condition['jitter'], a=2.4, b=.8, cycles=1000,
        unit_ns=1., measured_template=False, condition=condition,
        acquisition='original_mismatch_records')


def template(cfg):
    if not cfg['measured_template']:
        return model.Template()
    raw = loadmat(ROOT/'data/highflux_raw/2019_01_28/laser_calib_100.mat')
    return model.Template(mass=raw['smoothPulseShape'].ravel(),
                          raw_step=float(raw['binRes'].item())*1e9/cfg['unit_ns'], sigma=.6)


def calibration_record(cfg, seed):
    theta = [cfg['a'], 15.3, cfg['b']]
    rate, _ = template(cfg).rate(theta)
    q = cfg['condition']
    if q['afterpulse']:
        observed, _ = previous.with_afterpulses(rate, 3000, cfg['td']*q['td_scale'],
            cfg['nominal_te']*q['te_scale'], seed, q['afterpulse'])
        return observed
    return model.simulate(rate, 3000, cfg['td']*q['td_scale'],
        cfg['nominal_te']*q['te_scale'], 0., seed, cfg['jitter'], random_recovery=False)


def fit_sensor(cfg, index):
    if cfg['measured_template']:
        return dict(name=cfg['name'], mu=cfg['mu'], sd=cfg['sd'], jitter=0.,
            source='Existing independent high-flux measured calibration segment [100,20100)',
            source_sha256=sha(ROOT/'results/measured_recovery_v1/analysis.json'))
    observed = calibration_record(cfg, 82375001001+index*1000003)
    pulse = template(cfg); minimum = float(np.diff(observed).min())
    scale = len(observed)-1; candidates = []
    bounds = [(1e-5, 20.), (11.3, 19.3), (1e-5, 30.),
              (70., min(90., minimum+max(2., 5*cfg['jitter']))), (np.log(.06), np.log(5.))]
    for sd in [.2, 1.2]:
        initial = np.array([2.4, 15.3, .8, min(80., minimum-.1), np.log(sd)])
        def value_gradient(x):
            geo = model.prepare(observed, 64, cfg['td'], x[3], np.exp(x[4]), cfg['jitter'])
            rate, design = pulse.rate(x[:3]); ll, grad = model.evaluate(rate, design, geo)
            if not np.isfinite(ll):
                return 1e50, np.zeros(5)
            derivatives = []
            for j, step in [(3, 1e-3), (4, 1e-3)]:
                plus, minus = x.copy(), x.copy(); plus[j]+=step; minus[j]-=step
                pg = model.prepare(observed, 64, cfg['td'], plus[3], np.exp(plus[4]), cfg['jitter'])
                mg = model.prepare(observed, 64, cfg['td'], minus[3], np.exp(minus[4]), cfg['jitter'])
                derivatives.append((model.evaluate(rate, design, pg)[0]-model.evaluate(rate, design, mg)[0])/(2*step))
            return -ll/scale, -np.r_[grad, derivatives]/scale
        start=time.perf_counter()
        fit = minimize(value_gradient, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(ftol=2e-9, gtol=3e-5, maxiter=70, maxls=20))
        candidates.append(dict(x=fit.x.tolist(), negative_mean_logdensity=float(fit.fun),
            success=bool(fit.success), evaluations=int(fit.nfev), seconds=time.perf_counter()-start))
    best=min(candidates,key=lambda r:r['negative_mean_logdensity'])
    return dict(name=cfg['name'], mu=best['x'][3], sd=float(np.exp(best['x'][4])),
        jitter=cfg['jitter'], candidates=candidates, calibration_events=len(observed),
        calibration_data_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        minimum_calibration_gap=minimum,
        source='Independent 3000-cycle single-surface acquisition; Td nominal, jitter supplied as instrument setting')


def calibrate():
    OUT.mkdir(exist_ok=True)
    names=[c['name'] for c in json.loads(previous.P.read_text())['robustness']['conditions']]+['measured_high']
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fit_sensor, setting(name), i):name for i,name in enumerate(names)
                 if not (OUT/('sensor_'+name+'.json')).exists()}
        for future in as_completed(futures):
            row=future.result(); (OUT/('sensor_'+row['name']+'.json')).write_text(json.dumps(row,indent=2))
            print('Sensor calibration',row['name'],'mu',row['mu'],'sd',row['sd'],flush=True)


def freeze():
    if P.exists():raise FileExistsError(P)
    names=[c['name'] for c in json.loads(previous.P.read_text())['robustness']['conditions']]+['measured_high']
    sensors={name:(json.loads((OUT/('sensor_'+name+'.json')).read_text())
                   if (OUT/('sensor_'+name+'.json')).exists() else None) for name in names}
    protocol=dict(status='FIXED_BEFORE_EXTENSION_DETECTION_OUTCOMES',
        source_hashes={str(Path(n)):sha(ROOT/n) for n in ['src/calibrated_recovery.py','tools/calibrated_recovery_study.py']},
        sensor_calibration=sensors, settings={name:setting(name) for name in names},
        pending_calibration='Any null entry is a not-yet-completed output of the fixed independent sensor-fit algorithm, not a free choice. It is read from sensor_<name>.json at execution and all final fitted outputs are hash-pinned in analysis.json. No query outcomes determine the fit.',
        previous_mismatch_protocol_sha256=sha(previous.P),
        previous_results='The original 864 mismatch acquisitions and their fixed-law outcomes were previously inspected. This extension reuses every acquisition and reports all conditions. New sensor calibration, normalization training and cohort threshold simulations use disjoint seeds.',
        model='Electronics recovery normal truncated to [Td,3Td], integrated inside each transition density with eight probability-space Gauss-Legendre nodes. Up to two hidden avalanches. Jitter: three-point Gauss-Hermite for preceding endpoint; split succeeding-endpoint error at every event-rate bin boundary and use three-point Gauss-Legendre per segment over +/-7 standard deviations. Integrated pair criterion, not exact noisy-path likelihood. Complete simulated paths retain shared endpoint errors. One strongest histogram-template peak starts the null fit.',
        calibration='Independent sensor calibration precedes all queries. Independent matrix bank: two amplitudes (0.7,1.3)*nominal, two backgrounds (0.75,1.25)*nominal, fractional phases 0,0.5 at integer origin15; eight acquisitions/cell. Convex interpolation and integer phase rotation, as in original bank. Independent cohort-level null calibration: 499 paths drawn over stated nuisance ranges. Each path refits nuisance parameters and scans all candidate delays. This is population-mixture calibration, not original per-query bootstrap; all extension methods use the same calibration paths.',
        null_training_records=499, matrix_records_per_cell=8,
        alpha=.01, recovery_order=8, jitter_order=3,
        seeds=dict(bank=93271000001, threshold=104772000001, measured_positive=119372000001),
        evaluation='Original 32 positive and64 null acquisitions in all nine mismatch conditions. Measured-setting simulation: measured high-flux pulse shape, measured fitted a/B/Td/recovery; ratios0.0625,0.125,0.25,64 positives per ratio and128 nulls, separation10bins (15.625ns). Public measured nulls:128 disjoint1000-cycle blocks after calibration/development/evaluation segments. Known single-surface records evaluate false alarms only. Simulated second returns are added to incident flux before both recovery clocks.',
        endpoints='All records: applicability, declaration, correct/wrong localization, detection miss; conditional location RMSE reported with count. Bin tolerance2; thresholds estimated on training only; no outcome-dependent scenario removal.',
        practical='This stage addresses M2/M3. No new equal-population-FAR superiority claim over GLRT, no physical two-surface measurement, no real-time claim.', workers=12)
    P.write_text(json.dumps(protocol,indent=2));print('Frozen extension protocol',sha(P),flush=True)


def check():
    protocol=json.loads(P.read_text())
    for path,digest in protocol['source_hashes'].items():assert sha(ROOT/path)==digest,path
    for name,value in protocol['sensor_calibration'].items():
        path=OUT/('sensor_'+name+'.json')
        if value is None and path.exists():protocol['sensor_calibration'][name]=json.loads(path.read_text())
    return protocol


def geometry(observed,cfg,sensor):
    return model.prepare(observed,64,cfg['td'],sensor['mu'],sensor['sd'],sensor['jitter'])


def bank_cell(task):
    name, ci, ai, bi, fi=task; protocol=check(); cfg=protocol['settings'][name]
    sensor=protocol['sensor_calibration'][name]; pulse=template(cfg)
    theta=[cfg['a']*[.7,1.3][ai],15.+[0.,.5][fi],cfg['b']*[.75,1.25][bi]]
    rate,_=pulse.rate(theta); h=np.zeros((64,64)); timings=[]
    for j in range(protocol['matrix_records_per_cell']):
        seed=protocol['seeds']['bank']+ci*10000019+(ai*4+bi*2+fi)*100003+j*1009
        observed=model.simulate(rate,cfg['cycles'],cfg['td'],sensor['mu'],sensor['sd'],seed,sensor['jitter'])
        start=time.perf_counter();geo=geometry(observed,cfg,sensor)
        _,_,_,pieces=model.evaluate(rate,np.eye(64),geo,True)
        h+=pieces.T@pieces;timings.append(time.perf_counter()-start)
    return name,ai,bi,fi,h/protocol['matrix_records_per_cell'],sum(timings)


def train():
    protocol=check(); names=list(protocol['settings']); matrices={name:np.zeros((1,2,2,2,64,64)) for name in names}
    seconds={name:0. for name in names}
    tasks=[(name,ci,ai,bi,fi) for ci,name in enumerate(names)
           if protocol['sensor_calibration'][name] is not None and not (OUT/('bank_'+name+'.npz')).exists()
           for ai in range(2) for bi in range(2) for fi in range(2)]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        for name,ai,bi,fi,h,elapsed in pool.map(bank_cell,tasks):
            matrices[name][0,ai,bi,fi]=h;seconds[name]+=elapsed
            print('Bank cell',name,ai,bi,fi,flush=True)
    for name in names:
        if not any(task[0]==name for task in tasks):continue
        cfg=protocol['settings'][name]
        np.savez_compressed(OUT/('bank_'+name+'.npz'),information=matrices[name],
            amplitudes=np.array([[.7,1.3]])*cfg['a'], backgrounds=np.array([[.75,1.25]])*cfg['b'],
            fractions=[0.,.5],regimes=[0],base_phase_integer=15)
    available=[name for name in names if (OUT/('bank_'+name+'.npz')).exists()]
    (OUT/'bank_training.json').write_text(json.dumps(dict(status='COMPLETE' if len(available)==len(names) else 'PARTIAL',records=64*len(available),
        seconds=time.perf_counter()-start,component_seconds=seconds,protocol_sha256=sha(P)),indent=2))


def censored(observed,pulse,cfg,sensor):
    lag=cfg['td']+min(3*cfg['td'],sensor['mu']+4*sensor['sd'])+6*sensor['jitter']
    # First/last noisy events outside the acquisition do not define an exposure.
    # The conditional full criterion uses all pairs; exposure control uses the
    # specified finite acquisition and any outside event is reported, not clipped.
    if observed[0]<0 or observed[-1]>=cfg['cycles']*64:
        return dict(score=None,applicable=False)
    counts,exposure,_=safe_statistics(observed,cfg['cycles'],64,lag)
    scale=max(1.,counts.sum());grid, candidates=pulse.bank(1)
    centres=np.argsort(candidates@counts)[-2:][::-1];fits=[]
    for t in centres:
        def objective(x):
            rate,j=pulse.rate(x);ll=counts@np.log(rate)-exposure@rate
            return -ll/scale,-j.T@(counts/rate-exposure)/scale
        fit=minimize(objective,[cfg['a'],t,cfg['b']],jac=True,method='L-BFGS-B',
            bounds=[(1e-5,20),(float(t)-4,float(t)+4),(1e-5,30)],options=dict(ftol=1e-10,gtol=1e-6,maxiter=80))
        fits.append(fit)
    fit=min(fits,key=lambda r:r.fun);rate,j=pulse.rate(fit.x);u=counts/rate-exposure;h=np.diag(exposure/rate)
    grid, candidates=pulse.bank();cross=candidates@h@j;inverse=np.linalg.pinv(j.T@h@j,rcond=1e-10)
    raw=candidates@u-cross@inverse@(j.T@u)
    variance=np.einsum('mi,ij,mj->m',candidates,h,candidates)-np.einsum('mi,ij,mj->m',cross,inverse,cross)
    distance=np.abs((grid-fit.x[1]+32)%64-32);allowed=(distance>=max(3.,2.5*pulse.sigma))&(variance>1e-9)
    scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(variance,1e-9),-np.inf);best=int(np.argmax(scores))
    centre=grid[best];a,t,b=fit.x
    def alternative(x):
        rate,j=pulse.rate(x);return -(counts@np.log(rate)-exposure@rate)/scale,-j.T@(counts/rate-exposure)/scale
    alt=minimize(alternative,[a,t,b,.02,centre],jac=True,method='L-BFGS-B',
        bounds=[(1e-5,20),(t-4,t+4),(1e-5,30),(0,20),(centre-4,centre+4)],
        options=dict(ftol=1e-10,gtol=1e-6,maxiter=100))
    x=alt.x;p,d=x[1]%64,x[4]%64
    if x[3]>x[0]:p,d=d,p
    return dict(score=float(scores[best]),primary=float(p),delay=float(d),applicable=True,
        null_theta=fit.x.tolist(),optimizer_success=bool(fit.success),localizer_success=bool(alt.success))


def acquisition(task, protocol):
    name,kind,index=task;cfg=protocol['settings'][name];sensor=protocol['sensor_calibration'][name]
    pulse=template(cfg);ci=list(protocol['settings']).index(name)
    if kind=='threshold':
        seed=protocol['seeds']['threshold']+ci*10000019+index*1009
        rng=np.random.default_rng(seed);theta=[cfg['a']*rng.uniform(.7,1.3),rng.uniform(14.15,16.45),cfg['b']*rng.uniform(.75,1.25)]
        observed=model.simulate(pulse.rate(theta)[0],cfg['cycles'],cfg['td'],sensor['mu'],sensor['sd'],seed+100003,sensor['jitter'])
        return observed,False,None,theta
    if name!='measured_high':
        p=json.loads(previous.P.read_text())
        cases=[c for c in previous.cases(p,'robust') if c['condition']['name']==name]
        case=cases[index];observed,truth,theta=previous.acquisition(case,p)
        return observed,case['positive'],truth,theta
    seed=protocol['seeds']['measured_positive']+index*100003
    rng=np.random.default_rng(seed);a=cfg['a']*rng.uniform(.7,1.3);b=cfg['b']*rng.uniform(.75,1.25);t=rng.uniform(14.15,16.45)
    positive=index<192;ratio=[.0625,.125,.25][index//64] if positive else 0.
    d=t+10+rng.uniform(-.35,.35);theta=[a,t,b,ratio*a,d]
    observed=model.simulate(pulse.rate(theta)[0],cfg['cycles'],cfg['td'],cfg['mu'],cfg['sd'],seed+1009,0.)
    return observed,positive,dict(primary=t,secondary=d,ratio=ratio),theta


def worker(task):
    protocol=check();name,kind,index=task;cfg=protocol['settings'][name];sensor=protocol['sensor_calibration'][name]
    start=time.perf_counter();observed,positive,truth,theta=acquisition(task,protocol)
    pulse=template(cfg);geo=geometry(observed,cfg,sensor)
    fit=model.fit_null(observed,pulse,geo);bank=NullInformationBank.load(OUT/('bank_'+name+'.npz'))
    information,flags=bank.lookup(fit['theta'],0)
    full=model.scan(pulse,fit['theta'],geo,information)
    if kind!='threshold':
        full.update(model.localize(pulse,fit['theta'],full['delay'],geo))
    full.update(applicable=True,optimizer_success=fit['success'])
    safe=censored(observed,pulse,cfg,sensor)
    return dict(id=f'{name}/{kind}/{index:04}',name=name,kind=kind,index=index,positive=positive,
        truth=truth,generating_theta=theta,events=len(observed),
        data_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        methods=dict(calibrated=full,censored=safe),null_fit=fit,bank_flags=flags,
        seconds=time.perf_counter()-start)


def run(kind):
    protocol=check();start=time.perf_counter();names=[name for name in protocol['settings'] if (OUT/('bank_'+name+'.npz')).exists()]
    tasks=[(name,kind,j) for name in names for j in range(protocol['null_training_records'] if kind=='threshold' else (320 if name=='measured_high' else 96))]
    path=OUT/'records.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={r[0] for r in db.execute('SELECT id FROM records')}
        tasks=[task for task in tasks if f'{task[0]}/{task[1]}/{task[2]:04}' not in done]
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures={pool.submit(worker,task):task for task in tasks}
            for i,future in enumerate(as_completed(futures),1):
                row=future.result();db.execute('INSERT INTO records VALUES(?,?)',(row['id'],json.dumps(row)));db.commit()
                if i%16==0:print(kind,i,'/',len(tasks),'latest',row['name'],'seconds',round(time.perf_counter()-start,1),flush=True)
    (OUT/(kind+'_completion.json')).write_text(json.dumps(dict(status='COMPLETE',new_records=len(tasks),seconds=time.perf_counter()-start,protocol_sha256=sha(P)),indent=2))


def measured_blocks():
    cfg=setting('measured_high');raw=loadmat(ROOT/'data/highflux_raw/2019_01_28/FGS_td048_2019_01_28_acq2.mat')
    frames=raw['frameNum'].ravel().astype(np.int64);phase=raw['detTime'].ravel().astype(float)/cfg['unit_ns']
    first=int(frames[70400])+1000;rows=[]
    for i in range(128):
        start=first+i*1000;lo=np.searchsorted(frames,start);hi=np.searchsorted(frames,start+1000)
        observed=(frames[lo:hi]-start)*64.+phase[lo:hi]
        rows.append((i,observed))
    return rows


def measured_worker(task):
    index,observed=task;protocol=check();name='measured_high';cfg=protocol['settings'][name];sensor=protocol['sensor_calibration'][name]
    pulse=template(cfg);geo=geometry(observed,cfg,sensor);fit=model.fit_null(observed,pulse,geo)
    bank=NullInformationBank.load(OUT/('bank_'+name+'.npz'));h,flags=bank.lookup(fit['theta'],0)
    full=model.scan(pulse,fit['theta'],geo,h);full.update(applicable=True,optimizer_success=fit['success'])
    return dict(id=f'{name}/public/{index:04}',name=name,kind='public',index=index,positive=False,truth=None,
        events=len(observed),data_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        methods=dict(calibrated=full,censored=censored(observed,pulse,cfg,sensor)),null_fit=fit,bank_flags=flags)


def public():
    check()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        done={r[0] for r in db.execute('SELECT id FROM records')}
        tasks=[t for t in measured_blocks() if f'measured_high/public/{t[0]:04}' not in done]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for row in pool.map(measured_worker,tasks):
                db.execute('INSERT INTO records VALUES(?,?)',(row['id'],json.dumps(row)));db.commit()
                if (row['index']+1)%16==0:print('Measured public nulls',row['index']+1,'/128',flush=True)


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def analyze():
    protocol=check()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        rows=[json.loads(row[0]) for row in db.execute('SELECT payload FROM records ORDER BY id')]
    output=[];thresholds={}
    for name in protocol['settings']:
        train=[r for r in rows if r['name']==name and r['kind']=='threshold']
        assert len(train)==499,(name,len(train))
        thresholds[name]={}
        for method in ['calibrated','censored']:
            scores=np.array([r['methods'][method]['score'] for r in train],float)
            assert np.all(np.isfinite(scores))
            thresholds[name][method]=float(np.sort(scores)[494])
        tests=[r for r in rows if r['name']==name and r['kind']=='evaluate']
        assert len(tests)==(320 if name=='measured_high' else 96),(name,len(tests))
        for kind in ['evaluate','public']:
            subset=[r for r in rows if r['name']==name and r['kind']==kind]
            if not subset:continue
            groups=[None] if name!='measured_high' or kind=='public' else [.0625,.125,.25]
            for ratio in groups:
                cohort=[r for r in subset if not r['positive'] or ratio is None or r['truth']['ratio']==ratio]
                for method in ['calibrated','censored']:
                    npos=sum(r['positive'] for r in cohort);nnull=len(cohort)-npos
                    correct=wrong=declare=false=unavailable=0;errors=[]
                    for r in cohort:
                        v=r['methods'][method]
                        if not v['applicable']:unavailable+=1;continue
                        detected=v['score']>thresholds[name][method]
                        if not r['positive']:false+=int(detected);continue
                        declare+=int(detected)
                        if detected:
                            truth=r['truth'];ep=abs((v['primary']-truth['primary']+32)%64-32);ed=abs((v['delay']-truth['secondary']+32)%64-32)
                            correct+=int(ep<=2 and ed<=2);wrong+=int(ep>2 or ed>2);errors.append(ed)
                    output.append(dict(name=name,kind=kind,ratio=ratio,method=method,
                        positives=npos,nulls=nnull,declared=declare,correct=correct,wrong=wrong,false_alarms=false,
                        unavailable=unavailable,false_alarm_interval=interval(false,nnull) if nnull else None,
                        conditional_secondary_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                        threshold=thresholds[name][method]))
    result=dict(status='COMPLETE',protocol_sha256=sha(P),records=len(rows),thresholds=thresholds,rows=output,
        sensor_calibration=protocol['sensor_calibration'],sensor_output_hashes={name:sha(OUT/('sensor_'+name+'.json')) for name in protocol['settings']})
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    for row in output:print(row)


if __name__=='__main__':
    action=sys.argv[1]
    if action in ['threshold','evaluate']:run(action)
    else:globals()[action]()
