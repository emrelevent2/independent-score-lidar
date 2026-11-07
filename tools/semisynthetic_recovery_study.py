"""Conditionally complete measured photon paths before injecting a weak return.

These are model-dependent counterfactuals, not measured two-surface scenes.
The observed one-surface trace is reproduced at zero injection. Unobserved
incident photons during detector recovery are completed too, so adding photons
can change subsequent avalanches and electronics state correctly.
"""
from pathlib import Path
import sys,json,hashlib,time,sqlite3
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import ndtr,ndtri
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import calibrated_recovery_study as study
import calibrated_recovery as model
from measurement_recovery_model import primitive,instantaneous
from null_information_bank import NullInformationBank
from deadtime_path_bootstrap import njit
OUT=ROOT/'results/semisynthetic_recovery_v1'
P=ROOT/'SEMISYNTHETIC_RECOVERY_PROTOCOL_20261107.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    assert not P.exists()
    p=dict(status='FIXED_BEFORE_COUNTERFACTUAL_DETECTION_RESULTS',source_sha256=sha(__file__),
        parent_protocol_sha256=sha(study.P),source_blocks='First64 of the128 prospectively selected disjoint OD0 public blocks, after event70400 and1000-period buffer. Reuse across three injection strengths; no independence claim across strengths.',
        ratios=[.0625,.125,.25],separation_bins=10.,separation_perturbation=.35,blocks=64,seed_base=134719000001,
        augmentation='Condition on each source block and its fitted one-return rate. For every gap sample zero/one hidden avalanche from the calibrated transition law (R<=2Td). Sample recovery R conditional on the path and observed gap. Complete incident Poisson photons during all detector-dead intervals. Source recorded-event recovery marks are posterior draws; all other photon marks are prior draws. Counterfactual acquisition is conditioned on its first recorded event; no photons injected before that anchor. Apply both physical recovery filters to all incident photons, not a merge of recorded streams.',
        computation='Hidden-event mixture weight integrated by16-point Gauss-Legendre on segments split at rate edges and recovery-CDF transition points; exact rejection sampling inside the selected hidden path. Gaussian recovery truncated to[Td,2Td] (the mass above2Td under high-flux calibration is below1e-39).',
        checks='All64 source traces must be reproduced at zero injection. Complete background photon path is shared across all ratios. Separate injection seeds, no selection by detector output. Quadrature refinement checked before efficacy. Score bank and499-record thresholds are unchanged from measured-sensor study.',
        endpoints='Secondary-return declarations, injected-secondary localization within2bins, wrong-secondary count, and secondary RMSE conditional on declaration. No measured primary-surface ground truth is inferred. Compare calibrated and censored methods on identical counterfactual records. Public128 single-surface null results supply the measured false-alarm evaluation.',workers=12)
    P.write_text(json.dumps(p,indent=2));print('Frozen semi-synthetic protocol',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());assert sha(__file__)==p['source_sha256'];assert sha(study.P)==p['parent_protocol_sha256']
    return p,study.check()


def cdf(s,mu,sd,d):
    lo,hi=ndtr((d-mu)/sd),ndtr((2*d-mu)/sd)
    return np.clip((ndtr((np.asarray(s)-mu)/sd)-lo)/(hi-lo),0.,1.)


def recovery_draw(rng,lower,upper,mu,sd):
    lo,hi=ndtr((lower-mu)/sd),ndtr((upper-mu)/sd)
    if hi<=lo:raise FloatingPointError('Empty posterior recovery interval')
    return float(mu+sd*ndtri(np.clip(lo+(hi-lo)*rng.uniform(),1e-300,1-1e-16)))


def hidden_mass(phase,gap,rate,d,mu,sd,order=16):
    upper=min(2*d,gap-d)
    if upper<=d:return 0.
    knots=[d,upper]
    for shift in [0.,-d]:
        for integer in range(int(np.floor(d+phase-shift))-1,int(np.ceil(upper+phase-shift))+2):
            v=integer-phase+shift
            if d<v<upper:knots.append(v)
    for z in [-8,-4,-2,-1,0,1,2,4,8]:
        v=mu+sd*z
        if d<v<upper:knots.append(v)
    knots=np.unique(knots);z,w=model.legendre(order)
    v=knots[:-1,None]+np.diff(knots)[:,None]*z
    h=instantaneous(rate,phase+v,64)*np.exp(primitive(rate,phase+v+d,64)-primitive(rate,phase+v,64))
    return float(np.sum(h*(cdf(gap,mu,sd,d)-cdf(v,mu,sd,d))*np.diff(knots)[:,None]*w))


def complete(observed,rate,cfg,sensor,seed):
    rng=np.random.default_rng(seed);d=cfg['td'];mu=sensor['mu'];sd=sensor['sd'];end=cfg['cycles']*64.
    latent=[];record_marks=[];max_difference=0.;hidden_count=0
    for i,gap in enumerate(np.diff(observed)):
        t=observed[i];phase=t%64
        mass=hidden_mass(phase,gap,rate,d,mu,sd)
        if i%71==0:
            refined=hidden_mass(phase,gap,rate,d,mu,sd,32)
            max_difference=max(max_difference,abs(refined-mass))
        g=float(cdf(gap,mu,sd,d));latent.append(t)
        if rng.uniform() < mass/(g+mass):
            upper=min(2*d,gap-d)
            bound=float(rate.max()*np.exp(np.ceil(d/64)*rate.sum())*g)
            for attempt in range(100000):
                v=rng.uniform(d,upper)
                h=float(instantaneous(rate,phase+v,64)*np.exp(primitive(rate,phase+v+d,64)-primitive(rate,phase+v,64)))
                density=h*(g-float(cdf(v,mu,sd,d)))
                assert density<=bound*(1+1e-12)
                if rng.uniform()*bound<density:break
            else:raise RuntimeError('Hidden-path rejection sampler exhausted')
            latent.append(t+v);hidden_count+=1
            R=recovery_draw(rng,max(d,v),min(2*d,gap),mu,sd)
        else:R=recovery_draw(rng,d,min(2*d,gap),mu,sd)
        record_marks.append(R)
    latent.append(observed[-1]);record_marks.append(recovery_draw(rng,d,2*d,mu,sd))
    latent=np.sort(latent);assert np.all(np.diff(latent)>=d-1e-9)
    length=np.minimum(d,end-latent);numbers=rng.poisson(rate.max()*length)
    owners=np.repeat(np.arange(len(latent)),numbers)
    candidates=latent[owners]+length[owners]*rng.uniform(size=len(owners))
    keep=rng.uniform(size=len(candidates))<instantaneous(rate,candidates,64)/rate.max()
    incident=np.r_[latent,candidates[keep]]
    marks=mu+sd*ndtri(ndtr((d-mu)/sd)+(ndtr((2*d-mu)/sd)-ndtr((d-mu)/sd))*rng.uniform(size=len(incident)))
    # Replace exactly the marks whose conditioning produced the measured trace.
    for t,R in zip(observed,record_marks):
        positions=np.flatnonzero(incident==t);assert len(positions)==1
        marks[positions[0]]=R
    order=np.argsort(incident);incident=incident[order];marks=marks[order]
    replay=filter_incident(incident,marks,d)
    assert np.array_equal(replay,observed),('zero-injection roundtrip failed',len(replay),len(observed))
    return incident,marks,dict(hidden_avalanches=hidden_count,incident_photons=len(incident),
        source_events=len(observed),quadrature_max_absolute_difference=max_difference,
        zero_injection_roundtrip=True,completed_path_sha256=hashlib.sha256(np.c_[incident,marks].tobytes()).hexdigest())


@njit(cache=True)
def filter_incident(incident,marks,td):
    out=np.empty(len(incident));n=0;detector=0.;electronics=0.
    for j in range(len(incident)):
        t=incident[j]
        if t<detector:continue
        detector=t+td
        if t>=electronics:
            out[n]=t;n+=1;electronics=t+marks[j]
    return out[:n]


def worker(task):
    index,observed,fit=task;p,parent=check();cfg=parent['settings']['measured_high'];sensor=parent['sensor_calibration']['measured_high']
    pulse=study.template(cfg);theta=fit['theta'];rate,_=pulse.rate(theta)
    incident,marks,diagnostic=complete(observed,rate,cfg,sensor,p['seed_base']+index*1000003)
    rng=np.random.default_rng(p['seed_base']+index*1000003+101)
    target=(theta[1]+10+rng.uniform(-.35,.35))%64
    shape=pulse.pulse(target)[0];result=[]
    bank=NullInformationBank.load(study.OUT/'bank_measured_high.npz')
    for ratio in p['ratios']:
        injection_seed=p['seed_base']+index*1000003+int(ratio*100000)*1009
        rr=np.random.default_rng(injection_seed);a=ratio*theta[0];n=rr.poisson(a*cfg['cycles'])
        phase=rr.choice(64,size=n,p=shape)+rr.uniform(size=n)
        extra=rr.integers(0,cfg['cycles'],size=n)*64.+phase
        extra=extra[extra>=observed[0]]
        extra_marks=sensor['mu']+sensor['sd']*ndtri(ndtr((cfg['td']-sensor['mu'])/sensor['sd'])+
            (ndtr((2*cfg['td']-sensor['mu'])/sensor['sd'])-ndtr((cfg['td']-sensor['mu'])/sensor['sd']))*rr.uniform(size=len(extra)))
        all_events=np.r_[incident,extra];all_marks=np.r_[marks,extra_marks];order=np.argsort(all_events)
        counterfactual=filter_incident(all_events[order],all_marks[order],cfg['td'])
        geo=study.geometry(counterfactual,cfg,sensor);null=model.fit_null(counterfactual,pulse,geo)
        h,flags=bank.lookup(null['theta'],0);full=model.scan(pulse,null['theta'],geo,h)
        full.update(model.localize(pulse,null['theta'],full['delay'],geo));full['applicable']=True
        safe=study.censored(counterfactual,pulse,cfg,sensor)
        result.append(dict(id=f'{index:03}/{ratio}',index=index,ratio=ratio,target_delay=float(target),
            source_data_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
            counterfactual_data_sha256=hashlib.sha256(counterfactual.tobytes()).hexdigest(),
            injected_photons=len(extra),events=len(counterfactual),methods=dict(calibrated=full,censored=safe),
            source_completion=diagnostic,bank_flags=flags))
    return result


def run():
    p,parent=check();OUT.mkdir(exist_ok=True)
    with sqlite3.connect(study.OUT/'records.sqlite3') as db:
        saved=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
    by_index={r['index']:r for r in saved if r['kind']=='public'}
    assert len(by_index)==128
    blocks=study.measured_blocks()[:64]
    tasks=[(index,obs,by_index[index]['null_fit']) for index,obs in blocks]
    start=time.perf_counter()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={json.loads(r[0])['index'] for r in db.execute('SELECT payload FROM records')}
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            futures=[pool.submit(worker,t) for t in tasks if t[0] not in done]
            for future in as_completed(futures):
                group=future.result()
                for row in group:db.execute('INSERT INTO records VALUES(?,?)',(row['id'],json.dumps(row)))
                db.commit();print('Counterfactual source block',group[0]['index'],'complete',flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(status='COMPLETE',source_blocks=64,records=192,
        protocol_sha256=sha(P),seconds=time.perf_counter()-start),indent=2))


def analyze():
    p,parent=check();base=json.loads((study.OUT/'analysis.json').read_text())
    with sqlite3.connect(OUT/'records.sqlite3') as db:records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY id')]
    assert len(records)==192 and len({r['source_data_sha256'] for r in records})==64
    assert all(r['source_completion']['zero_injection_roundtrip'] for r in records)
    assert max(r['source_completion']['quadrature_max_absolute_difference'] for r in records)<1e-7
    output=[]
    for ratio in p['ratios']:
        selected=[r for r in records if r['ratio']==ratio];assert len(selected)==64
        for method in ['calibrated','censored']:
            threshold=base['thresholds']['measured_high'][method];declare=correct=wrong=0;errors=[]
            for r in selected:
                v=r['methods'][method]
                if v['score']>threshold:
                    declare+=1;error=abs((v['delay']-r['target_delay']+32)%64-32)
                    correct+=int(error<=2);wrong+=int(error>2);errors.append(error)
            output.append(dict(ratio=ratio,method=method,records=64,declared=declare,correct_secondary=correct,
                wrong_secondary=wrong,conditional_secondary_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None))
    result=dict(status='COMPLETE',protocol_sha256=sha(P),parent_analysis_sha256=sha(study.OUT/'analysis.json'),
        records=192,source_blocks=64,zero_injection_roundtrips=64,
        hidden_avalanches_completed=sum(r['source_completion']['hidden_avalanches'] for r in records if r['ratio']==.0625),
        quadrature_max_absolute_difference=max(r['source_completion']['quadrature_max_absolute_difference'] for r in records),rows=output)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':globals()[sys.argv[1]]()
