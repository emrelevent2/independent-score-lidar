"""Pre-specified development grid for fast second-surface testing.

Monte Carlo calibration is exchangeable ONLY within the declared simulator
regime (including its randomized primary phase). It is NOT a uniform test for
unknown null fluxes, pulse shapes, recovery times,or real hardware.
"""
from pathlib import Path
import sys,json,time,hashlib,sqlite3,platform
import numpy as np
import scipy
from deadtime_exposure import simulate_events,sufficient_statistics
from deadtime_scan import pulse,rounded_opportunities,scan,multistart_glrt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'DEADTIME_SCAN_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/deadtime_scan_development_v1'
METHODS=['histogram','rounded_ded','duration_raw','duration_efficient','duration_glrt']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Protocol already frozen')
    names=['deadtime_scan_development.py','deadtime_scan.py','deadtime_exposure.py','test_deadtime_scan.py','test_deadtime_exposure.py']
    regimes=[dict(name='moderate_long',a=.8,b=.3,cycles=1500,dead=83.7,secondary=.065,gap=12),
             dict(name='near_double_period',a=.8,b=1.,cycles=1500,dead=126.7,secondary=.09,gap=6),
             dict(name='high_flux_short',a=2.4,b=.8,cycles=1000,dead=31.7,secondary=.15,gap=10),
             dict(name='low_flux_long',a=.25,b=.1,cycles=1500,dead=83.7,secondary=.035,gap=20)]
    p=dict(status='FROZEN_BEFORE_NEW_SIMULATED_SCANS',source_hashes={n:sha(ROOT/'src'/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
           methods=METHODS,regimes=regimes,settings=dict(bins=64,sigma=1.5,primary_phase=15.3,phase_jitter_uniform=[-.4,.4],seed_base=624815731,calibration_null_per_regime=199,test_null_per_regime=64,test_positive_per_regime=64,alpha=.01,tolerance_bins=2,initial_state='active'),
           calibration='Per-method/per-regime Monte Carlo rank p=(1+count(cal>=test))/(199+1);declare p<=.01. Calibration excludes all test scores;truth used only for evaluation. Exchangeability only in the prespecified simulator mixture,not composite-null uniformity.',
           strong_control='Continuous-duration five-parameter multistart GLRT; six separated scan seeds plus nested null,continuous delays. Numerical local optimization,not a certified global maximum. RoundedDED control includes partial-active bins as whole opportunities,matching source discretization principle;known likelihood,not claiming new theory.',
           continuation_rule='Exploratory only: duration_efficient gains correct over rounded_ded with no more wrong-positive or null declarations,AND retains >=90% GLRT correct with no more wrong/null and >=2x median pipeline speed. Failure retained. Fresh confirmation and further priority review needed even if pass.',
           scope='New synthetic DEVELOPMENT grid for a distinct cross-cycle dead-time observation model. Not independent confirmation of old first-event CASPI method,not FPGA speed,not real physical accuracy,not Q1 readiness.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text())
    for n,h in p['source_hashes'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    return p


def cases(p):
    cfg=p['settings'];i=0
    for gi,g in enumerate(p['regimes']):
        for group,num in [('calibration_null',cfg['calibration_null_per_regime']),('test_null',cfg['test_null_per_regime']),('test_positive',cfg['test_positive_per_regime'])]:
            for j in range(num):
                yield dict(id=f'{gi}_{group}_{j:03}',index=i,regime=gi,group=group,seed=cfg['seed_base']+i*1009)
                i+=1


def data(case,p):
    cfg=p['settings'];g=p['regimes'][case['regime']];k=cfg['bins'];sig=cfg['sigma']
    tau=cfg['primary_phase']+np.random.default_rng(case['seed']).uniform(*cfg['phase_jitter_uniform'])
    f,_=pulse(tau,k,sig);rates=g['a']*f+g['b']/k
    if case['group']=='test_positive':rates=rates+g['secondary']*pulse((tau+g['gap'])%k,k,sig)[0]
    events=simulate_events(rates,g['cycles'],g['dead'],case['seed']+1)
    s,e=sufficient_statistics(events,g['cycles'],k,g['dead']);a=rounded_opportunities(events,g['cycles'],k,g['dead'])
    assert np.all(a>=s) and np.all(a>=e-1e-7)
    return events,s,e,a,dict(primary=float(tau),secondary=float((tau+g['gap'])%k))


def pipeline(s,e,a,case,p,method):
    start=time.perf_counter();g=p['regimes'][case['regime']];sig=p['settings']['sigma']
    if method=='histogram':r=scan(s,np.full_like(e,g['cycles']),sig)
    elif method=='rounded_ded':r=scan(s,a,sig,mode='ded')
    else:
        r=scan(s,e,sig,adjusted=method!='duration_raw')
        if method=='duration_glrt':r=multistart_glrt(s,e,sig,r)
    # Keep scalar outputs;raw phase scores are reproducible from the data.
    result=dict(method=method,score=float(r['score']),delay=float(r['delay']),primary=float(r['primary']),theta=np.asarray(r['theta']).tolist(),seconds=time.perf_counter()-start)
    if 'fit' in r:result['null_optimizer_success']=r['fit']['optimizer_success']
    if 'optimizer_success' in r:result['alternative_optimizer_success']=r['optimizer_success']
    return result


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3')
    c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(v[0]) for v in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();c=connect();done={v[0] for v in c.execute('SELECT id FROM records')};start=time.perf_counter();total=len(list(cases(p)));new=0
    for case in cases(p):
        if case['id'] in done:continue
        events,s,e,a,truth=data(case,p);samples=[];order=METHODS[case['index']%5:]+METHODS[:case['index']%5]
        for method in order:samples.append(pipeline(s,e,a,case,p,method))
        h=hashlib.sha256();h.update(events.tobytes());h.update(s.tobytes());h.update(e.tobytes());h.update(a.tobytes())
        row=dict(case=case,truth=truth,events=len(events),data_sha256=h.hexdigest(),samples=samples)
        c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(row)));c.commit();new+=1
        if (len(done)+new)%20==0:print('New dead-time development',len(done)+new,'/',total,'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==total
    v=dict(status='COMPLETE',cases=total,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),newly_computed_this_invocation=new,seconds_this_invocation=time.perf_counter()-start)
    (OUT/'completion.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)


def analyze():
    p=check();rows=read();assert len(rows)==len(list(cases(p)))
    cfg=p['settings'];totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS};groups=[];samples={m:[] for m in METHODS}
    for gi,g in enumerate(p['regimes']):
        group=dict(regime=g['name'],methods={})
        for method in METHODS:
            cal=[next(v for v in r['samples'] if v['method']==method)['score'] for r in rows if r['case']['regime']==gi and r['case']['group']=='calibration_null'];assert len(cal)==199
            summary=dict(correct=0,wrong=0,false=0,positive=0,null=0)
            for row in rows:
                if row['case']['regime']!=gi or row['case']['group']=='calibration_null':continue
                r=next(v for v in row['samples'] if v['method']==method);samples[method].append(r['seconds'])
                pv=(1+sum(z>=r['score'] for z in cal))/(len(cal)+1);declared=pv<=cfg['alpha'];positive=row['case']['group']=='test_positive'
                dist=lambda x,y:abs((x-y+cfg['bins']/2)%cfg['bins']-cfg['bins']/2)
                correct=positive and declared and dist(r['primary'],row['truth']['primary'])<=2 and dist(r['delay'],row['truth']['secondary'])<=2
                summary['positive']+=int(positive);summary['null']+=int(not positive);summary['correct']+=int(correct);summary['wrong']+=int(positive and declared and not correct);summary['false']+=int(not positive and declared)
            for n,v in summary.items():totals[method][n]+=v
            group['methods'][method]=dict(**summary,calibration_second_largest_threshold=float(sorted(cal)[-2]))
        groups.append(group)
    medians={m:float(np.median(v)) for m,v in samples.items()};x=totals['duration_efficient'];b=totals['rounded_ded'];gl=totals['duration_glrt']
    rule=dict(more_correct_than_rounded=x['correct']>b['correct'],no_more_wrong_than_rounded=x['wrong']<=b['wrong'],no_more_null_than_rounded=x['false']<=b['false'],retains_ninety_percent_glrt=x['correct']>=.9*gl['correct'],no_more_wrong_than_glrt=x['wrong']<=gl['wrong'],no_more_null_than_glrt=x['false']<=gl['false'],twice_faster_than_glrt=medians['duration_glrt']>=2*medians['duration_efficient'])
    result=dict(status='DEVELOPMENT_ANALYSIS_ONLY',totals=totals,per_regime=groups,median_method_pipeline_seconds=medians,glrt_over_efficient_median_ratio=medians['duration_glrt']/medians['duration_efficient'],rule=rule,continue_to_fresh_confirmation=all(rule.values()),calibration_scope=p['calibration'],scope=p['scope'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
