"""Frozen unknown-flux calibration development;not a finite-sample guarantee."""
from pathlib import Path
import json,sys,time,hashlib,sqlite3,platform
import numpy as np
import scipy,numba
from deadtime_exposure import sufficient_statistics
from deadtime_scan import fit_null,pulse,rounded_opportunities
from deadtime_path_bootstrap import simulate,fine_scan,bootstrap_paths,fixed_fit_scores,one_step,refit_from_start,pvalue
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'DEADTIME_BOOTSTRAP_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/deadtime_bootstrap_development_v1'
METHODS=['duration_projected','duration_onestep','duration_refit','rounded_refit']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol exists')
    names=['deadtime_bootstrap_development.py','deadtime_path_bootstrap.py','test_deadtime_path_bootstrap.py','deadtime_scan.py','deadtime_exposure.py']
    regimes=[dict(name='moderate_long',a=.8,b=.3,cycles=1500,dead=83.7,secondary=.065,gap=12),dict(name='near_double_period',a=.8,b=1.,cycles=1500,dead=126.7,secondary=.09,gap=6),dict(name='high_flux_short',a=2.4,b=.8,cycles=1000,dead=31.7,secondary=.15,gap=10),dict(name='low_flux_long',a=.25,b=.1,cycles=1500,dead=83.7,secondary=.035,gap=20)]
    p=dict(status='FROZEN_BEFORE_NEW_UNKNOWN_NUISANCE_DATA',sources={n:sha(ROOT/'src'/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),methods=METHODS,regimes=regimes,
           settings=dict(bins=64,sigma=1.5,primary_phase=15.3,phase_jitter_uniform=[-.4,.4],seed_base=3748261731,null_per_regime=64,positive_per_regime=32,bootstrap_replicates=199,alpha=.01,tolerance=2,subdivision=4,initial_state='active,known'),
           calibration='Each observed path fits a,tau,B from its own counts/exposure,or rounded opportunities. No oracle flux,delay or secondary truth enters bootstrap. Known sigma and dead time assumed. Simulate continuous nonparalyzable event paths at fitted null;do not conditionally simulate independent Poisson counts given E. Plug-in p=(1+sum(Tboot>=Tobs))/200 is approximate for composite null;no finite-sample or uniform claim.',
           refit_scope='Three-parameter numerical local MLE at each bootstrap,initialized at observed generating fit;not certified global. Source-style three-start fits audit first/last bootstrap draw for fixed selected cases,without replacing the original fits. Any material likelihood discrepancy is a control-validity warning,not silently ignored.',
           timing='JIT warmup excluded. Observed fit+scan and generated-path stages shared and measured separately;per-method seconds are sums of required observed+path+calibration stages,not separately measured whole pipelines. Timings not isolated,cold or FPGA claims.',
           primary_rule='Exploratory: projected method strictly more correct than quarter-bin rounded_refit AND strictly fewer total errors(wrong positive+null). No more wrong/null separately. Also retain >=95% correct of duration_refit with no more wrong/null. If fail,no fresh efficacy confirmation. No retuning alpha or per-case location selection.',
           scope='New DEVELOPMENT on a cross-cycle physical simulator,not old first-event v5 confirmation. Four-times finer scan is applied to every score control. No independent real multi-surface ground truth,novelty or Q1 readiness established.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text())
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__)
    return p


def cases(p):
    i=0
    for gi,g in enumerate(p['regimes']):
        for group,num in [('null',p['settings']['null_per_regime']),('positive',p['settings']['positive_per_regime'])]:
            for j in range(num):
                yield dict(id=f'{gi}_{group}_{j:03}',index=i,regime=gi,group=group,seed=p['settings']['seed_base']+i*1009,bootstrap_seed=98137001+i*1000003)
                i+=1


def data(case,p):
    cfg=p['settings'];g=p['regimes'][case['regime']];k=cfg['bins'];sig=cfg['sigma'];tau=cfg['primary_phase']+np.random.default_rng(case['seed']).uniform(*cfg['phase_jitter_uniform'])
    rates=g['a']*pulse(tau,k,sig)[0]+g['b']/k
    if case['group']=='positive':rates+=g['secondary']*pulse((tau+g['gap'])%k,k,sig)[0]
    events=simulate(rates,g['cycles'],g['dead'],case['seed']+1);s,e=sufficient_statistics(events,g['cycles'],k,g['dead']);a=rounded_opportunities(events,g['cycles'],k,g['dead'])
    h=hashlib.sha256()
    for v in [events,s,e,a]:h.update(v.tobytes())
    return events,s,e,a,dict(primary=float(tau),secondary=float((tau+g['gap'])%k)),h.hexdigest()


def worker(case,p):
    cfg=p['settings'];g=p['regimes'][case['regime']];events,s,e,a,truth,dh=data(case,p);out={};fits={};path_hashes={};bootfits=[];diag=[]
    for mode,exp in [('duration',e),('ded',a)]:
        start=time.perf_counter();fit=fit_null(s,exp,cfg['sigma'],mode);obs=fine_scan(s,exp,cfg['sigma'],mode,fit,subdivision=cfg['subdivision']);observed_seconds=time.perf_counter()-start;fits[mode]=fit['theta'].tolist()
        start=time.perf_counter();ss,ee,parts=bootstrap_paths(fit['theta'],cfg['sigma'],g['cycles'],g['dead'],cfg['bins'],cfg['bootstrap_replicates'],case['bootstrap_seed'],mode);path_seconds=time.perf_counter()-start
        h=hashlib.sha256()
        for v in parts:h.update(v)
        path_hashes[mode]=h.hexdigest();base=observed_seconds+path_seconds
        variants=['projected','onestep','refit'] if mode=='duration' else ['refit']
        for variant in variants:
            start=time.perf_counter();success=[]
            if variant=='projected':scores,_=fixed_fit_scores(ss,ee,fit['theta'],cfg['sigma'],mode,subdivision=cfg['subdivision'])
            else:
                scores=[]
                for j in range(len(ss)):
                    boot=(one_step if variant=='onestep' else refit_from_start)(ss[j],ee[j],fit['theta'],cfg['sigma'],mode)
                    r=fine_scan(ss[j],ee[j],cfg['sigma'],mode,boot,subdivision=cfg['subdivision']);scores.append(r['score'])
                    if variant=='refit':
                        success.append(bool(boot['optimizer_success']))
                        if case['id'] in {f'{gi}_{group}_{n:03}' for gi in range(4) for group,n in [('null',0),('null',63),('positive',0),('positive',31)]} and j in [0,cfg['bootstrap_replicates']-1]:
                            # Store for separate three-start audit AFTER timed run.
                            diag.append(dict(mode=mode,draw=j,theta=boot['theta'].tolist(),loglik=boot['loglik']))
            seconds=base+time.perf_counter()-start;name=f'{mode}_{variant}' if mode=='duration' else 'rounded_refit';scores=np.asarray(scores,dtype=float)
            out[name]=dict(score=obs['score'],primary=obs['primary'],delay=obs['delay'],pvalue=pvalue(obs['score'],scores),bootstrap_scores=scores.tolist(),seconds=seconds,stage_seconds=dict(observed=observed_seconds,paths=path_seconds),bootstrap_optimizer_failures=int(len(success)-sum(success)))
    return dict(case=case,truth=truth,data_sha256=dh,events=len(events),fitted_nulls=fits,bootstrap_path_hashes=path_hashes,samples=out,refit_diagnostics=diag)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(v[0]) for v in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();simulate(np.full(64,.01),2,83.7,1);c=connect();done={v[0] for v in c.execute('SELECT id FROM records')};start=time.perf_counter();new=0;total=len(list(cases(p)))
    for case in cases(p):
        if case['id'] in done:continue
        row=worker(case,p);c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(row)));c.commit();new+=1
        if (len(done)+new)%8==0:print('Unknown-nuisance bootstrap development',len(done)+new,'/',total,'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==total
    v=dict(status='COMPLETE',cases=total,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),newly_computed_this_invocation=new,seconds_this_invocation=time.perf_counter()-start)
    (OUT/'completion.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)


def endpoint(row,method,p):
    v=row['samples'][method];declared=v['pvalue']<=p['settings']['alpha'];positive=row['case']['group']=='positive';k=p['settings']['bins'];tol=p['settings']['tolerance']
    dist=lambda x,y:abs((x-y+k/2)%k-k/2)
    correct=positive and declared and dist(v['primary'],row['truth']['primary'])<=tol and dist(v['delay'],row['truth']['secondary'])<=tol
    return dict(correct=int(correct),wrong=int(positive and declared and not correct),false=int(not positive and declared),positive=int(positive),null=int(not positive))


def analyze():
    p=check();rows=read();assert len(rows)==len(list(cases(p)));totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS};groups=[]
    for gi,g in enumerate(p['regimes']):
        per={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
        for row in rows:
            if row['case']['regime']!=gi:continue
            for m in METHODS:
                for k,v in endpoint(row,m,p).items():per[m][k]+=v;totals[m][k]+=v
        groups.append(dict(regime=g['name'],methods=per))
    times={m:float(np.median([r['samples'][m]['seconds'] for r in rows])) for m in METHODS};x,b,z=[totals[m] for m in ['duration_projected','rounded_refit','duration_refit']]
    rule=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_total_errors=x['wrong']+x['false']<b['wrong']+b['false'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'],retains_ninetyfive_percent_refit=x['correct']>=.95*z['correct'],no_more_wrong_than_refit=x['wrong']<=z['wrong'],no_more_null_than_refit=x['false']<=z['false'])
    result=dict(status='DEVELOPMENT_ONLY_UNKNOWN_NUISANCE',totals=totals,per_regime=groups,median_composed_processing_seconds=times,refit_over_projected_composed_median_ratio=times['duration_refit']/times['duration_projected'],bootstrap_optimizer_failures={m:sum(r['samples'][m]['bootstrap_optimizer_failures'] for r in rows) for m in METHODS},rule=rule,continue_to_fresh_efficacy_confirmation=all(rule.values()),scope=p['scope'],calibration_scope=p['calibration'],timing_scope=p['timing'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
