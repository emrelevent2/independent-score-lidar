"""Fresh broader development;joint event-path calibration repeats the pilot."""
from pathlib import Path
import sys,json,time,hashlib,sqlite3,platform
import numpy as np
import scipy,numba,flint
from flint import ctx
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import refit_from_start,fine_scan,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from outward_safety_lag import outward_safe_lag
from context_agreement_features import numeric_statistics,donor_features,pilot_location
from context_agreement_score import context_agreement_score
from safe_prior_score import prior_score
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'CONTEXT_AGREEMENT_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/context_agreement_development_v1'
METHODS=['local','spatial_prior','full_supported','pilot_supported']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    names=['context_agreement_development.py','context_agreement_features.py','context_agreement_score.py','test_context_agreement_features.py','test_context_agreement_score.py','outward_safety_lag.py','certified_safe_components.py','safe_deadtime_exposure.py','deadtime_path_bootstrap.py','deadtime_scan.py','deadtime_exposure.py','safe_prior_score.py']
    p=dict(status='FROZEN_BEFORE_NEW_BROADER_CONTEXT_DATA',sources={n:sha(ROOT/'src'/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__),methods=METHODS,
           settings=dict(bins=64,sigma=1.5,donors=16,positive_per_family_regime=16,null_per_family_regime=32,bootstrap_replicates=199,seed_base=7139462187,primary_phase=15.3,base_phase_jitter=[-.4,.4],query_phase_offset=[-.75,.75],query_amplitude_scale=[.7,1.3],query_background_scale=[.75,1.25],donor_phase_offset=[-.15,.15],donor_amplitude_scale=[.9,1.1],donor_background_scale=[.9,1.1],gap_jitter=[-.35,.35],jump_magnitude=4.,jump_jitter=[-.3,.3],pilot_time_fraction=.25,alpha=.01,tolerance=2),
           regimes=[dict(name='moderate',a=.8,b=.3,c=.065,gap=12,cycles=1500,td=83.7,te=40.),dict(name='near',a=.8,b=1.,c=.09,gap=6,cycles=1500,td=126.7,te=80.),dict(name='bright_hidden',a=2.4,b=.8,c=.15,gap=10,cycles=1000,td=31.7,te=80.),dict(name='weak_hidden',a=.25,b=.1,c=.035,gap=20,cycles=1500,td=83.7,te=100.)],families=dict(positive=['coherent','isolated','misleading','primary_jump_positive'],null=['clean_null','borrow_null','primary_jump_null']),
           design='Fresh640queries:256positive384null with16independent context paths each. Query primary phase/amplitude/background differ from context;donors also vary slightly. Jump cases shift query primary and secondary together byabout4bins,keeping true separation. Secondary separation has fractional jitter. Misleading donors shift secondary10bins;isolated donors have none;borrow/jump-null donors have secondary but query does not. Methods never receive truth parameters beyond known sigma/recovery/horizon.',
           statistic='Primary candidate pilot_supported: local max projected score+2log(K/FWHM)*strength(Tdonor)*exp(-distance(local,donor)^2/(2sigma^2))*exp(-distance(pilot,donor)^2/(8sigma^2));strength=-expm1(-max(Tdonor-20,0)/20). Pilot firstquarter only,one damped Fisher step from independent donor primary initialization. Full_supported omits pilot factor,ablation only. No tuning these heuristics after seeing outputs.',
           controls='Local fine score and classical weighted spatial prior score at SAME donor and coverage locations. All methods report query-fitted primary;local/full/pilot report same local secondary. Spatial prior reports its selected secondary. Query-fitted primary corrects the earlier flat-patch reporting convention fairly for new jumps.',
           calibration='199sharedfulltwo-componenteventpaths at observed unrestricted single-return null fit. Every replicate reconstructs exact-dyadic safe statistics at outward lag,fully refits all3querynullparameters,and recomputes earlypilot from only its firstquarter plus fixedcontext. Both adaptive scores and controls recalibrated jointly. No iidPoissoncounts conditionalonrandomrisk. Plug-in composite-null p-values approximate;NOTe-values or finiteuniformguarantees.',
           primary_rule='Predeclared pilot_supported strictly more correct AND strictly fewer total errors(wrong positive+null) than local,with no more wrong and no more null separately. Additionally at least as many correct and no more wrong/null than equal-context spatial_prior,with at least one strict improvement. Full_supported is an ablation,not a substitute posthoc winner.',
           scope='New DEVELOPMENT,not independent confirmation of any old effect. No primary-sharing restriction on query null. Known pulse width and nonparalyzable recovery;IRF/jitter/recovery uncertainty unresolved. Heuristic context agreement and joint bootstrap are established ingredients,not new general theory. Continue to fresh confirmation only if predeclared primary criteria and integrity checks pass.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());ctx.prec=128
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__)
    return p


def cases(p):
    c=p['settings'];i=0
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:
            positive=family in p['families']['positive'];num=c['positive_per_family_regime'] if positive else c['null_per_family_regime']
            for j in range(num):
                yield dict(id=f'{gi}_{family}_{j:03}',index=i,regime=gi,family=family,positive=positive,seed=c['seed_base']+1009*i,bootstrap_seed=31000000000+1000003*i)
                i+=1


def data(case,p):
    c=p['settings'];g=p['regimes'][case['regime']];rng=np.random.default_rng(case['seed']+29);base=c['primary_phase']+rng.uniform(*c['base_phase_jitter']);gap=g['gap']+rng.uniform(*c['gap_jitter']);jump='primary_jump' in case['family'];offset=(rng.choice([-1.,1.])*c['jump_magnitude']+rng.uniform(*c['jump_jitter'])) if jump else rng.uniform(*c['query_phase_offset']);query_theta=[g['a']*rng.uniform(*c['query_amplitude_scale']),(base+offset)%c['bins'],g['b']*rng.uniform(*c['query_background_scale'])];qs=query_theta[1]+gap;ds=[];de=[];query=None;h=hashlib.sha256();params=[];lag=outward_safe_lag(g['td'],g['te'])
    for pixel in range(c['donors']+1):
        if pixel==0:a,t,b=query_theta;presence=case['positive'];second=(t+gap)%c['bins'];secondary=g['c']*a/g['a']
        else:
            a=g['a']*rng.uniform(*c['donor_amplitude_scale']);t=(base+rng.uniform(*c['donor_phase_offset']))%c['bins'];b=g['b']*rng.uniform(*c['donor_background_scale']);presence=case['family'] in ['coherent','misleading','borrow_null','primary_jump_positive','primary_jump_null'];second=(t+gap+(10 if case['family']=='misleading' else 0))%c['bins'];secondary=g['c']*a/g['a']
        rates=a*pulse(t,c['bins'],c['sigma'])[0]+b/c['bins']
        if presence:rates+=secondary*pulse(second,c['bins'],c['sigma'])[0]
        obs,latent=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['seed']+1000003*(pixel+1));h.update(obs.tobytes());h.update(latent.tobytes());params.append([float(a),float(t),float(b),float(secondary if presence else 0),float(second)])
        if pixel==0:query=obs
        else:s,e=numeric_statistics(obs,g['cycles'],c['bins'],lag);ds.append(s);de.append(e)
    return query,np.asarray(ds),np.asarray(de),dict(primary=float(query_theta[1]),secondary=float(qs%c['bins'])),params,h.hexdigest()


def statistic_set(s,e,fit,pilot,context,cfg):
    local=fine_scan(s,e,cfg['sigma'],fit=fit);prior=prior_score(s,e,fit['theta'],cfg['sigma'],context['prior_bank']);prior['primary']=float(fit['theta'][1]);scores=dict(local=float(local['score']),spatial_prior=float(prior['score']))
    for method,use_pilot in [('full_supported',False),('pilot_supported',True)]:scores[method]=float(context_agreement_score(local['score'],local['delay'],pilot,context['secondary'],context['two_pulse_loglr'],cfg['bins'],cfg['sigma'],use_pilot))
    locations={m:[float(local['primary']),float(local['delay'])] for m in METHODS};locations['spatial_prior']=[prior['primary'],prior['delay']]
    return scores,locations


def worker(case,p):
    start=time.perf_counter();cfg=p['settings'];g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=data(case,p);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],cfg['bins'],lag);context=donor_features(ds,de,cfg['sigma']);fit=fit_null(s,e,cfg['sigma']);pilot_cycles=int(g['cycles']*cfg['pilot_time_fraction']);pilot,count=pilot_location(obs,pilot_cycles,cfg['bins'],lag,cfg['sigma'],context);scores,locations=statistic_set(s,e,fit,pilot,context,cfg);boots={m:[] for m in METHODS};pilots=[];bh=hashlib.sha256();failed=0;diagnostics=[]
    rates=fit['theta'][0]*pulse(fit['theta'][1],cfg['bins'],cfg['sigma'])[0]+fit['theta'][2]/cfg['bins']
    for j in range(cfg['bootstrap_replicates']):
        bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*j);bh.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],cfg['bins'],lag);bf=refit_from_start(bs,be,fit['theta'],cfg['sigma'],'duration');failed+=int(not bf['optimizer_success']);bp,_=pilot_location(bo,pilot_cycles,cfg['bins'],lag,cfg['sigma'],context);v,_=statistic_set(bs,be,bf,bp,context,cfg);pilots.append(float(bp))
        for m in METHODS:boots[m].append(v[m])
        if case['id'] in {f'{i}_{f}_000' for i in range(4) for f in p['families']['positive']+p['families']['null']} and j in [0,cfg['bootstrap_replicates']-1]:diagnostics.append(dict(draw=j,theta=bf['theta'].tolist(),loglik=bf['loglik']))
    samples={m:dict(score=scores[m],primary=locations[m][0],delay=locations[m][1],pvalue=pvalue(scores[m],boots[m]),bootstrap_scores=boots[m]) for m in METHODS}
    if context['two_pulse_loglr']<=20:
        for m in ['full_supported','pilot_supported']:assert samples[m]==dict(samples['local'])
    return dict(case=case,truth=truth,generating_parameters=params,data_sha256=dh,observed_events=len(obs),scored_events=int(s.sum()),pilot_scored_events=count,pilot_delay=pilot,context=dict(primary=context['primary'],secondary=context['secondary'],two_pulse_loglr=context['two_pulse_loglr'],initial_theta=context['initial_theta'].tolist()),null_fit=fit['theta'].tolist(),bootstrap_path_sha256=bh.hexdigest(),bootstrap_pilot_delays=pilots,bootstrap_optimizer_failures=failed,refit_diagnostics=diagnostics,samples=samples,seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(v[0]) for v in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();simulate_two_dead_times(np.full(16,.01),2,8.,12.,14);c=connect();done={r[0] for r in c.execute('SELECT id FROM records')};start=time.perf_counter();new=0;total=len(list(cases(p)))
    for case in cases(p):
        if case['id'] in done:continue
        row=worker(case,p);c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(row)));c.commit();new+=1
        if (len(done)+new)%8==0:print('Broader context/pilot development',len(done)+new,'/',total,'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==total;v=dict(status='COMPLETE',cases=total,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),newly_computed_this_invocation=new,seconds_this_invocation=time.perf_counter()-start);(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)


def endpoint(r,m,p):
    v=r['samples'][m];declare=v['pvalue']<=p['settings']['alpha'];pos=r['case']['positive'];k=p['settings']['bins'];d=lambda x,y:abs((x-y+k/2)%k-k/2);correct=pos and declare and d(v['primary'],r['truth']['primary'])<=2 and d(v['delay'],r['truth']['secondary'])<=2
    return dict(correct=int(correct),wrong=int(pos and declare and not correct),false=int(not pos and declare),positive=int(pos),null=int(not pos))


def analyze():
    p=check();rows=read();assert len(rows)==len(list(cases(p)));totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS};groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:
            ep={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
            for r in rows:
                if r['case']['regime']!=gi or r['case']['family']!=family:continue
                for m in METHODS:
                    for k,z in endpoint(r,m,p).items():ep[m][k]+=z;totals[m][k]+=z
            groups.append(dict(regime=g['name'],family=family,methods=ep))
    x,l,b=[totals[m] for m in ['pilot_supported','local','spatial_prior']];rule=dict(strict_more_correct_than_local=x['correct']>l['correct'],strict_fewer_errors_than_local=x['wrong']+x['false']<l['wrong']+l['false'],no_more_wrong_than_local=x['wrong']<=l['wrong'],no_more_null_than_local=x['false']<=l['false'],at_least_spatial_correct=x['correct']>=b['correct'],no_more_spatial_wrong=x['wrong']<=b['wrong'],no_more_spatial_null=x['false']<=b['false'],at_least_one_strict_spatial=bool(x['correct']>b['correct'] or x['wrong']<b['wrong'] or x['false']<b['false']))
    r=dict(status='BROADER_DEVELOPMENT_ONLY',totals=totals,per_regime_family=groups,primary_rule=rule,continue_to_fresh_confirmation=all(rule.values()),zero_context_support_cases=sum(z['context']['two_pulse_loglr']<=20 for z in rows),bootstrap_optimizer_failures=sum(z['bootstrap_optimizer_failures'] for z in rows),scope=p['scope'],calibration_scope=p['calibration']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='per_regime_family'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
