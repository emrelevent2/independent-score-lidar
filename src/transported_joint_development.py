"""Fresh two-return/context-transport development with matched strong controls.

Weighted profiled likelihood aggregation is a classical heuristic statistic;
the adaptive local branch is NOT a proper likelihood mixture/e-value.
Full event-path plug-in bootstrap calibration is approximate under the null.
"""
from pathlib import Path
import sys,json,time,hashlib,sqlite3,platform
import numpy as np
import scipy,numba,flint
from flint import ctx
import context_agreement_development as base
from context_agreement_features import numeric_statistics,donor_features
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import fine_scan,refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from outward_safety_lag import outward_safe_lag
from fast_two_return_fit import reports,objective
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'TRANSPORTED_JOINT_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/transported_joint_development_v1'
METHODS=['local_score','local_joint','context_max','context_transport']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    old=base.check();p={k:old[k] for k in ['settings','regimes','families']};p['settings']=dict(p['settings'],positive_per_family_regime=8,null_per_family_regime=16,seed_base=9208741361,bootstrap_replicates=199)
    names=list(old['sources'])+['fast_two_return_fit.py','test_fast_two_return_fit.py','transported_joint_development.py']
    p.update(status='FROZEN_BEFORE_FRESH_TRANSPORTED_JOINT_DATA',sources={n:sha(ROOT/'src'/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__),methods=METHODS,base_physical_protocol_sha256=sha(base.P),primary_method='context_transport',
             design='Fresh320queries:128positive192null;four regimes and same seven broad coherent/isolated/misleading/primary-jump/clean/borrow families as prior broader development. Same heterogeneous physical parameter ranges;new independent seeds.16independent donor paths per query. Development,not confirmation of the old pilot candidate.',
             statistic='Fit unknown query a,tau,B under null. Six separated quarter-bin efficient-score peaks seed full five-parameter two-return MLEs. Local_joint=max of these6likelihoods. If independently fitted donor two-return score>20,add absolute donor-secondary start and query-primary+(donor-secondary-donor-primary) start;all query nuisances freely refit in every branch. Context_max=maximum over same8fitted components. Proposed context_transport=2*(logsumexp([local_max_loglik+log(.5),absolute_loglik+log(.25),relative_loglik+log(.25)])-null_loglik). Report locations of strongest raw-likelihood component. If donor score<=20,all joint methods EXACTLY equal local_joint. Repeated candidate-selection branch makes this a heuristic weighted-profile statistic,NOT proper-mixture evidence or a new statistical theorem.',
             numerical='Stable erfc integrated Gaussian bin masses and analytic gradient in optional Numba kernel,no fastmath;matches independent NumPy objective/optimizer in tests. Five wrapped images as fixed reference,not infinite-tail certificate. Eight numerical local fits,not global maximum;amplitude caps20,Bcap30 and +/-3sigma centre refinement as common established control. Alternative includes fitted null as fallback. Record optimizer failures and preselected numerical diagnostics.',
             calibration='199sharedcomplete latent-SPAD/filtered-electronics paths at full query-null MLE. Every draw recomputes outward-lag exact-dyadic risk durations,full three-parameter null refit,all6query-adaptive candidate starts and allsupported context fits/aggregation. Donors fixed independent of query path. No iidPoissoncounts conditional on random exposure. Composite-null plug-in Monte Carlo p-values approximate;no finiteuniformguarantee. Main alpha=.01. Allmethods also report .005/.02descriptive operating points;do not select a winning threshold from this cohort.',
             primary_rule='At alpha=.01,context_transport strictly more correct AND strictly fewer total errors than strong local_joint,with no more wrong and no more null separately. At least local_score correct,no more local_score wrong/null. Also Pareto improvement over matched equal-information context_max(at least correct,no more wrong/null,and at least one strict improvement). Any development pass requires fresh larger confirmation with uncertainty and real-data assessment;no posthoc promotion of ablation or different alpha.',
             scope='Established joint MLE/weighted profiles/context transport ingredients. Potential contribution only their exposure-aware construction and measured utility if demonstrated. Known Gaussian width/recovery,initial active state,bin-constant intensity. No physical response/recovery uncertainty validation,new theory,novelty,Q1 or submission readiness inferred.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());ctx.prec=128
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__)
    assert sha(base.P)==p['base_physical_protocol_sha256'];return p


def cases(p):
    for case in base.cases(p):
        case['bootstrap_seed']=51000000000+1000003*case['index'];yield case


def statistics(s,e,fit,context,cfg):
    start=time.perf_counter();local=fine_scan(s,e,cfg['sigma'],fit=fit);r=reports(s,e,cfg['sigma'],fit,context);r['samples']['local_score']={k:local[k] for k in ['score','primary','delay']};r['seconds']=time.perf_counter()-start;return r


def worker(case,p):
    start=time.perf_counter();cfg=p['settings'];g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=base.data(case,p);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],cfg['bins'],lag);context=donor_features(ds,de,cfg['sigma']);fit=fit_null(s,e,cfg['sigma']);observed=statistics(s,e,fit,context,cfg);boots={m:[] for m in METHODS};pathhash=hashlib.sha256();failure_indices=[];diagnostics=[];alt_failures=0;alt_fits=0
    rates=fit['theta'][0]*pulse(fit['theta'][1],cfg['bins'],cfg['sigma'])[0]+fit['theta'][2]/cfg['bins']
    selected=case['id'].endswith('_000')
    for j in range(cfg['bootstrap_replicates']):
        bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*j);pathhash.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],cfg['bins'],lag);bf=refit_from_start(bs,be,fit['theta'],cfg['sigma'],'duration');br=statistics(bs,be,bf,context,cfg)
        if not bf['optimizer_success']:failure_indices.append(j)
        alt_failures+=sum(not f['optimizer_success'] for f in br['fits']);alt_fits+=len(br['fits'])
        for m in METHODS:boots[m].append(br['samples'][m]['score'])
        if selected and j in [0,198]:diagnostics.append(dict(draw=j,null_theta=bf['theta'].tolist(),null_loglik=bf['loglik'],fits=br['fits']))
    samples={m:dict(observed['samples'][m],pvalue=pvalue(observed['samples'][m]['score'],boots[m]),bootstrap_scores=boots[m]) for m in METHODS}
    if not observed['context_supported']:assert samples['local_joint']==samples['context_max']==samples['context_transport']
    return dict(case=case,truth=truth,generating_parameters=params,data_sha256=dh,observed_events=len(obs),scored_events=int(s.sum()),context={k:context[k] for k in ['primary','secondary','two_pulse_loglr']},null_fit=fit['theta'].tolist(),observed_fits=observed['fits'],samples=samples,bootstrap_path_sha256=pathhash.hexdigest(),bootstrap_null_failure_indices=failure_indices,bootstrap_alternative_fits=alt_fits,bootstrap_alternative_failures=alt_failures,refit_diagnostics=diagnostics,seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(z[0]) for z in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();objective(np.array([.8,15.3,.3,.02,27.3]),np.ones(64),np.ones(64),1.5);c=connect();done={z[0] for z in c.execute('SELECT id FROM records')};new=0;start=time.perf_counter()
    for case in cases(p):
        if case['id'] in done:continue
        row=worker(case,p);c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(row)));c.commit();new+=1
        if (len(done)+new)%8==0:print('Transported joint development',len(done)+new,'/320 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==320;r=dict(status='COMPLETE',cases=320,newly_computed=new,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),seconds=time.perf_counter()-start);(OUT/'completion.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)


def endpoint(row,method,alpha=.01):
    z=row['samples'][method];positive=row['case']['positive'];declare=z['pvalue']<=alpha;distance=lambda a,b:abs((a-b+32)%64-32);correct=positive and declare and distance(z['primary'],row['truth']['primary'])<=2 and distance(z['delay'],row['truth']['secondary'])<=2
    return dict(correct=int(correct),wrong=int(positive and declare and not correct),false=int(not positive and declare),positive=int(positive),null=int(not positive))


def totals(rows,alpha):
    total={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
    for row in rows:
        for m in METHODS:
            for k,n in endpoint(row,m,alpha).items():total[m][k]+=n
    return total


def analyze():
    p=check();rows=read();assert len(rows)==320;t=totals(rows,.01);x,l,s,b=[t[m] for m in ['context_transport','local_joint','local_score','context_max']];rule=dict(strict_more_correct_than_joint=x['correct']>l['correct'],strict_fewer_errors_than_joint=x['wrong']+x['false']<l['wrong']+l['false'],no_more_joint_wrong=x['wrong']<=l['wrong'],no_more_joint_null=x['false']<=l['false'],at_least_score_correct=x['correct']>=s['correct'],no_more_score_wrong=x['wrong']<=s['wrong'],no_more_score_null=x['false']<=s['false'],at_least_context_max_correct=x['correct']>=b['correct'],no_more_context_max_wrong=x['wrong']<=b['wrong'],no_more_context_max_null=x['false']<=b['false'],at_least_one_strict_context_max=x['correct']>b['correct'] or x['wrong']<b['wrong'] or x['false']<b['false'])
    groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:groups.append(dict(regime=g['name'],family=family,methods=totals([r for r in rows if r['case']['regime']==gi and r['case']['family']==family],.01)))
    r=dict(status='FRESH_TRANSPORTED_JOINT_DEVELOPMENT_ONLY',totals=t,primary_rule=rule,continue_to_fresh_confirmation=all(rule.values()),descriptive_operating_points={str(alpha):totals(rows,alpha) for alpha in [.005,.01,.02]},per_regime_family=groups,null_fit_failures=sum(len(z['bootstrap_null_failure_indices']) for z in rows),alternative_fits=sum(z['bootstrap_alternative_fits'] for z in rows),alternative_fit_failures=sum(z['bootstrap_alternative_failures'] for z in rows),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='per_regime_family'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
