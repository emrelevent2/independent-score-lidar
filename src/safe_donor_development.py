"""Frozen heterogeneous context study with latent detector/electronics recovery."""
from pathlib import Path
import sys,json,time,hashlib,sqlite3,platform
import numpy as np
import scipy,numba,flint
from flint import ctx
from safe_deadtime_exposure import simulate_two_dead_times,safe_statistics,safe_lag
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import refit_from_start,fine_scan,pvalue
from safe_donor_alternative import build_alternative,mixture_scores
from certified_safe_components import exact_safe_statistics,mixture_log_lower
from certified_safe_null import declare_from_fixed_mixture
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'SAFE_DONOR_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/safe_donor_development_v1'
METHODS=['local_bootstrap','donor_bootstrap','donor_certified']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    names=['safe_donor_development.py','safe_donor_alternative.py','certified_safe_components.py','certified_safe_null.py','continuous_rate_profile.py','safe_deadtime_exposure.py','deadtime_path_bootstrap.py','deadtime_exposure.py','deadtime_scan.py']
    p=dict(status='FROZEN_BEFORE_HETEROGENEOUS_TWO_DEAD_TIME_CONTEXT_DATA',sources={n:sha(ROOT/'src'/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__),methods=METHODS,
           settings=dict(bins=64,sigma=1.5,cycles_default=1500,donors=16,positive_per_family_regime=16,null_per_family_regime=32,bootstrap_replicates=199,seed_base=4129361691,primary_phase=15.3,phase_jitter_uniform=[-.4,.4],alpha=.01,tolerance=2,cover_initial_cells=8,cover_max_boxes=192),
           regimes=[dict(name='moderate',a=.8,b=.3,c=.065,gap=12,cycles=1500,td=83.7,te=40.),dict(name='near',a=.8,b=1.,c=.09,gap=6,cycles=1500,td=126.7,te=80.),dict(name='bright_hidden',a=2.4,b=.8,c=.15,gap=10,cycles=1000,td=31.7,te=80.),dict(name='weak_hidden',a=.25,b=.1,c=.035,gap=20,cycles=1500,td=83.7,te=100.)],
           families=dict(positive=['coherent','isolated','misleading'],null=['clean_null','borrow_null']),
           design='Within each case all17pixels share a fixed primary delay/amplitude/background;photon noise independent by pixel. Coherent neighbors share secondary;isolated neighbors have no secondary;misleading neighbors have secondary10bins displaced. Borrow-null query has no secondary but all donors do. Same query supplied to every method. Local control uses query only;both donor methods use identical16extra pixels. No assertion of information-budget equality with local control or optimality versus every spatial method.',
           alternative='Donor-only pooled two-pulse fit;3/4mass on4spatial amplitude factors,1/4on all admissible integer relative offsets times3weak amplitudes. Proper original weights never renormalized or chosen with query. Query chooses MAP component for reported location only. Mixture/borrowing/inference principles are established,not new general ideas.',
           calibration='Shared199latent-two-detector bootstrap paths at observed fitted single-return null;fully warm-start refit all3nullparameters per path. Local statistic=fine projected scan;donor statistic=fixed proper mixture loglik minus numerical null maximum. Both p-values approximate composite-null plug-in tests,not exact1%guarantees. Certified donor uses Arb lower numerator and complete continuous-delay/unbounded-amplitude null upper at1/100;known sigma/D and idealized bin-constant intensity required.',
           primary_rule='Exploratory practical donor_bootstrap strictly more correct AND strictly fewer total errors(wrong positive+null) than local_bootstrap,with no more wrong or null separately. Report certified method separately against ALL controls and every family;do not demand/claim uniform superiority. No new efficacy confirmation if primary fails.',
           scope='New DEVELOPMENT.192positive/256null observations,16contextpixels each. Not prior v5/oneD confirmation. No physical quantization/jitter/afterpulsing/recovery uncertainty guarantee. Same model-safe interval restriction for controls;not a full latent-state likelihood or claim of optimal information use. Proper null/tangent/censoring ideas are prior art.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());ctx.prec=128
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    assert p['environment']==dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,flint=flint.__version__)
    return p


def cases(p):
    i=0;c=p['settings']
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:
            positive=family in p['families']['positive'];num=c['positive_per_family_regime'] if positive else c['null_per_family_regime']
            for j in range(num):
                yield dict(id=f'{gi}_{family}_{j:03}',index=i,regime=gi,family=family,positive=positive,seed=c['seed_base']+1009*i,bootstrap_seed=19000000000+1000003*i)
                i+=1


def data(case,p):
    cfg=p['settings'];g=p['regimes'][case['regime']];k=cfg['bins'];sig=cfg['sigma'];tau=cfg['primary_phase']+np.random.default_rng(case['seed']).uniform(*cfg['phase_jitter_uniform']);second=(tau+g['gap'])%k;lag=safe_lag(g['td'],g['te']);h=hashlib.sha256();donor_s=[];donor_e=[];query=None
    for pixel in range(cfg['donors']+1):
        presence=case['positive'] if pixel==0 else case['family'] in ['coherent','misleading','borrow_null'];delay=(second+10)%k if pixel>0 and case['family']=='misleading' else second
        rates=g['a']*pulse(tau,k,sig)[0]+g['b']/k
        if presence:rates+=g['c']*pulse(delay,k,sig)[0]
        obs,latent=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['seed']+1000003*(pixel+1));h.update(obs.tobytes());h.update(latent.tobytes())
        if pixel==0:query=obs
        else:
            s,e,_=safe_statistics(obs,g['cycles'],k,lag);donor_s.append(s);donor_e.append(e)
    return query,np.asarray(donor_s),np.asarray(donor_e),dict(primary=float(tau),secondary=float(second)),h.hexdigest()


def worker(case,p):
    cfg=p['settings'];g=p['regimes'][case['regime']];start=time.perf_counter();obs,ds,de,truth,dh=data(case,p);lag=safe_lag(g['td'],g['te']);stats=exact_safe_statistics(obs,g['cycles'],cfg['bins'],lag);s=stats['counts'];e=np.array([float(v.mid()) for v in stats['exposure']])
    alternative=build_alternative(ds,de,cfg['sigma']);num,locations,idx=mixture_scores(s,e,alternative);numlo=mixture_log_lower(stats,alternative['rates'],alternative['weights']);assert numlo<=num[0]+1e-9
    fit=fit_null(s,e,cfg['sigma']);local=fine_scan(s,e,cfg['sigma'],fit=fit);donor_score=float(num[0]-fit['loglik']);bootstrap_local=[];ss=[];ee=[];ll=[];bh=hashlib.sha256();failed=0;diagnostics=[]
    rates=fit['theta'][0]*pulse(fit['theta'][1],cfg['bins'],cfg['sigma'])[0]+fit['theta'][2]/cfg['bins']
    for j in range(cfg['bootstrap_replicates']):
        bobs,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*j);bs,be,_=safe_statistics(bobs,g['cycles'],cfg['bins'],lag);bh.update(bobs.tobytes());ss.append(bs);ee.append(be)
        bf=refit_from_start(bs,be,fit['theta'],cfg['sigma'],'duration');failed+=int(not bf['optimizer_success']);bl=fine_scan(bs,be,cfg['sigma'],fit=bf);bootstrap_local.append(bl['score']);ll.append(bf['loglik'])
        if case['id'] in {f'{i}_{f}_000' for i in range(4) for f in p['families']['positive']+p['families']['null']} and j in [0,cfg['bootstrap_replicates']-1]:diagnostics.append(dict(draw=j,theta=bf['theta'].tolist(),loglik=bf['loglik']))
    bn,_,_=mixture_scores(ss,ee,alternative);bootstrap_donor=bn-np.asarray(ll)
    certified=declare_from_fixed_mixture(stats,numlo,sigma_range=(cfg['sigma'],cfg['sigma']),initial_cells=cfg['cover_initial_cells'],max_boxes=cfg['cover_max_boxes'],fit=fit)
    pos=locations[0];samples=dict(local_bootstrap=dict(pvalue=pvalue(local['score'],bootstrap_local),score=local['score'],primary=local['primary'],delay=local['delay'],bootstrap_scores=list(map(float,bootstrap_local))),donor_bootstrap=dict(pvalue=pvalue(donor_score,bootstrap_donor),score=donor_score,primary=float(pos[0]),delay=float(pos[1]),bootstrap_scores=bootstrap_donor.tolist()),donor_certified=dict(declared=certified['declared'],primary=float(pos[0]),delay=float(pos[1]),certificate=certified))
    return dict(case=case,truth=truth,data_sha256=dh,alternative_sha256=alternative['sha256'],alternative_components=len(alternative['rates']),selected_kind=alternative['kinds'][int(idx[0])],selected_component=int(idx[0]),observed_events=len(obs),scored_events=int(s.sum()),bootstrap_path_sha256=bh.hexdigest(),null_fit=fit['theta'].tolist(),bootstrap_optimizer_failures=failed,refit_diagnostics=diagnostics,samples=samples,seconds_including_data_and_all_methods=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(v[0]) for v in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();simulate_two_dead_times(np.full(16,.01),2,8.,12.,12);c=connect();done={v[0] for v in c.execute('SELECT id FROM records')};new=0;start=time.perf_counter();total=len(list(cases(p)))
    for case in cases(p):
        if case['id'] in done:continue
        row=worker(case,p);c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(row)));c.commit();new+=1
        if (len(done)+new)%8==0:print('Safe context development',len(done)+new,'/',total,'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==total;v=dict(status='COMPLETE',cases=total,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),seconds_this_invocation=time.perf_counter()-start,newly_computed_this_invocation=new)
    (OUT/'completion.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)


def endpoint(row,method,p):
    v=row['samples'][method];declared=v.get('declared',False) if method=='donor_certified' else v['pvalue']<=p['settings']['alpha'];positive=row['case']['positive'];k=p['settings']['bins'];distance=lambda x,y:abs((x-y+k/2)%k-k/2)
    correct=positive and declared and distance(v['primary'],row['truth']['primary'])<=2 and distance(v['delay'],row['truth']['secondary'])<=2
    return dict(correct=int(correct),wrong=int(positive and declared and not correct),false=int(not positive and declared),positive=int(positive),null=int(not positive))


def analyze():
    p=check();rows=read();assert len(rows)==len(list(cases(p)));totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS};groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:
            per={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
            for r in rows:
                if r['case']['regime']!=gi or r['case']['family']!=family:continue
                for m in METHODS:
                    for k,v in endpoint(r,m,p).items():per[m][k]+=v;totals[m][k]+=v
            groups.append(dict(regime=g['name'],family=family,methods=per))
    x,b=[totals[m] for m in ['donor_bootstrap','local_bootstrap']];rule=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_total_errors=x['wrong']+x['false']<b['wrong']+b['false'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
    certs=[r['samples']['donor_certified']['certificate'] for r in rows];reasons={z:sum(v['reason']==z for v in certs) for z in sorted({v['reason'] for v in certs})}
    result=dict(status='HETEROGENEOUS_DEVELOPMENT_ONLY',totals=totals,per_regime_family=groups,primary_rule=rule,continue_to_fresh_efficacy_confirmation=all(rule.values()),certified_reasons=reasons,bootstrap_optimizer_failures=sum(r['bootstrap_optimizer_failures'] for r in rows),scope=p['scope'],calibration_scope=p['calibration'],information_budget=p['design'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_regime_family'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
