"""Prespecified before endpoint analysis;add equal-context classical control."""
from pathlib import Path
import sys,json,time,hashlib,sqlite3
import numpy as np
from flint import ctx
import safe_donor_development as d
from safe_prior_score import candidate_bank,prior_score
from safe_donor_alternative import build_alternative,mixture_scores
from safe_deadtime_exposure import safe_lag,safe_statistics,simulate_two_dead_times
from certified_safe_components import exact_safe_statistics
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import refit_from_start,fine_scan,pvalue
P=d.ROOT/'SAFE_DONOR_SPATIAL_CONTROL_ADDENDUM.json';OUT=d.ROOT/'results/safe_donor_spatial_control_v1'


def freeze():
    if P.exists():raise FileExistsError('Frozen addendum exists')
    assert not(d.OUT/'analysis.json').exists(),'Must predeclare before any base endpoint analysis'
    p=d.check();sources={str(Path(__file__).relative_to(d.ROOT)):d.sha(Path(__file__)),'src/safe_prior_score.py':d.sha(d.ROOT/'src/safe_prior_score.py')}
    v=dict(status='FROZEN_BEFORE_BASE_ENDPOINT_ANALYSIS_OR_SPATIAL_CONTROL_EXECUTION',base_protocol_sha256=d.sha(d.P),sources=sources,base_data_generation_already_started=True,
           comparator='Classical nuisance-projected score at same donor spatial center and all same coverage centers,weighted max with prior3/4spatial and1/4uniformcoverage. Same16contextpixels as proposed method,sameknownsigma/recovery,same199twoDbootstrap paths and fully refitted nullparameters. No new data or oracletruth in comparator.',
           additional_gate='Retain original primary criterion. Separately require practical donor mixture strictly more correct AND fewer total errors than this equal-context control,with no more wrong/null separately,before claiming practical superiority over a strong spatial test. If it fails,report failure even if local-only primary passes.',
           audit='During control calculation regenerate every base observation/donor predictor/allbootstrap paths and all base bootstrap statistics;do not replace original results/times. Fixed selected refit draws additionally compared to source three-start fits.',
           scope='Additional DEVELOPMENT comparator prespecified before endpoint analysis,not a new independent confirmation. Generic matched scores/prior weighting and bootstrap are prior art;no e-value guarantee for this maximum.')
    P.write_text(json.dumps(v,indent=2));print('Frozen spatial control addendum',d.sha(P),flush=True)


def check():
    p=d.check();v=json.loads(P.read_text());assert v['base_protocol_sha256']==d.sha(d.P)
    for n,h in v['sources'].items():assert d.sha(d.ROOT/n)==h,n
    return p,v


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(v[0]) for v in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p,v=check();base=d.read();assert len(base)==len(list(d.cases(p))) and (d.OUT/'completion.json').exists();c=connect();done={x[0] for x in c.execute('SELECT id FROM records')};cfg=p['settings'];start=time.perf_counter();new=0
    for row in base:
        case=row['case']
        if case['id'] in done:continue
        g=p['regimes'][case['regime']];obs,ds,de,truth,dh=d.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'];alt=build_alternative(ds,de,cfg['sigma']);assert alt['sha256']==row['alternative_sha256'];bank=candidate_bank(alt,cfg['sigma']);lag=safe_lag(g['td'],g['te']);st=exact_safe_statistics(obs,g['cycles'],cfg['bins'],lag);s=st['counts'];e=np.array([float(x.mid()) for x in st['exposure']]);fit=fit_null(s,e,cfg['sigma']);np.testing.assert_array_equal(fit['theta'],row['null_fit']);observed=prior_score(s,e,fit['theta'],cfg['sigma'],bank);values=[];local=[];ll=[];ss=[];ee=[];bh=hashlib.sha256();strong=[]
        rates=fit['theta'][0]*pulse(fit['theta'][1],cfg['bins'],cfg['sigma'])[0]+fit['theta'][2]/cfg['bins']
        for j in range(cfg['bootstrap_replicates']):
            bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*j);bs,be,_=safe_statistics(bo,g['cycles'],cfg['bins'],lag);bh.update(bo.tobytes());ss.append(bs);ee.append(be);bf=refit_from_start(bs,be,fit['theta'],cfg['sigma'],'duration');values.append(prior_score(bs,be,bf['theta'],cfg['sigma'],bank)['score']);local.append(fine_scan(bs,be,cfg['sigma'],fit=bf)['score']);ll.append(bf['loglik'])
            for diag in row['refit_diagnostics']:
                if diag['draw']==j:
                    sf=fit_null(bs,be,cfg['sigma']);strong.append(sf['loglik']-bf['loglik'])
        assert bh.hexdigest()==row['bootstrap_path_sha256'];np.testing.assert_array_equal(local,row['samples']['local_bootstrap']['bootstrap_scores']);bn,_,_=mixture_scores(ss,ee,alt);np.testing.assert_array_equal(bn-np.asarray(ll),row['samples']['donor_bootstrap']['bootstrap_scores']);result=dict(case=case,sample=dict(**observed,pvalue=pvalue(observed['score'],values),bootstrap_scores=list(map(float,values))),base_statistics_all_replayed=True,strong_minus_warm_loglik=strong)
        c.execute('INSERT INTO records VALUES(?,?)',(case['id'],json.dumps(result)));c.commit();new+=1
        if (len(done)+new)%8==0:print('Equal-context control and base replay',len(done)+new,'/',len(base),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert len(read())==len(base);z=dict(status='COMPLETE',cases=len(base),addendum_sha256=d.sha(P),base_records_sha256=d.sha(d.OUT/'records.sqlite3'),records_sha256=d.sha(OUT/'records.sqlite3'),newly_computed_this_invocation=new,seconds_this_invocation=time.perf_counter()-start);(OUT/'completion.json').write_text(json.dumps(z,indent=2));print(json.dumps(z),flush=True)


def analyze():
    p,v=check();base={r['case']['id']:r for r in d.read()};rows=read();assert len(rows)==len(base);total=dict(correct=0,wrong=0,false=0,positive=0,null=0);per=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:
            ep=dict(correct=0,wrong=0,false=0,positive=0,null=0)
            for r in rows:
                if r['case']['regime']!=gi or r['case']['family']!=family:continue
                copy=dict(base[r['case']['id']]);copy['samples']=dict(copy['samples'],spatial_prior_score=r['sample'])
                for k,z in d.endpoint(copy,'spatial_prior_score',p).items():ep[k]+=z;total[k]+=z
            per.append(dict(regime=g['name'],family=family,endpoint=ep))
    a=json.loads((d.OUT/'analysis.json').read_text());x=a['totals']['donor_bootstrap'];rule=dict(strict_more_correct=x['correct']>total['correct'],strict_fewer_total_errors=x['wrong']+x['false']<total['wrong']+total['false'],no_more_wrong=x['wrong']<=total['wrong'],no_more_null=x['false']<=total['false']);gaps=[z for r in rows for z in r['strong_minus_warm_loglik']]
    result=dict(status='PREDECLARED_ADDITIONAL_CONTEXT_CONTROL',totals=total,per_regime_family=per,additional_superiority_gate=rule,passes_additional_gate=all(rule.values()),all_base_bootstrap_statistics_replayed=True,source_three_start_comparisons=len(gaps),maximum_strong_minus_warm_loglik=max(gaps),material_refit_warnings=sum(z>1e-5 for z in gaps),scope=v['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:z for k,z in result.items() if k!='per_regime_family'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
