"""Streaming complete observed-data/rank replay and independent score audit."""
import json,time,sqlite3
import numpy as np
from scipy.stats import binomtest
import null_bank_development as d
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from ct_full_score import templates,statistic as old_full_score
from audit_null_bank_validation import interpolate
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null
from transported_joint_integrity import independent_endpoint


def await_row(case):
    while True:
        c=sqlite3.connect(d.OUT/'records.sqlite3');r=c.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();c.close()
        if r:return json.loads(r[0])
        time.sleep(2)


def reference_score(gradient,pieces,j,theta,h=None):
    grid,bank=templates(64,1.5)
    if h is None:design=pieces;target=np.ones(len(pieces))
    else:design=np.linalg.cholesky(h).T;target=np.linalg.solve(design.T,gradient)
    u,s,v=np.linalg.svd(design@j,full_matrices=False);basis=u[:,s>s[0]*1e-5];z=design@bank.T;z-=basis@(basis.T@z);raw=z.T@target;variance=np.sum(z*z,axis=0);allowed=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(variance>1e-8);scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(variance,1e-8),-np.inf);best=int(np.argmax(scores));return dict(score=float(scores[best]),primary=float(theta[1]),delay=float(grid[best]))


def main():
    start=time.perf_counter();p=d.check();rows=[];whole=0;paths=0;rankchecks=0;identities=0;refchecks=0;maxgap=0.;bankarrays={k:v for k,v in np.load(d.training.OUT/'bank.npz').items()};bankmatrixchecks=0;events=[dict(observed=0,scored=0) for _ in p['regimes']]
    for i,case in enumerate(d.cases(p)):
        row=await_row(case);assert row['case']==case;rows.append(row);g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=d.previous.old.base.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'] and params==row['generating_parameters'];assert len(obs)==row['observed_events'];s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));assert int(s.sum())==row['scored_events'];events[case['regime']]['observed']+=len(obs);events[case['regime']]['scored']+=int(s.sum());context=donor_features(ds,de,1.5);assert {k:context[k] for k in ['primary','secondary','two_pulse_loglr']}==row['context'];pilot=fit_null(s,e,1.5);assert d.json_fit(pilot)==row['safe_null'];geometry=d.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);full=d.candidate.statistics(s,e,1.5,pilot,geometry,d.bank(),case['regime']);assert d.json_fit(full['full_null'])==row['observed_full_null'] and full['bank_flags']==row['observed_bank_flags'];cal=d.previous.fit_full_null(s,e,1.5,pilot,geometry);assert d.json_fit(cal)==row['calibration_null'];old=d.previous.old.statistics(s,e,pilot,context,p['settings']);assert old['fits']==row['observed_alternative_fits']
        for cal in ['common','original']:
            for key,name in d.BASE.items():assert {k:row['samples'][name+'_'+cal][k] for k in ['score','primary','delay']}==old['samples'][key]
        for m in d.candidate.METHODS:assert {k:row['samples'][m][k] for k in ['score','primary','delay']}==full['samples'][m]
        for m in d.METHODS:
            z=row['samples'][m];b=z['bootstrap_scores'];assert len(b)==199 and np.all(np.isfinite(b));assert z['pvalue']==(1+sum(x>=z['score'] for x in b))/200;rankchecks+=1
        if g['td']>=g['te']:
            for m in d.candidate.METHODS:assert row['samples'][m]==row['samples']['safe_score_common']==row['samples']['safe_score_original']
            for name in d.BASE.values():assert row['samples'][name+'_common']==row['samples'][name+'_original']
            assert row['bootstrap_path_sha256']['common']==row['bootstrap_path_sha256']['original'];identities+=1
        else:
            for theta in [pilot['theta'],full['full_null']['theta']]:
                actual,_=d.bank().lookup(theta,case['regime']);expected=interpolate(bankarrays,theta,case['regime']);np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-12);bankmatrixchecks+=1
        if case['id'].endswith('_000'):
            replay=d.worker(case,p);assert {k:v for k,v in replay.items() if k!='seconds'}=={k:v for k,v in row.items() if k!='seconds'};whole+=1;paths+=199*(1 if g['td']>=g['te'] else 2)
            if g['td']<g['te']:
                rg=reference.prepare_record(obs,g['cycles']*64,64,g['td'],g['te'])
                for kind,theta in [('safe',pilot['theta']),('refit',full['full_null']['theta'])]:
                    rate,j=rates_and_design(theta,64,1.5);ll,gradient,terms,pieces=reference.full_factor(rate,np.eye(64),rg,32,True);h=interpolate(bankarrays,theta,case['regime'])
                    for weight,matrix in [('opg',None),('bank',h)]:
                        ref=reference_score(gradient,pieces,j,theta,matrix);saved=full['samples']['full_'+weight+'_'+kind];gap=abs(ref['score']-saved['score']);maxgap=max(maxgap,gap);assert gap<1e-5 and ref['delay']==saved['delay'] and ref['primary']==saved['primary'];refchecks+=1
                comparator,_=old_full_score(s,e,1.5,pilot,geometry);saved=full['samples']['full_opg_safe'];assert abs(comparator['score']-saved['score'])<1e-6 and comparator['delay']==saved['delay']
        if (i+1)%32==0:print('Bank independent audit',i+1,'/640;wholeworkers',whole,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    while True:
        try:a=json.loads((d.OUT/'analysis.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1)
    assert rows==d.read();d.check();c=json.loads((d.OUT/'completion.json').read_text());assert c['records_sha256']==d.sha(d.OUT/'records.sqlite3') and c['protocol_sha256']==d.sha(d.P);before={n:d.sha(d.OUT/n) for n in ['records.sqlite3','completion.json','analysis.json']};all_totals={};groups={}
    for alpha in [.005,.01,.02]:
        t={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS}
        for row in rows:
            key=(row['case']['regime'],row['case']['family']);groups.setdefault(key,{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS})
            for m in d.METHODS:
                for k,n in independent_endpoint(row,m,alpha).items():
                    t[m][k]+=n
                    if alpha==.01:groups[key][m][k]+=n
        assert t==a['descriptive_operating_points'][str(alpha)];all_totals[str(alpha)]=t
    for group in a['per_regime_family']:
        gi=next(i for i,g in enumerate(p['regimes']) if g['name']==group['regime']);assert group['methods']==groups[(gi,group['family'])]
    t=all_totals['0.01'];assert t==a['totals'];x=t[d.PRIMARY];rule={};paired={}
    for m in [m for m in d.METHODS if m!=d.PRIMARY]:
        b=t[m];rule[m]=dict(no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
        if m in ['safe_score_common','safe_score_original','full_opg_safe']:rule[m].update(strict_more_correct=x['correct']>b['correct'],strict_fewer_errors=x['wrong']+x['false']<b['wrong']+b['false'])
        else:rule[m]['at_least_correct']=x['correct']>=b['correct']
        gains=losses=fixes=new=0
        for row in rows:
            u=independent_endpoint(row,d.PRIMARY,.01);v=independent_endpoint(row,m,.01);gains+=int(u['correct']>v['correct']);losses+=int(u['correct']<v['correct']);fixes+=int(u['wrong']+u['false']<v['wrong']+v['false']);new+=int(u['wrong']+u['false']>v['wrong']+v['false'])
        paired[m]=dict(correct_gains=gains,correct_losses=losses,error_fixes=fixes,new_errors=new,correct_exact_p=binomtest(gains,gains+losses).pvalue if gains+losses else 1.,error_exact_p=binomtest(fixes,fixes+new).pvalue if fixes+new else 1.)
    passed=all(all(z.values()) for z in rule.values());assert rule==a['preliminary_rule'] and passed==a['preliminary_rule_passed'] and not a['continue_to_fresh_confirmation'];assert all(d.sha(d.OUT/n)==h for n,h in before.items())
    result=dict(status='PASS',all_observed_query_context_data_replayed=640,all12observed_statistics_and_null_fits_recomputed=True,all12rankpvalues_reranked=rankchecks,all_operatingpoint_endpoints_independently_recounted=640*12*3,nohidden_exact_pipeline_identity_cases=identities,independent_bank_matrix_interpolations=bankmatrixchecks,whole_workers_replayed=whole,unique_bootstrap_paths_replayed=paths,independent_quadrature_SVD_Cholesky_score_checks=refchecks,max_independent_score_gap=maxgap,event_counts_per_regime=events,paired=paired,outputs_sha256=before,seconds=time.perf_counter()-start,scope='All640query/contextsets/observed12scores/nullfits,7680rankpvalues,23040endpoints.28wholeworkers8358uniqueMCpaths;56hiddenqueryscoresviaindependentquadrature/SVDorCholeskyprojection. Limitedbootstrapreplay,not all191040paths. Optimizerwarningsauditedseparately. No efficacy/physical/uniformrisk orQ1certification.')
    (d.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
