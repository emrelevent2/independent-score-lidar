"""Read-only streaming replay of the prospective continuous-score study.

Numerical independence uses segment quadrature rather than analytic mixed
moments, and weighted-design SVD rather than normal-equation projection.
Neither this audit nor Monte Carlo calibration establishes physical validity.
"""
import json,time,sqlite3
import numpy as np
from scipy.stats import binomtest
import ct_full_score_development as d
import ct_dual_reference as reference
from ct_full_score import statistic,fit_full_null,templates
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null
from transported_joint_integrity import reference_objective,independent_endpoint


def independent_full_score(rate,j,geometry,theta):
    value,gradient,terms,pieces=reference.full_factor(rate,np.eye(64),geometry,order=32,return_pieces=True)
    grid,bank=templates(64,1.5);u,s,v=np.linalg.svd(pieces@j,full_matrices=False)
    basis=u[:,s>s[0]*1e-5];z=pieces@bank.T;z-=basis@(basis.T@z)
    # Per-factor likelihood score sums to gradient; projecting its unit vector
    # gives the identical efficient score without forming a64x64 OPG matrix.
    raw=np.sum(z,axis=0);variance=np.sum(z*z,axis=0)
    allowed=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(variance>1e-8)
    scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(variance,1e-8),-np.inf)
    best=int(np.argmax(scores))
    return dict(score=float(scores[best]),delay=float(grid[best]),primary=float(theta[1])),float(value)


def await_row(case):
    while True:
        c=sqlite3.connect(d.OUT/'records.sqlite3');found=c.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();c.close()
        if found:return json.loads(found[0])
        time.sleep(2)


def main():
    start=time.perf_counter();p=d.check();rows=[];whole=0;pathcount=0;rankchecks=0;identity=0;reference_checks=0;max_score_gap=0.;max_ll_gap=0.;point_gap=0.;events=[dict(observed=0,scored=0) for _ in p['regimes']]
    for i,case in enumerate(d.cases(p)):
        row=await_row(case);assert row['case']==case;rows.append(row);g=p['regimes'][case['regime']]
        obs,ds,de,truth,params,dh=d.old.base.data(case,p)
        assert dh==row['data_sha256'] and truth==row['truth'] and params==row['generating_parameters'] and len(obs)==row['observed_events']
        s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));assert int(s.sum())==row['scored_events'];events[case['regime']]['observed']+=len(obs);events[case['regime']]['scored']+=int(s.sum())
        context=donor_features(ds,de,1.5);assert {k:context[k] for k in ['primary','secondary','two_pulse_loglr']}==row['context'];pilot=fit_null(s,e,1.5);assert pilot['theta'].tolist()==row['safe_null']
        geometry=d.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);v,diag=statistic(s,e,1.5,pilot,geometry);assert diag==row['full_score_diagnostics']
        null=fit_full_null(s,e,1.5,pilot,geometry);assert {k:null[k] for k in ['loglik','optimizer_success','failures','mode']}=={k:row['calibration_null'][k] for k in ['loglik','optimizer_success','failures','mode']} and null['theta'].tolist()==row['calibration_null']['theta']
        observed=d.old.statistics(s,e,pilot,context,p['settings']);assert observed['fits']==row['observed_alternative_fits']
        for component in observed['fits']:
            val,_=reference_objective(component['theta'],s,e,1.5);gap=abs(-val*max(1.,sum(s))-component['loglik']);point_gap=max(point_gap,gap);assert gap<1e-6
        for calibration in ['common','original']:
            for key,name in d.BASE.items():
                saved=row['samples'][name+'_'+calibration];assert {k:saved[k] for k in ['score','primary','delay']}==observed['samples'][key]
        assert {k:row['samples']['full_score_common'][k] for k in ['score','primary','delay']}==v
        for method in d.METHODS:
            z=row['samples'][method];b=z['bootstrap_scores'];assert len(b)==199 and np.all(np.isfinite(b));assert (1+sum(x>=z['score'] for x in b))/200==z['pvalue'];rankchecks+=1
        if g['td']>=g['te']:
            assert row['samples']['full_score_common']==row['samples']['safe_score_common']==row['samples']['safe_score_original'];assert row['bootstrap_path_sha256']['common']==row['bootstrap_path_sha256']['original']
            for name in d.BASE.values():assert row['samples'][name+'_common']==row['samples'][name+'_original']
            identity+=1
        if case['id'].endswith('_000'):
            replay=d.worker(case,p);assert {k:v for k,v in replay.items() if k!='seconds'}=={k:v for k,v in row.items() if k!='seconds'};whole+=1;pathcount+=199*(1 if g['td']>=g['te'] else 2)
            rate,j=rates_and_design(pilot['theta'],64,1.5)
            if g['td']<g['te']:
                # Independent Python geometry plus quadrature forward factors.
                rg=reference.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);ref,ll=independent_full_score(rate,j,rg,pilot['theta']);gap=abs(ref['score']-v['score']);max_score_gap=max(max_score_gap,gap);max_ll_gap=max(max_ll_gap,abs(ll-diag['full_loglik']));assert gap<1e-5 and ref['delay']==v['delay'] and abs(ll-diag['full_loglik'])<1e-6;reference_checks+=1
        if (i+1)%32==0:print('Full continuous score audit',i+1,'/640;fullworkers',whole,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    while True:
        try:analysis=json.loads((d.OUT/'analysis.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1)
    assert rows==d.read();d.check();complete=json.loads((d.OUT/'completion.json').read_text());assert complete['cases']==640 and complete['protocol_sha256']==d.sha(d.P) and complete['records_sha256']==d.sha(d.OUT/'records.sqlite3');before={name:d.sha(d.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    totals={};groups={}
    for alpha in [.005,.01,.02]:
        t={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS}
        for row in rows:
            key=(row['case']['regime'],row['case']['family']);groups.setdefault(key,{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS})
            for m in d.METHODS:
                for k,n in independent_endpoint(row,m,alpha).items():
                    t[m][k]+=n
                    if alpha==.01:groups[key][m][k]+=n
        assert t==analysis['descriptive_operating_points'][str(alpha)];totals[str(alpha)]=t
    assert totals['0.01']==analysis['totals']
    for saved in analysis['per_regime_family']:
        gi=next(i for i,g in enumerate(p['regimes']) if g['name']==saved['regime']);assert saved['methods']==groups[(gi,saved['family'])]
    x=totals['0.01']['full_score_common'];rule={};paired={}
    for control in [m for m in d.METHODS if m!='full_score_common']:
        b=totals['0.01'][control]
        if control in ['safe_score_common','safe_score_original']:rule[control]=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_errors=x['wrong']+x['false']<b['wrong']+b['false'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
        else:rule[control]=dict(at_least_correct=x['correct']>=b['correct'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
        win=loss=fix=new=0
        for row in rows:
            a=independent_endpoint(row,'full_score_common',.01);b=independent_endpoint(row,control,.01);win+=int(a['correct']>b['correct']);loss+=int(a['correct']<b['correct']);fix+=int(a['wrong']+a['false']<b['wrong']+b['false']);new+=int(a['wrong']+a['false']>b['wrong']+b['false'])
        paired[control]=dict(correct_gains=win,correct_losses=loss,error_fixes=fix,new_errors=new,correct_exact_p=binomtest(win,win+loss).pvalue if win+loss else 1.,error_exact_p=binomtest(fix,fix+new).pvalue if fix+new else 1.)
    assert rule==analysis['primary_rule'] and analysis['continue_to_fresh_confirmation']==all(all(r.values()) for r in rule.values());assert all(d.sha(d.OUT/name)==h for name,h in before.items())
    r=dict(status='PASS',all_observed_query_context_data_replayed=640,all9observed_scores_and_calibration_nulls_recomputed=True,rankpvalues_checked=rankchecks,operating_point_endpoints_independently_recounted=640*9*3,no_hidden_exact_pipeline_identity_cases=identity,whole_workers_replayed=whole,unique_bootstrap_paths_replayed=pathcount,independent_quadrature_SVD_full_score_checks=reference_checks,max_reference_score_gap=max_score_gap,max_reference_full_loglik_gap=max_ll_gap,max_numpy_safe_alternative_point_loglik_gap=point_gap,event_counts_per_regime=events,paired=paired,outputs_sha256=before,seconds=time.perf_counter()-start,scope='All640observations/contextsets,9method scores,fullcalibrationnullfits and5760rankpvalues;17280endpointdecisions.28fullworkers;14hiddenqueryscore checks using32pointsegmentquadrature and SVD projection. Limited bootstrap replay,not all191040unique generated paths. Numerical optimizer warnings audited separately. No physical/uniformnull/efficacy certification.')
    (d.OUT/'integrity.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
