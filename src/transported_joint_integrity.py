"""Read-only scientific replay;independent NumPy objective and endpoint recount.

First case in each regime/family is selected before outcomes for full worker
replay and reference optimizer comparison. No failed point is overwritten.
"""
import json,time,sqlite3,math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binomtest
import transported_joint_development as d
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import pulse,fit_null,likelihood_parts


def reference_objective(x,s,e,sigma):
    a,t,b,c,z=x;k=len(s);f,df=pulse(t,k,sigma);g,dg=pulse(z,k,sigma);ll,u,_=likelihood_parts(s,e,a*f+b/k+c*g,'duration');j=np.column_stack([f,a*df,np.ones(k)/k,g,c*dg]);return -ll/max(1.,sum(s)),-j.T@u/max(1.,sum(s))


def independent_endpoint(row,method,alpha):
    v=row['samples'][method];positive=row['case']['positive'];declared=v['pvalue']<=alpha;distance=lambda a,b:abs((a-b+32)%64-32);correct=positive and declared and distance(v['primary'],row['truth']['primary'])<=2 and distance(v['delay'],row['truth']['secondary'])<=2
    return dict(correct=int(correct),wrong=int(positive and declared and not correct),false=int(not positive and declared),positive=int(positive),null=int(not positive))


def main():
    start=time.perf_counter();p=d.check();cfg=p['settings'];expected=list(d.cases(p));rows=[];reference_checks=0;largest_reference_gap=0.;largest_point_gap=0.;whole=0;rankchecks=0;zero=0;events=0
    def await_row(case):
        while True:
            c=sqlite3.connect(d.OUT/'records.sqlite3');found=c.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();c.close()
            if found:return json.loads(found[0])
            time.sleep(2)
    for i,case in enumerate(expected):
        row=await_row(case);assert row['case']==case;rows.append(row);g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=d.base.data(case,p)
        assert truth==row['truth'] and params==row['generating_parameters'] and dh==row['data_sha256'] and len(obs)==row['observed_events'];events+=len(obs)
        lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);assert int(s.sum())==row['scored_events'];context=donor_features(ds,de,1.5);assert {k:context[k] for k in ['primary','secondary','two_pulse_loglr']}==row['context'];fit=fit_null(s,e,1.5);assert fit['theta'].tolist()==row['null_fit'];observed=d.statistics(s,e,fit,context,cfg);assert observed['fits']==row['observed_fits']
        for component in row['observed_fits']:
            val,_=reference_objective(component['theta'],s,e,1.5);gap=abs(-val*max(1.,sum(s))-component['loglik']);largest_point_gap=max(largest_point_gap,gap);assert gap<1e-6
        fits=row['observed_fits'];supported=context['two_pulse_loglr']>20
        if supported:
            local=max(fits[:-2],key=lambda z:z['loglik']);best=max(fits,key=lambda z:z['loglik']);weighted=float(logsumexp([local['loglik']+math.log(.5),fits[-2]['loglik']+math.log(.25),fits[-1]['loglik']+math.log(.25)]));expected_score=max(0.,2*(weighted-fit['loglik']))
        else:local=best=max(fits,key=lambda z:z['loglik']);expected_score=max(0.,2*(local['loglik']-fit['loglik']));zero+=1;assert row['samples']['local_joint']==row['samples']['context_max']==row['samples']['context_transport']
        assert expected_score==row['samples']['context_transport']['score'];assert [best['primary'],best['delay']]==[row['samples']['context_transport']['primary'],row['samples']['context_transport']['delay']]
        for m in d.METHODS:
            v=row['samples'][m];assert {k:v[k] for k in ['score','primary','delay']}==observed['samples'][m];b=np.asarray(v['bootstrap_scores']);assert len(b)==199 and np.all(np.isfinite(b));assert (1+sum(float(x)>=v['score'] for x in b))/200==v['pvalue'];rankchecks+=1
        if case['id'].endswith('_000'):
            replay=d.worker(case,p);assert {k:v for k,v in replay.items() if k!='seconds'}=={k:v for k,v in row.items() if k!='seconds'};whole+=1
            a,t,b=fit['theta']
            for component in fits:
                z=component['start'];reference=minimize(reference_objective,[a,t,b,.02,z],args=(s,e,1.5),jac=True,method='L-BFGS-B',bounds=[(0,20),(t-4.5,t+4.5),(1e-6,30),(0,20),(z-4.5,z+4.5)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=160));refll=max(float(fit['loglik']),-reference.fun*max(1.,sum(s)));gap=abs(refll-component['loglik']);largest_reference_gap=max(largest_reference_gap,gap);reference_checks+=1;assert gap<1e-4,(case['id'],z,gap)
        if (i+1)%32==0:print('Transport audit',i+1,'/320;whole workers',whole,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    while True:
        try:analysis=json.loads((d.OUT/'analysis.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1)
    assert rows==d.read();d.check();complete=json.loads((d.OUT/'completion.json').read_text());assert complete['cases']==320 and complete['protocol_sha256']==d.sha(d.P) and complete['records_sha256']==d.sha(d.OUT/'records.sqlite3')
    totals={}
    for alpha in [.005,.01,.02]:
        bymethod={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS}
        for row in rows:
            for m in d.METHODS:
                for k,n in independent_endpoint(row,m,alpha).items():bymethod[m][k]+=n
        assert bymethod==analysis['descriptive_operating_points'][str(alpha)];totals[str(alpha)]=bymethod
    assert totals['0.01']==analysis['totals'];paired={}
    for control in ['local_score','local_joint','context_max']:
        win=loss=fix=new=0
        for row in rows:
            a=independent_endpoint(row,'context_transport',.01);b=independent_endpoint(row,control,.01);win+=int(a['correct']>b['correct']);loss+=int(a['correct']<b['correct']);fix+=int(a['wrong']+a['false']<b['wrong']+b['false']);new+=int(a['wrong']+a['false']>b['wrong']+b['false'])
        paired[control]=dict(correct_gains=win,correct_losses=loss,error_fixes=fix,new_errors=new,correct_mcnemar_p=binomtest(win,win+loss).pvalue if win+loss else 1.,error_mcnemar_p=binomtest(fix,fix+new).pvalue if fix+new else 1.)
    r=dict(status='PASS',all_queries_contexts_observed_statistics_replayed=320,all_bootstrap_pvalues_reranked=rankchecks,all_operating_point_endpoint_decisions_recounted=320*4*3,whole_workers_replayed=whole,bootstrap_paths_replayed=whole*199,independent_NumPy_reference_optimizations=reference_checks,max_reference_optimizer_likelihood_gap=largest_reference_gap,max_saved_point_likelihood_gap=largest_point_gap,unsupported_context_identity_cases=zero,query_events=events,paired=paired,scientific_protocol_sha256=d.sha(d.P),outputs_sha256={n:d.sha(d.OUT/n) for n in ['records.sqlite3','completion.json','analysis.json']},seconds=time.perf_counter()-start,scope='All320observations/contexts/observedfitpoints/statistics and1280pvalues,all3840operating-point endpoints;28fullworkers5572bootstrap paths plus reference NumPy optimizers on their observed components. Not a replay of all63680bootstrap workers/fits,nor physical/finite-composite-null/efficacy validation.')
    (d.OUT/'integrity.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
