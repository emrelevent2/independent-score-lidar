"""Audit ALL recorded optimizer failures without replacing frozen outcomes."""
import json,time,hashlib,math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import transported_joint_development as d
from transported_joint_integrity import reference_objective
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times


def branch_scores(fits,local_count,supported,null_ll):
    local=max(fits[:local_count],key=lambda z:z['loglik']);best=max(fits,key=lambda z:z['loglik']);ll=local['loglik']
    if supported:ll=float(logsumexp([local['loglik']+math.log(.5),fits[-2]['loglik']+math.log(.25),fits[-1]['loglik']+math.log(.25)]))
    return dict(local_joint=max(0.,2*(local['loglik']-null_ll)),context_max=max(0.,2*(best['loglik']-null_ll)),context_transport=max(0.,2*(ll-null_ll)))


def inspect_fits(fits,s,e,fit,supported,caseid,draw):
    out=[dict(z) for z in fits];records=[];a,t,b=fit['theta'];k=len(s)
    for index,z in enumerate(fits):
        if z['optimizer_success']:continue
        start=z['start'];bounds=[(0,20),(t-4.5,t+4.5),(1e-6,30),(0,20),(start-4.5,start+4.5)]
        saved=np.array(z['theta']);saved[1]+=k*round((t-saved[1])/k);saved[4]+=k*round((start-saved[4])/k)
        results=[]
        for initial in [np.array([a,t,b,.02,start]),saved]:
            ref=minimize(reference_objective,initial,args=(s,e,1.5),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-13,gtol=1e-9,maxiter=1000,maxls=60));results.append(ref)
        ref=min(results,key=lambda r:r.fun);ll=max(z['loglik'],-float(ref.fun)*max(1.,sum(s)));out[index]['loglik']=ll
        records.append(dict(case=caseid,draw=draw,component=index,start=start,frozen_loglik=z['loglik'],reference_refined_loglik=ll,improvement=ll-z['loglik'],reference_optimizer_success=bool(ref.success)))
    return out,records


def main():
    while not (d.OUT/'completion.json').exists():time.sleep(2)
    start=time.perf_counter();p=d.check();rows=d.read();before=d.sha(d.OUT/'records.sqlite3');failures=[];case_replays=0;paths=0;decision_changes=[];max_improvement=0.
    for row in rows:
        if not row['bootstrap_alternative_failures'] and all(z['optimizer_success'] for z in row['observed_fits']) and not row['bootstrap_null_failure_indices']:continue
        assert not row['bootstrap_null_failure_indices'],'Null failures require independent stronger-null audit'
        case=row['case'];g=p['regimes'][case['regime']];cfg=p['settings'];obs,ds,de,truth,params,dh=d.base.data(case,p);assert dh==row['data_sha256'];context=donor_features(ds,de,1.5);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);fit=fit_null(s,e,1.5);supported=context['two_pulse_loglr']>20;local_count=len(row['observed_fits'])-(2 if supported else 0);observed_corrected,records=inspect_fits(row['observed_fits'],s,e,fit,supported,case['id'],'observed');failures.extend(records);observed_scores=branch_scores(observed_corrected,local_count,supported,fit['loglik']);corrected={m:[] for m in ['local_joint','context_max','context_transport']};frozen={m:[] for m in corrected};h=hashlib.sha256();count=0
        rates=fit['theta'][0]*pulse(fit['theta'][1],64,1.5)[0]+fit['theta'][2]/64
        for draw in range(199):
            bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);h.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,fit['theta'],1.5,'duration');r=d.statistics(bs,be,bf,context,cfg);newfits,records=inspect_fits(r['fits'],bs,be,bf,supported,case['id'],draw);count+=len(records);failures.extend(records);new=branch_scores(newfits,len(newfits)-(2 if supported else 0),supported,bf['loglik'])
            for m in corrected:corrected[m].append(new[m]);frozen[m].append(r['samples'][m]['score'])
        assert h.hexdigest()==row['bootstrap_path_sha256'] and count==row['bootstrap_alternative_failures'];case_replays+=1;paths+=199
        for m in corrected:
            assert frozen[m]==row['samples'][m]['bootstrap_scores'];oldp=row['samples'][m]['pvalue'];newp=pvalue(observed_scores[m],corrected[m])
            for alpha in [.005,.01,.02]:
                if (oldp<=alpha)!=(newp<=alpha):decision_changes.append(dict(case=case['id'],method=m,alpha=alpha,oldp=oldp,refinedp=newp))
        print('Audited all failed alternative fits for',case['id'],'count',count,flush=True)
    assert before==d.sha(d.OUT/'records.sqlite3');max_improvement=max([z['improvement'] for z in failures],default=0.)
    result=dict(status='PASS' if not decision_changes else 'MATERIAL_NUMERICAL_DECISION_SENSITIVITY',failure_cases_replayed=case_replays,bootstrap_paths_replayed=paths,failed_components_independently_refined=len(failures),max_likelihood_improvement=max_improvement,decision_changes=decision_changes,failed_fits=failures,records_sha256=before,seconds=time.perf_counter()-start,scope='Every recorded alternative optimizer failure,including observed fits,checked with independent NumPy objective and stricter two-start optimization. Hypothetical score/pvalue changes are numerical diagnostics only;frozen scientific records/endpoints unchanged. Does not prove global optimality or audit successful alternative fits beyond main integrity checks.')
    (d.OUT/'failure_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
