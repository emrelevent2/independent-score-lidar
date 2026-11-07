"""Read-only audit of all numerical warnings;no frozen endpoint replacement."""
import json,time,sqlite3,hashlib
import numpy as np
from scipy.optimize import minimize
import one_step_location_development as d
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import pulse,fit_null
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from transported_joint_failure_audit import inspect_fits,branch_scores
from one_step_location_integrity import independent_point,endpoint
from transported_joint_integrity import reference_objective


def stronger_null(s,e,generating_theta,warm):
    centre=float(generating_theta[1]);multi=fit_null(s,e,1.5);scale=max(1.,sum(s));candidates=[warm,multi]
    def objective(x):
        value,gradient=reference_objective(np.array([x[0],x[1],x[2],0.,0.]),s,e,1.5);return value,gradient[:3]
    for start in [warm['theta'],multi['theta'],generating_theta]:
        x=np.asarray(start,float).copy();x[1]+=64*round((centre-x[1])/64);x[1]=np.clip(x[1],centre-4.5,centre+4.5);r=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=[(0,20),(centre-4.5,centre+4.5),(1e-6,30)],options=dict(ftol=1e-13,gtol=1e-9,maxiter=1000,maxls=60));theta=r.x.copy();theta[1]%=64;candidates.append(dict(theta=theta,loglik=-float(r.fun)*scale,optimizer_success=bool(r.success)))
    best=max(candidates,key=lambda z:z['loglik']);return best


def stronger_profile(row,p):
    case=row['case'];g=p['regimes'][case['regime']];obs,_,_,_,_,dh=d.old.base.data(case,p);assert dh==row['data_sha256'];s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));point=row['one_step_location'];x=np.array(point['initial_theta']);f,_=pulse(x[1],64,1.5);gg,_=pulse(x[4],64,1.5);design=np.column_stack([f,np.ones(64)/64,gg]);scale=max(1.,sum(s))
    def objective(coeff):
        rate=np.maximum(design@coeff,1e-14);return float(-(s@np.log(rate)-e@rate)/scale),-design.T@(s/rate-e)/scale
    opt=minimize(objective,x[[0,2,3]],jac=True,method='SLSQP',bounds=[(0,20),(1e-6,30),(0,20)],options=dict(ftol=1e-14,maxiter=2000));improvement=-float(opt.fun)*scale-point['initial_loglik'];candidate=x.copy();candidate[[0,2,3]]=opt.x;theta,ll,n,kkt=independent_point(s,e,None,dict(initial_theta=candidate));primary,secondary=theta[1]%64,theta[4]%64
    if theta[3]>theta[0]:primary,secondary=secondary,primary
    hypothetical=dict(row);hypothetical['samples']=dict(row['samples']);hypothetical['samples']['transport_one_step_location']=dict(row['samples']['transport_one_step_location'],primary=float(primary),delay=float(secondary));changes=[]
    for alpha in [.005,.01,.02]:
        if endpoint(row,'transport_one_step_location',alpha)!=endpoint(hypothetical,'transport_one_step_location',alpha):changes.append(alpha)
    return dict(case=case['id'],reference_optimizer_success=bool(opt.success),initial_factor_improvement=float(improvement),reference_scaled_KKT=float(kkt),hypothetical_primary=float(primary),hypothetical_secondary=float(secondary),endpoint_change_alphas=changes)


def main():
    start=time.perf_counter();p=d.check();rows=[];failed_components=[];warnings_replayed=0;paths=0;decision_changes=[];profiles=[];profile_ids=set();null_checks=[]
    def await_row(case):
        while True:
            c=sqlite3.connect(d.OUT/'records.sqlite3');found=c.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();c.close()
            if found:return json.loads(found[0])
            time.sleep(2)
    for case in d.cases(p):
        row=await_row(case);rows.append(row)
        if not row['one_step_location']['amplitude_optimizer_success']:profiles.append(stronger_profile(row,p));profile_ids.add(case['id'])
        if not row['bootstrap_alternative_failures'] and all(z['optimizer_success'] for z in row['observed_fits']) and not row['bootstrap_null_failure_indices']:continue
        g=p['regimes'][case['regime']];cfg=p['settings'];obs,ds,de,truth,params,dh=d.old.base.data(case,p);assert dh==row['data_sha256'];context=donor_features(ds,de,1.5);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);fit=fit_null(s,e,1.5);supported=context['two_pulse_loglr']>20;corrected_fits,recs=inspect_fits(row['observed_fits'],s,e,fit,supported,case['id'],'observed');failed_components.extend(recs);corrected_observed=branch_scores(corrected_fits,len(corrected_fits)-(2 if supported else 0),supported,fit['loglik']);corrected_observed['local_score']=row['samples']['local_score']['score'];corrected={m:[] for m in corrected_observed};frozen={m:[] for m in corrected};rates=fit['theta'][0]*pulse(fit['theta'][1],64,1.5)[0]+fit['theta'][2]/64;pathhash=hashlib.sha256();failed=0;null_failed=[]
        for draw in range(199):
            bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);pathhash.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,fit['theta'],1.5,'duration');r=d.old.statistics(bs,be,bf,context,cfg);newfits,recs=inspect_fits(r['fits'],bs,be,bf,supported,case['id'],draw);failed_components.extend(recs);failed+=len(recs);new_scores=branch_scores(newfits,len(newfits)-(2 if supported else 0),supported,bf['loglik']);new_scores['local_score']=r['samples']['local_score']['score']
            if not bf['optimizer_success']:
                null_failed.append(draw);strong=stronger_null(bs,be,fit['theta'],bf);ref=d.old.statistics(bs,be,strong,context,cfg);null_checks.append(dict(case=case['id'],draw=draw,frozen_loglik=bf['loglik'],reference_loglik=strong['loglik'],improvement=strong['loglik']-bf['loglik'],max_statistic_change=max(abs(ref['samples'][m]['score']-r['samples'][m]['score']) for m in d.old.METHODS)))
                reference_fits,reference_failures=inspect_fits(ref['fits'],bs,be,strong,supported,case['id'],f'{draw}_strong_null');new_scores=branch_scores(reference_fits,len(reference_fits)-(2 if supported else 0),supported,strong['loglik']);new_scores['local_score']=ref['samples']['local_score']['score']
            for m in corrected:corrected[m].append(new_scores[m]);frozen[m].append(r['samples'][m]['score'])
        assert failed==row['bootstrap_alternative_failures'] and null_failed==row['bootstrap_null_failure_indices'] and pathhash.hexdigest()==row['bootstrap_path_sha256'];warnings_replayed+=1;paths+=199
        for m in corrected:
            assert frozen[m]==row['samples'][m]['bootstrap_scores'];newp=pvalue(corrected_observed[m],corrected[m]);oldp=row['samples'][m]['pvalue']
            for alpha in [.005,.01,.02]:
                if (newp<=alpha)!=(oldp<=alpha):decision_changes.append(dict(case=case['id'],method=m,alpha=alpha,oldp=oldp,hypothetical_refined_p=newp))
        print('Replayed all alternative optimizer warnings',case['id'],'components',failed,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    while not (d.OUT/'integrity.json').exists():time.sleep(1)
    integrity=json.loads((d.OUT/'integrity.json').read_text())
    for warning in integrity['weak_KKT_cases']:
        caseid=warning['case']
        if caseid not in profile_ids:profiles.append(stronger_profile(next(row for row in rows if row['case']['id']==caseid),p));profile_ids.add(caseid)
    assert rows==d.read();complete=json.loads((d.OUT/'completion.json').read_text());assert complete['records_sha256']==d.sha(d.OUT/'records.sqlite3');location_changes=[z for z in profiles if z['endpoint_change_alphas']]
    material_null=any(z['improvement']>1e-5 for z in null_checks)
    r=dict(status='PASS' if not decision_changes and not location_changes and not material_null else 'MATERIAL_NUMERICAL_SENSITIVITY',all_warning_cases_replayed=warnings_replayed,bootstrap_paths_replayed=paths,failed_components_independently_refined=len(failed_components),max_alternative_loglik_improvement=max([x['improvement'] for x in failed_components],default=0.),null_warning_multistart_checks=null_checks,detection_decision_changes=decision_changes,one_step_warning_or_weak_KKT_profiles_checked=profiles,one_step_endpoint_changes=location_changes,failed_alternative_fits=failed_components,records_sha256=complete['records_sha256'],seconds=time.perf_counter()-start,scope='All recorded null/alternative optimizer warnings and all weak initial-amplitude KKT cases receive stronger independent objective/profile checks. Alternative warnings use independent NumPy objective;null warnings use3start reference fits plus broad3start null fit. Hypothetical calibration statistics recomputed;frozen scientific records/endpoints preserved. No global optimization/efficacy claim.')
    (d.OUT/'failure_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
