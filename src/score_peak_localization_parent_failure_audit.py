"""Parent12-statistic warning sensitivity on the NEW localization cohort. Localization warning sensitivity is audited separately; this PASS alone is insufficient."""
import json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import null_bank_development as d
import score_peak_localization_development as study
from score_peak_localization_integrity import await_row
import ct_dual_reference as reference
import ct_dual_reverse as reverse
from clock_pulse_likelihood import rates_and_design
from ct_full_score import projected
from ct_full_score_failure_audit import refine_full_null as refine_generator
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue,fine_scan
from safe_deadtime_exposure import simulate_two_dead_times
from one_step_location_failure_audit import stronger_null
from transported_joint_integrity import reference_objective,independent_endpoint


def refine_full(counts,pilot,fit,observed,g):
    geometry=reference.prepare_record(observed,g['cycles']*64,64,g['td'],g['te']);scale=max(1.,sum(counts));centre=float(pilot['theta'][1]);bounds=[(0,20),(centre-4.5,centre+4.5),(1e-6,30)];candidates=[fit]
    def objective(x):
        rate,j=rates_and_design(x,64,1.5);ll,gradient=reference.full_factor(rate,j,geometry,32);return -ll/scale,-gradient/scale
    for initial in [fit['theta'],pilot['theta']]:
        x=np.array(initial,float);x[1]+=64*round((centre-x[1])/64);x[1]=np.clip(x[1],centre-4.5,centre+4.5);r=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-13,gtol=1e-9,maxiter=1000,maxls=60));theta=r.x.copy();theta[1]%=64;candidates.append(dict(theta=theta,loglik=-float(r.fun)*scale,optimizer_success=bool(r.success)))
    best=max(candidates,key=lambda z:z['loglik']);return best,dict(improvement=best['loglik']-fit['loglik'],reference_optimizer_success=best['optimizer_success'])


def refine_alternatives(fits,s,e,pilot,caseid,draw):
    a,t,b=pilot['theta'];out=[dict(f) for f in fits];records=[];scale=max(1.,sum(s))
    for index,z in enumerate(fits):
        if z['optimizer_success']:continue
        centre=z['start'];saved=np.array(z['theta']);saved[1]+=64*round((t-saved[1])/64);saved[4]+=64*round((centre-saved[4])/64);bounds=[(0,20),(t-4.5,t+4.5),(1e-6,30),(0,20),(centre-4.5,centre+4.5)];refs=[]
        for initial in [saved,np.array([a,t,b,.02,centre])]:refs.append(minimize(reference_objective,initial,args=(s,e,1.5),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-13,gtol=1e-9,maxiter=1000,maxls=60)))
        best=min(refs,key=lambda x:x.fun);ll=-float(best.fun)*scale;x=best.x.copy();x[1]%=64;x[4]%=64;primary,secondary=(x[1],x[4]) if x[0]>=x[3] else (x[4],x[1]);out[index].update(loglik=max(ll,z['loglik']))
        if ll>=z['loglik']-1e-8:out[index].update(theta=x.tolist(),primary=float(primary),delay=float(secondary))
        records.append(dict(case=caseid,draw=draw,component=index,improvement=max(0.,ll-z['loglik']),reference_optimizer_success=bool(best.success)))
    return out,records


def safe_statistics(s,e,pilot,context,fits):
    supported=context['two_pulse_loglr']>20;local=max(fits[:len(fits)-(2 if supported else 0)],key=lambda z:z['loglik']);best=max(fits,key=lambda z:z['loglik']);weighted=local['loglik']
    if supported:weighted=float(logsumexp([local['loglik']+np.log(.5),fits[-2]['loglik']+np.log(.25),fits[-1]['loglik']+np.log(.25)]))
    score=fine_scan(s,e,1.5,fit=pilot);out=dict(local_score={k:score[k] for k in ['score','primary','delay']})
    for name,ll,point in [('local_joint',local['loglik'],local),('context_max',best['loglik'],best),('context_transport',weighted,best)]:out[name]=dict(score=max(0.,2*(ll-pilot['loglik'])),primary=point['primary'],delay=point['delay'])
    return out


def full_statistics(s,e,pilot,fit,geometry,regime):
    if geometry['td']>=geometry['te']:
        x=fine_scan(s,e,1.5,fit=pilot);return {m:{k:x[k] for k in ['score','primary','delay']} for m in d.candidate.METHODS}
    out={}
    for name,theta in [('safe',pilot['theta']),('refit',fit['theta'])]:
        rate,j=rates_and_design(theta,64,1.5);ll,gradient,terms,pieces=reverse.full_factor(rate,np.eye(64),geometry,return_pieces=True);hb,_=d.bank().lookup(theta,regime)
        for kind,h in [('opg',pieces.T@pieces),('bank',hb)]:
            v=projected(gradient,h,j,theta,1.5);out['full_'+kind+'_'+name]={k:v[k] for k in ['score','primary','delay']}
    return out


def main():
    start=time.perf_counter();p=study.check();rows=[];alt_checks=[];null_checks=[];full_checks=[];generator_checks=[];changes=[];draws=0;observed_warnings=0
    for case in study.cases(p):
        row=await_row(case);rows.append(row);observed_warning=(not row['safe_null']['optimizer_success'] or not row['observed_full_null']['optimizer_success'] or any(not f['optimizer_success'] for f in row['observed_alternative_fits']));generator_warning=row['calibration_null'].get('failures',0)>0 or not row['calibration_null']['optimizer_success']
        if not observed_warning and not generator_warning and not row['bootstrap_warnings']:continue
        g=p['regimes'][case['regime']];obs,ds,de,_,_,dh=d.previous.old.base.data(case,p);assert dh==row['data_sha256'];lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);pilot=fit_null(s,e,1.5);context=donor_features(ds,de,1.5);geometry=d.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);saved_full=d.candidate.refit_full_null(s,1.5,pilot,geometry);corrected_scores={m:row['samples'][m]['score'] for m in d.METHODS};corrected_locations={m:{k:row['samples'][m][k] for k in ['primary','delay']} for m in d.METHODS};boots={m:list(row['samples'][m]['bootstrap_scores']) for m in d.METHODS}
        if generator_warning:generator_checks.append(dict(case=case['id'],**refine_generator(obs,g,pilot,row['calibration_null'])))
        if observed_warning:
            observed_warnings+=1;strong=stronger_null(s,e,pilot['theta'],pilot) if not pilot['optimizer_success'] else pilot
            if not pilot['optimizer_success']:null_checks.append(dict(case=case['id'],draw='observed',improvement=strong['loglik']-pilot['loglik']))
            alt=d.previous.old.statistics(s,e,strong,context,p['settings']) if strong is not pilot else dict(fits=row['observed_alternative_fits']);fits,recs=refine_alternatives(alt['fits'],s,e,strong,case['id'],'observed');alt_checks.extend(recs);safe=safe_statistics(s,e,strong,context,fits);fullfit=d.candidate.refit_full_null(s,1.5,strong,geometry)
            if not saved_full['optimizer_success'] and g['td']<g['te']:fullfit,rec=refine_full(s,strong,fullfit,obs,g);full_checks.append(dict(case=case['id'],draw='observed',**rec))
            full=full_statistics(s,e,strong,fullfit,geometry,case['regime'])
            for cal in ['common','original']:
                for key,name in d.BASE.items():corrected_scores[name+'_'+cal]=safe[key]['score'];corrected_locations[name+'_'+cal]={k:safe[key][k] for k in ['primary','delay']}
            for m,v in full.items():corrected_scores[m]=v['score'];corrected_locations[m]={k:v[k] for k in ['primary','delay']}
        for warning in row['bootstrap_warnings']:
            cal=warning['calibration'];draw=warning['draw'];theta=np.array(row['calibration_null']['theta'] if cal=='common' else row['safe_null']['theta']);rate,_=rates_and_design(theta,64,1.5);bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,theta,1.5,'duration');r=d.previous.old.statistics(bs,be,bf,context,p['settings']);failed=[i for i,x in enumerate(r['fits']) if not x['optimizer_success']];assert failed==warning['alternative_failure_components'] and (not bf['optimizer_success'])==warning['safe_null_failure'];draws+=1
            for key,name in d.BASE.items():assert r['samples'][key]['score']==row['samples'][name+'_'+cal]['bootstrap_scores'][draw]
            strong=stronger_null(bs,be,theta,bf) if warning['safe_null_failure'] else bf
            if warning['safe_null_failure']:null_checks.append(dict(case=case['id'],draw=f'{cal}_{draw}',improvement=strong['loglik']-bf['loglik']))
            rr=d.previous.old.statistics(bs,be,strong,context,p['settings']) if strong is not bf else r;fits,recs=refine_alternatives(rr['fits'],bs,be,strong,case['id'],f'{cal}_{draw}');alt_checks.extend(recs);safe=safe_statistics(bs,be,strong,context,fits);targets=['common','original'] if g['td']>=g['te'] else [cal]
            for target in targets:
                for key,name in d.BASE.items():boots[name+'_'+target][draw]=safe[key]['score']
            if cal=='common':
                bg=d.kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);frozen=d.candidate.statistics(bs,be,1.5,bf,bg,d.bank(),case['regime']);assert (not frozen['full_null']['optimizer_success'])==warning['full_null_failure']
                for m in d.candidate.METHODS:assert frozen['samples'][m]['score']==row['samples'][m]['bootstrap_scores'][draw]
                fullfit=d.candidate.refit_full_null(bs,1.5,strong,bg) if strong is not bf else frozen['full_null']
                if warning['full_null_failure'] and g['td']<g['te']:fullfit,rec=refine_full(bs,strong,fullfit,bo,g);full_checks.append(dict(case=case['id'],draw=draw,**rec))
                full=full_statistics(bs,be,strong,fullfit,bg,case['regime'])
                for m in d.candidate.METHODS:boots[m][draw]=full[m]['score']
        hypothetical=dict(row,samples=dict(row['samples']))
        for m in d.METHODS:
            pv=pvalue(corrected_scores[m],boots[m]);hypothetical['samples'][m]=dict(row['samples'][m],score=corrected_scores[m],pvalue=pv,**corrected_locations[m])
            for alpha in [.005,.01,.02]:
                if independent_endpoint(hypothetical,m,alpha)!=independent_endpoint(row,m,alpha):changes.append(dict(case=case['id'],method=m,alpha=alpha,frozen_p=row['samples'][m]['pvalue'],hypothetical_reference_p=pv))
        print('All bank numerical warnings checked',case['id'],'draws',len(row['bootstrap_warnings']),flush=True)
    while not (study.OUT/'completion.json').exists():time.sleep(1)
    assert rows==study.read();c=json.loads((study.OUT/'completion.json').read_text());assert c['records_sha256']==d.sha(study.OUT/'records.sqlite3');assert draws==sum(len(row['bootstrap_warnings']) for row in rows);material=any(x['improvement']>1e-5 for x in generator_checks+null_checks)
    result=dict(status='PASS' if not changes and not material else 'MATERIAL_NUMERICAL_SENSITIVITY',all_warning_draws_regenerated=draws,observed_warning_cases_checked=observed_warnings,failed_safe_alternative_components_refined=len(alt_checks),max_safe_alternative_loglik_improvement=max([x['improvement'] for x in alt_checks],default=0.),safe_null_reference_checks=null_checks,full_null_reference_checks=full_checks,calibration_generator_reference_checks=generator_checks,operatingpoint_endpoint_changes=changes,failed_safe_alternative_fits=alt_checks,records_sha256=c['records_sha256'],seconds=time.perf_counter()-start,scope='Everyrecordedbootstrap/observedoptimizerwarningchecked. IndependentNumPysafealternativeobjective,strongersafenullfits,32pointquadraturefullnullsameboxrefinement. All12hypotheticalpvalues/locationsrecountedat3alphas;frozenrecordsunchanged. Materialgeneratorchangesrequirefurthercalibrationsensitivity. Not globaloptimization/physical/uniformnull/efficacycertification.')
    (study.OUT/'parent_failure_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
