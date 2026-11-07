"""Read-only reference refinement of every recorded numerical warning.

Only flagged draws need regeneration because the frozen record stores each
draw's location and calibration branch. Original endpoints are never edited.
"""
import json,time
import numpy as np
from scipy.optimize import minimize
import ct_full_score_development as d
import ct_dual_reference as reference
from ct_full_score_integrity import await_row
from ct_full_score import statistic
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from transported_joint_failure_audit import inspect_fits,branch_scores
from one_step_location_failure_audit import stronger_null


def refine_full_null(obs,g,pilot,saved):
    geometry=reference.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);scale=max(1,len(obs));centres=[float(saved['theta'][1]),float(pilot['theta'][1])]
    out=[]
    def objective(x):
        rate,j=rates_and_design(x,64,1.5);ll,gradient=reference.full_factor(rate,j,geometry,order=32);return -ll/scale,-gradient/scale
    for centre in centres:
        for theta in [saved['theta'],pilot['theta']]:
            x=np.asarray(theta,float).copy();x[1]+=64*round((centre-x[1])/64);x[1]=np.clip(x[1],centre-4.5,centre+4.5)
            r=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=[(0,20),(centre-4.5,centre+4.5),(1e-6,30)],options=dict(ftol=1e-13,gtol=1e-9,maxiter=1000,maxls=60));out.append(dict(loglik=-float(r.fun)*scale,success=bool(r.success),theta=r.x.tolist()))
    best=max(out,key=lambda x:x['loglik']);return dict(reference_loglik=best['loglik'],frozen_loglik=saved['loglik'],improvement=best['loglik']-saved['loglik'],reference_optimizer_success=best['success'],reference_theta=best['theta'])


def main():
    start=time.perf_counter();p=d.check();allcomponents=[];nulls=[];fullnulls=[];changes=[];replayed=0;warning_records=0;rows=[]
    for case in d.cases(p):
        row=await_row(case);rows.append(row);observed_warning=any(not f['optimizer_success'] for f in row['observed_alternative_fits']);cal_warning=row['calibration_null']['failures']>0 or not row['calibration_null']['optimizer_success']
        if not observed_warning and not row['bootstrap_warnings'] and not cal_warning:continue
        g=p['regimes'][case['regime']];cfg=p['settings'];obs,ds,de,truth,params,dh=d.old.base.data(case,p);assert dh==row['data_sha256'];lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);pilot=fit_null(s,e,1.5);context=donor_features(ds,de,1.5);supported=context['two_pulse_loglr']>20
        if cal_warning:fullnulls.append(dict(case=case['id'],**refine_full_null(obs,g,pilot,row['calibration_null'])))
        fits,records=inspect_fits(row['observed_alternative_fits'],s,e,pilot,supported,case['id'],'observed');allcomponents.extend(records);obs_corrected=branch_scores(fits,len(fits)-(2 if supported else 0),supported,pilot['loglik']);corrected={m:list(row['samples'][m]['bootstrap_scores']) for m in d.METHODS};scores={m:row['samples'][m]['score'] for m in d.METHODS}
        for key,name in d.BASE.items():
            if key=='local_score':continue
            for calibration in ['common','original']:scores[name+'_'+calibration]=obs_corrected[key]
        for warning in row['bootstrap_warnings']:
            calibration=warning['calibration'];draw=warning['draw'];theta=np.array(row['calibration_null']['theta'] if calibration=='common' else row['safe_null']);rate,_=rates_and_design(theta,64,1.5);bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,theta,1.5,'duration');r=d.old.statistics(bs,be,bf,context,cfg);failed=[i for i,f in enumerate(r['fits']) if not f['optimizer_success']];assert failed==warning['alternative_failure_components'] and (not bf['optimizer_success'])==warning['null_failure'];warning_records+=1;replayed+=1
            for key,name in d.BASE.items():assert r['samples'][key]['score']==row['samples'][name+'_'+calibration]['bootstrap_scores'][draw]
            strong=stronger_null(bs,be,theta,bf) if warning['null_failure'] else bf
            if warning['null_failure']:
                ref=d.old.statistics(bs,be,strong,context,cfg);nulls.append(dict(case=case['id'],calibration=calibration,draw=draw,frozen_loglik=bf['loglik'],reference_loglik=strong['loglik'],improvement=strong['loglik']-bf['loglik']))
            else:ref=r
            refined,records=inspect_fits(ref['fits'],bs,be,strong,supported,case['id'],f'{calibration}_{draw}');allcomponents.extend(records);v=branch_scores(refined,len(refined)-(2 if supported else 0),supported,strong['loglik']);v['local_score']=ref['samples']['local_score']['score']
            calibrations=[calibration]
            if g['td']>=g['te']:calibrations=['common','original']
            for target in calibrations:
                for key,name in d.BASE.items():corrected[name+'_'+target][draw]=v[key]
            if calibration=='common':
                bg=d.kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);frozen,_=statistic(bs,be,1.5,bf,bg);assert frozen['score']==row['samples']['full_score_common']['bootstrap_scores'][draw];new,_=statistic(bs,be,1.5,strong,bg);corrected['full_score_common'][draw]=new['score']
        for m in d.METHODS:
            newp=pvalue(scores[m],corrected[m]);oldp=row['samples'][m]['pvalue']
            for alpha in [.005,.01,.02]:
                if (newp<=alpha)!=(oldp<=alpha):changes.append(dict(case=case['id'],method=m,alpha=alpha,frozen_p=oldp,hypothetical_refined_p=newp))
        print('Full score warning audit',case['id'],'draws',len(row['bootstrap_warnings']),'components',len(allcomponents),flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    assert rows==d.read();completion=json.loads((d.OUT/'completion.json').read_text());assert completion['records_sha256']==d.sha(d.OUT/'records.sqlite3');assert warning_records==sum(len(r['bootstrap_warnings']) for r in rows)
    material=any(x['improvement']>1e-5 for x in fullnulls+nulls) or any(x['improvement']>1e-5 and x['draw']=='observed' for x in allcomponents)
    result=dict(status='PASS' if not changes and not material else 'MATERIAL_NUMERICAL_SENSITIVITY',all_recorded_warning_draws_replayed=warning_records,unique_bootstrap_paths_regenerated=replayed,failed_alternative_components_refined=len(allcomponents),max_alternative_loglik_improvement=max([x['improvement'] for x in allcomponents],default=0.),null_warning_reference_fits=nulls,query_calibration_null_warning_reference_fits=fullnulls,detection_changes=changes,failed_alternative_fits=allcomponents,records_sha256=completion['records_sha256'],seconds=time.perf_counter()-start,scope='Every saved numerical warning audited by explicit draw index. Independent NumPy alternative objective and stronger null fits;full calibration warnings use32pointsegmentquadrature and fourstricterstarts. Endpoint records unchanged. No global optimization or physical/efficacy guarantee. Material query-generator improvements require a separate full calibration sensitivity replay.')
    (d.OUT/'failure_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
