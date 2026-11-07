"""Streaming read-only replay and independent weighted-design Fisher audit."""
import json,time,sqlite3
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest
import one_step_location_development as d
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from transported_joint_integrity import reference_objective
from joint_one_step_location import estimate


def independent_point(s,e,null_fit,saved):
    # Independent weighted-design least squares computes the Fisher direction
    # without forming/pseudoinverting the5x5normal matrix used by the estimator.
    initial=np.asarray(saved['initial_theta']);a,t,b,c,z=initial;k=len(s);f,df=pulse(t,k,1.5);g,dg=pulse(z,k,1.5);rate=np.maximum(a*f+b/k+c*g,1e-14);u=s/rate-e;w=e/rate;j=np.column_stack([f,a*df,np.ones(k)/k,g,c*dg]);root=np.sqrt(w);target=np.divide(u,root,out=np.zeros(k),where=root>0);direction=np.linalg.lstsq(j*root[:,None],target,rcond=1e-5)[0]
    oldll=-reference_objective(initial,s,e,1.5)[0]*max(1.,sum(s));selected=initial.copy();selectedll=oldll;backtrack=21
    for n in range(21):
        x=initial+direction*2.**(-n);x[[0,3]]=np.clip(x[[0,3]],0,20);x[2]=np.clip(x[2],1e-6,30);x[1]=np.clip(x[1],initial[1]-4.5,initial[1]+4.5);x[4]=np.clip(x[4],initial[4]-4.5,initial[4]+4.5);ll=-reference_objective(x,s,e,1.5)[0]*max(1.,sum(s))
        if ll>=oldll-1e-10:selected=x;selectedll=ll;backtrack=n;break
    # KKT residual of the preceding3-amplitude concave profile.
    design=np.column_stack([f,np.ones(k)/k,g]);gradient=design.T@u/max(1.,sum(s));coefficients=initial[[0,2,3]];lower=np.array([0,1e-6,0]);upper=np.array([20,30,20]);residual=np.abs(gradient)
    residual=np.where(coefficients<=lower+1e-8,np.maximum(gradient,0),residual);residual=np.where(coefficients>=upper-1e-8,np.maximum(-gradient,0),residual)
    return selected,selectedll,backtrack,float(max(residual))


def endpoint(row,m,alpha):
    z=row['samples'][m];pos=row['case']['positive'];declared=z['pvalue']<=alpha;distance=lambda a,b:abs((a-b+32)%64-32);correct=pos and declared and distance(z['primary'],row['truth']['primary'])<=2 and distance(z['delay'],row['truth']['secondary'])<=2
    return dict(correct=int(correct),wrong=int(pos and declared and not correct),false=int(not pos and declared),positive=int(pos),null=int(not pos))


def main():
    start=time.perf_counter();p=d.check();rows=[];expected=list(d.cases(p));whole=0;rankchecks=0;equalcalls=0;max_direction_result_gap=0.;max_profile_kkt=0.;profile_checks=0;max_profile_improvement=0.;weak_kkt=[]
    def await_row(case):
        while True:
            c=sqlite3.connect(d.OUT/'records.sqlite3');found=c.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();c.close()
            if found:return json.loads(found[0])
            time.sleep(2)
    for i,case in enumerate(expected):
        row=await_row(case);assert row['case']==case;rows.append(row);g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=d.old.base.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'] and params==row['generating_parameters'];s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));assert int(s.sum())==row['scored_events'];context=donor_features(ds,de,1.5);assert {k:context[k] for k in ['primary','secondary','two_pulse_loglr']}==row['context'];fit=fit_null(s,e,1.5);assert fit['theta'].tolist()==row['null_fit'];observed=d.old.statistics(s,e,fit,context,p['settings']);assert observed['fits']==row['observed_fits'];point=estimate(s,e,1.5,fit);assert point==row['one_step_location'];independent,ll,n,kkt=independent_point(s,e,fit,point);gap=float(max(np.abs(independent-np.asarray(point['theta']))));max_direction_result_gap=max(max_direction_result_gap,gap);max_profile_kkt=max(max_profile_kkt,kkt);assert gap<2e-5,(case['id'],'Fisher direction/line search discrepancy',gap);assert abs(ll-point['loglik'])<1e-6;assert point['loglik']>=point['initial_loglik']-1e-10
        if kkt>1e-5:weak_kkt.append(dict(case=case['id'],scaled_kkt=kkt))
        for m in d.old.METHODS:
            v=row['samples'][m];assert {k:v[k] for k in ['score','primary','delay']}==observed['samples'][m];b=v['bootstrap_scores'];assert len(b)==199 and np.all(np.isfinite(b));assert (1+sum(z>=v['score'] for z in b))/200==v['pvalue'];rankchecks+=1
        src=row['samples']['context_transport']
        for m in ['transport_score_location','transport_one_step_location']:
            for key in ['score','pvalue','bootstrap_scores']:assert row['samples'][m][key]==src[key]
            for alpha in [.005,.01,.02]:assert (row['samples'][m]['pvalue']<=alpha)==(src['pvalue']<=alpha);equalcalls+=1
        assert [row['samples']['transport_one_step_location'][k] for k in ['primary','delay']]==[point[k] for k in ['primary','delay']]
        assert [row['samples']['transport_score_location'][k] for k in ['primary','delay']]==[row['samples']['local_score'][k] for k in ['primary','delay']]
        if case['id'].endswith('_000'):
            replay=d.worker(case,p);exclude=['seconds','location_seconds'];assert {k:v for k,v in replay.items() if k not in exclude}=={k:v for k,v in row.items() if k not in exclude};whole+=1
            # Independent SLSQP concave fixed-center amplitude profile check.
            theta=np.asarray(point['initial_theta']);f,_=pulse(theta[1],64,1.5);gg,_=pulse(theta[4],64,1.5);design=np.column_stack([f,np.ones(64)/64,gg]);scale=max(1.,sum(s))
            def objective(x):
                rate=np.maximum(design@x,1e-14);return float(-(s@np.log(rate)-e@rate)/scale),-design.T@(s/rate-e)/scale
            result=minimize(objective,theta[[0,2,3]],jac=True,method='SLSQP',bounds=[(0,20),(1e-6,30),(0,20)],options=dict(ftol=1e-13,maxiter=1000));improvement=-result.fun*scale-point['initial_loglik'];max_profile_improvement=max(max_profile_improvement,float(improvement));profile_checks+=1;assert improvement<1e-5,(case['id'],'Amplitude profile not optimized',improvement)
        if (i+1)%64==0:print('One-step independent audit',i+1,'/640;fullworkers',whole,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(1)
    while True:
        try:a=json.loads((d.OUT/'analysis.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1)
    assert d.read()==rows;d.check();complete=json.loads((d.OUT/'completion.json').read_text());assert complete['cases']==640 and complete['protocol_sha256']==d.sha(d.P) and complete['records_sha256']==d.sha(d.OUT/'records.sqlite3');before={name:d.sha(d.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']};totals={};group={}
    for alpha in [.005,.01,.02]:
        t={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS}
        for row in rows:
            key=(row['case']['regime'],row['case']['family']);group.setdefault(key,{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS})
            for m in d.METHODS:
                for k,v in endpoint(row,m,alpha).items():
                    t[m][k]+=v
                    if alpha==.01:group[key][m][k]+=v
        assert t==a['descriptive_operating_points'][str(alpha)];totals[str(alpha)]=t
    assert totals['0.01']==a['totals']
    for saved in a['per_regime_family']:
        gi=next(i for i,g in enumerate(p['regimes']) if g['name']==saved['regime']);assert saved['methods']==group[(gi,saved['family'])]
    paired={}
    for control in ['context_transport','transport_score_location','local_joint','local_score','context_max']:
        win=lose=fix=new=0
        for row in rows:
            x=endpoint(row,'transport_one_step_location',.01);b=endpoint(row,control,.01);win+=int(x['correct']>b['correct']);lose+=int(x['correct']<b['correct']);fix+=int(x['wrong']+x['false']<b['wrong']+b['false']);new+=int(x['wrong']+x['false']>b['wrong']+b['false'])
        paired[control]=dict(correct_gains=win,correct_losses=lose,error_fixes=fix,new_errors=new,correct_mcnemar_exact_p=binomtest(win,win+lose).pvalue if win+lose else 1.,error_mcnemar_exact_p=binomtest(fix,fix+new).pvalue if fix+new else 1.)
    # Independently verify each comparison behind the predeclared primary gate.
    t=totals['0.01'];x=t['transport_one_step_location'];same={m:dict(strict_more_correct=x['correct']>t[m]['correct'],strict_fewer_wrong=x['wrong']<t[m]['wrong'],identical_null=x['false']==t[m]['false']) for m in ['context_transport','transport_score_location']};assert same==a['primary_rule']['identical_call_location'];assert d.rule(t)==a['primary_rule'];assert a['continue_to_fresh_confirmation']==a['primary_rule']['pass_all'];assert all(d.sha(d.OUT/name)==hh for name,hh in before.items())
    result=dict(status='PASS',all_observed_query_context_datasets_replayed=640,all_observed_6method_statistics_locations_recomputed=True,all_independent_weighted_design_Fisher_steps_checked=640,max_independent_point_parameter_gap=max_direction_result_gap,max_fixed_center_scaled_KKT_residual=max_profile_kkt,weak_KKT_cases=weak_kkt,independent_SLSQP_profile_checks=profile_checks,max_reference_profile_loglik_improvement=max_profile_improvement,all_original4method_rankpvalues_checked=rankchecks,identical_call_checks=equalcalls,all_operating_point_endpoints_recounted=640*6*3,whole_workers_replayed=whole,bootstrap_paths_replayed=whole*199,paired=paired,outputs_sha256=before,seconds=time.perf_counter()-start,scope='All640observations/contextsets/pointestimates and2560originalrankpvalues;3840identicalcallchecks;11520endpointdecisions.28completeworkers5572bootstrap paths. Independent weighted-design leastsquares Fisher directions and28SLSQPamplitudeprofiles checked. Not replay ofall127360bootstrap replicates;not physical/composite-null/efficacy certification.')
    (d.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
