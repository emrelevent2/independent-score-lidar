"""Streaming, read-only replay of all new acquisitions and observed fits.

Every recorded bootstrap warning is regenerated and checked with stricter
independent objectives. Original scientific rows remain unchanged.
"""
import json,sqlite3,time,sys,hashlib
from pathlib import Path
import numpy as np
import revision_sensitivity as rev
import ct_dual_reference as reference
import ct_dual_reverse as reverse
from ct_full_score import projected
from null_bank_failure_audit import refine_full
from one_step_location_failure_audit import stronger_null
from ct_full_score_failure_audit import refine_full_null as refine_generator

def wait_row(path,case):
    while True:
        try:
            with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True) as db:r=db.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone()
            if r:return json.loads(r[0])
        except sqlite3.OperationalError:pass
        time.sleep(2)

def main(kind):
    p=rev.check();g=p['regime'];path=rev.OUT/(kind+'.sqlite3');start=time.perf_counter();n=ranks=incompatible=draws=quadratures=0;largest=0.;changes=[];refinements=[];generators=[];localizer_warnings=[]
    for case in rev.cases(p,kind):
        row=wait_row(path,case);assert row['case']==case
        obs,truth,theta=rev.acquisition(case,p);assert truth==row['truth'] and theta==row['generating_parameters'] and hashlib.sha256(obs.tobytes()).hexdigest()==row['data_sha256']
        invalid=bool(np.any(np.diff(obs)<80) or (len(obs) and (obs[0]<0 or obs[-1]>=64000)))
        assert invalid==(row['status']=='HARD_SUPPORT_INCOMPATIBLE');n+=1
        if invalid:incompatible+=1;continue
        lag=rev.outward_safe_lag(g['td'],g['te']);s,e=rev.numeric_statistics(obs,1000,64,lag);sf=rev.fit_null(s,e,1.5)
        assert rev.d.json_fit(sf)==row['nulls']['safe']
        geo=rev.kernel.prepare_record(obs,64000,64,31.7,80.)
        fv,fn,_=rev.primary_statistic(s,e,1.5,sf,geo,rev.d.bank(),2);assert rev.d.json_fit(fn)==row['nulls']['full']
        sv=rev.fine_scan(s,e,1.5,fit=sf);assert fv['score']==row['samples']['full']['score'] and sv['score']==row['samples']['safe']['score']
        rg=reference.prepare_record(obs,64000,64,31.7,80.)
        for m,loc in row['localizers'].items():
            rate,j=rev.rates_and_design(loc['theta'],64,1.5)
            ll=reference.full_factor(rate,j,rg,32)[0] if m=='full' else s@np.log(rate)-e@rate
            gap=abs(float(ll)-loc['loglik']);largest=max(largest,gap);assert gap<2e-6;quadratures+=m=='full'
            if not loc['optimizer_success']:localizer_warnings.append(dict(case=case['id'],method=m))
        for m,z in row['samples'].items():
            assert len(z['bootstrap_scores'])==199 and np.all(np.isfinite(z['bootstrap_scores']))
            assert z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200;ranks+=1
        cal=row['nulls']['calibration']; theta0=cal['theta'];rate,_=rev.rates_and_design(theta0,64,1.5)
        if cal.get('failures',0) or not cal['optimizer_success']:
            generators.append(dict(case=case['id'],**refine_generator(obs,g,sf,cal)))
        corrected={m:list(z['bootstrap_scores']) for m,z in row['samples'].items()}
        for draw in row['warning_draws']:
            bo,_=rev.simulate_two_dead_times(rate,1000,31.7,80.,p['bootstrap_seed_base']+1000003*case['index']+1009*draw)
            bs,be=rev.numeric_statistics(bo,1000,64,lag);bf=rev.refit_from_start(bs,be,theta0,1.5,'duration')
            bg=rev.kernel.prepare_record(bo,64000,64,31.7,80.);v,bfn,_=rev.primary_statistic(bs,be,1.5,bf,bg,rev.d.bank(),2)
            assert v['score']==row['samples']['full']['bootstrap_scores'][draw]
            assert rev.fine_scan(bs,be,1.5,fit=bf)['score']==row['samples']['safe']['bootstrap_scores'][draw]
            safe=stronger_null(bs,be,bf['theta'],bf) if not bf['optimizer_success'] else bf
            full=rev.primary_statistic(bs,be,1.5,safe,bg,rev.d.bank(),2)[1]
            if not full['optimizer_success']:
                full,record=refine_full(bs,safe,full,bo,g);refinements.append(dict(case=case['id'],draw=draw,**record))
            rr,j=rev.rates_and_design(full['theta'],64,1.5);gradient=reverse.full_factor(rr,np.eye(64),bg)[1];h,_=rev.d.bank().lookup(full['theta'],2)
            corrected['full'][draw]=projected(gradient,h,j,full['theta'],1.5)['score']
            corrected['safe'][draw]=rev.fine_scan(bs,be,1.5,fit=safe)['score'];draws+=1
        for m,z in row['samples'].items():
            altp=rev.pvalue(z['score'],corrected[m])
            for a in [.005,.01,.02]:
                if (z['pvalue']<=a)!=(altp<=a):changes.append(dict(case=case['id'],method=m,alpha=a,original_p=z['pvalue'],stricter_p=altp))
        if n%32==0:print(kind,'audit',n,flush=True)
    completion=rev.OUT/(kind+'_completion.json')
    while not completion.exists():time.sleep(2)
    saved=json.loads(completion.read_text());assert saved['records_sha256']==rev.sha(path);rev.check()
    result=dict(status='PASS' if not changes else 'MATERIAL_NUMERICAL_SENSITIVITY',cases=n,acquisition_hashes=n,incompatible_acquisitions=incompatible,observed_score_replays=2*(n-incompatible),rank_checks=ranks,independent_full_localizer_values=quadratures,max_localizer_loglik_gap=largest,warning_draws_checked=draws,strict_full_null_refinements=refinements,calibration_generator_checks=generators,localizer_warnings=localizer_warnings,operating_point_changes=changes,records_sha256=saved['records_sha256'],protocol_sha256=rev.sha(rev.P),auditor_sha256=rev.sha(__file__),seconds=time.perf_counter()-start)
    (rev.OUT/(kind+'_audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','cases','incompatible_acquisitions','rank_checks','warning_draws_checked','localizer_warnings','operating_point_changes']},indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])
