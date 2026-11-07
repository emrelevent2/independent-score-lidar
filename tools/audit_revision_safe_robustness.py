"""All original-censored mismatch records, including support-invalid full paths."""
import json,sqlite3,time
import numpy as np
import revision_safe_robustness as c
from audit_revision_sensitivity import wait_row
from one_step_location_failure_audit import stronger_null

def main():
    p=c.check();start=time.perf_counter();n=ranks=invalid=draws=0;changes=[];warnings=[]
    for case in c.r.cases(p,'robust'):
        a=wait_row(c.OUT/'records.sqlite3',case);b=wait_row(c.r.OUT/'robust.sqlite3',case)
        assert a['case']==b['case'] and a['truth']==b['truth'] and a['data_sha256']==b['data_sha256'];n+=1
        obs,_,_=c.r.acquisition(case,p);outside=bool(len(obs) and (obs[0]<0 or obs[-1]>=64000))
        assert outside==(a['status']=='TIMESTAMP_DOMAIN_INCOMPATIBLE')
        if outside:invalid+=1;continue
        s,e=c.r.numeric_statistics(obs,1000,64,c.r.outward_safe_lag(31.7,80.));fit=c.r.fit_null(s,e,1.5)
        assert c.r.d.json_fit(fit)==a['null_fit'];z=a['samples']['safe_original']
        value=c.r.fine_scan(s,e,1.5,fit=fit);assert value['score']==z['score']
        loc=c.r.localize(s,e,1.5,fit,value['delay']);assert loc==a['localizer']
        if not fit['optimizer_success'] or not loc['optimizer_success']:warnings.append(case['id'])
        assert len(z['bootstrap_scores'])==199 and z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200;ranks+=1
        boot=list(z['bootstrap_scores']);rate,_=c.r.rates_and_design(fit['theta'],64,1.5)
        for draw in a['warning_draws']:
            bo,_=c.r.simulate_two_dead_times(rate,1000,31.7,80.,p['bootstrap_seed_base']+1000003*case['index']+1009*draw)
            bs,be=c.r.numeric_statistics(bo,1000,64,c.r.outward_safe_lag(31.7,80.));bf=c.r.refit_from_start(bs,be,fit['theta'],1.5,'duration')
            assert c.r.fine_scan(bs,be,1.5,fit=bf)['score']==boot[draw]
            better=stronger_null(bs,be,bf['theta'],bf);boot[draw]=c.r.fine_scan(bs,be,1.5,fit=better)['score'];draws+=1
        altp=c.r.pvalue(z['score'],boot)
        for alpha in [.005,.01,.02]:
            if (altp<=alpha)!=(z['pvalue']<=alpha):changes.append(dict(case=case['id'],alpha=alpha,original_p=z['pvalue'],stricter_p=altp))
        if n%64==0:print('Original censored robustness audit',n,'/864',flush=True)
    assert n==864
    for f in [c.OUT/'completion.json',c.r.OUT/'robust_completion.json']:
        while not f.exists():time.sleep(2)
    result=dict(status='PASS' if not changes else 'MATERIAL_NUMERICAL_SENSITIVITY',cases=n,acquisition_hash_matches=n,domain_incompatible=invalid,observed_safe_score_and_localizer_replays=ranks,rank_checks=ranks,warning_draws_checked=draws,observed_optimizer_warnings=warnings,operating_point_changes=changes,records_sha256=c.r.sha(c.OUT/'records.sqlite3'),main_robust_records_sha256=c.r.sha(c.r.OUT/'robust.sqlite3'),protocol_sha256=c.r.sha(c.P),auditor_sha256=c.r.sha(__file__),seconds=time.perf_counter()-start)
    (c.OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
