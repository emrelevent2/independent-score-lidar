"""Independent recount, full observed replay, and preselected whole-worker audit.

This auditor does not change any frozen scientific source or experimental record.
Whole-bootstrap workers are selected by family/regime index000, not outcomes.
"""
import json,time,hashlib,sqlite3
import numpy as np
from scipy.stats import binomtest
import context_agreement_development as d
from context_agreement_features import numeric_statistics,donor_features,pilot_location
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start
from safe_deadtime_exposure import simulate_two_dead_times
from deadtime_exposure import periodic_measure


def distance(a,b,k):
    return abs((a-b+k/2)%k-k/2)


def independent_endpoint(row,method):
    z=row['samples'][method];positive=row['case']['positive'];declared=z['pvalue']<=.01
    correct=positive and declared and distance(z['primary'],row['truth']['primary'],64)<=2 and distance(z['delay'],row['truth']['secondary'],64)<=2
    return dict(correct=int(correct),wrong=int(positive and declared and not correct),false=int(not positive and declared),positive=int(positive),null=int(not positive))


def interval(n,total):
    ci=binomtest(n,total).proportion_ci(confidence_level=.95,method='exact')
    return dict(count=n,denominator=total,rate=n/total,exact95=[ci.low,ci.high])


def main():
    start=time.perf_counter();p=d.check();cfg=p['settings'];expected=list(d.cases(p));rows=[]
    totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS};group={};whole=0;strong=0;largest_gap=0.;events=0;boot_ranks=0;zero=0;max_exposure_gap=0.;safe_intervals=0
    # Read committed rows only; scientific job remains the sole record writer.
    # Recheck every audited row against the final immutable database below.
    def await_row(case):
        while True:
            connection=sqlite3.connect(d.OUT/'records.sqlite3');found=connection.execute('SELECT payload FROM records WHERE id=?',(case['id'],)).fetchone();connection.close()
            if found:return json.loads(found[0])
            time.sleep(2)
    for i,expected_case in enumerate(expected):
        row=await_row(expected_case);assert row['case']==expected_case;rows.append(row)
        case=row['case'];g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=d.data(case,p)
        assert truth==row['truth'] and params==row['generating_parameters'] and dh==row['data_sha256'] and len(obs)==row['observed_events'];events+=len(obs)
        lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],cfg['bins'],lag);assert int(s.sum())==row['scored_events']
        # Independent direct floating interval integrator, checked against exact
        # dyadic reconstruction; tolerance covers this diagnostic's own arithmetic.
        lo=np.r_[0.,obs+lag];hi=np.r_[obs,g['cycles']*64];keep=hi>lo;direct=periodic_measure(lo[keep],hi[keep],64);delta=float(np.max(np.abs(direct-e)));max_exposure_gap=max(max_exposure_gap,delta);assert delta<2e-8
        eligible=obs>=lo[:-1];direct_counts=np.bincount(np.floor(obs[eligible]%64).astype(int),minlength=64);np.testing.assert_array_equal(direct_counts,s)
        a,t,b,c,second=params[0];rates=a*pulse(t,64,1.5)[0]+b/64+c*pulse(second,64,1.5)[0];recorded,latent=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['seed']+1000003);np.testing.assert_array_equal(recorded,obs)
        left=lo[keep];idx=np.searchsorted(latent,left,side='left')-1;valid=idx>=0;assert np.all(latent[idx[valid]]+g['td']<=left[valid]+2e-8);safe_intervals+=len(left)
        context=donor_features(ds,de,cfg['sigma']);saved=dict(primary=context['primary'],secondary=context['secondary'],two_pulse_loglr=context['two_pulse_loglr'],initial_theta=context['initial_theta'].tolist());assert saved==row['context']
        fit=fit_null(s,e,cfg['sigma']);assert fit['theta'].tolist()==row['null_fit'];pilot,count=pilot_location(obs,int(g['cycles']*.25),64,lag,cfg['sigma'],context);assert pilot==row['pilot_delay'] and count==row['pilot_scored_events']
        scores,locations=d.statistic_set(s,e,fit,pilot,context,cfg)
        # Independently evaluate the declared heuristic on observed queries.
        z=row['samples']['local'];strength=-np.expm1(-max(context['two_pulse_loglr']-20,0)/20);bonus=2*np.log(64/(2*np.sqrt(2*np.log(2))*1.5));agreement=np.exp(-distance(z['delay'],context['secondary'],64)**2/(2*1.5**2));early=np.exp(-distance(pilot,context['secondary'],64)**2/(8*1.5**2))
        np.testing.assert_allclose(scores['full_supported'],z['score']+bonus*strength*agreement,rtol=1e-14,atol=1e-13);np.testing.assert_allclose(scores['pilot_supported'],z['score']+bonus*strength*agreement*early,rtol=1e-14,atol=1e-13)
        key=(case['regime'],case['family']);group.setdefault(key,{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS})
        for m in d.METHODS:
            v=row['samples'][m];assert scores[m]==v['score'] and locations[m]==[v['primary'],v['delay']];b=np.asarray(v['bootstrap_scores']);assert len(b)==199 and np.all(np.isfinite(b));assert (1+sum(float(t)>=v['score'] for t in b))/200==v['pvalue'];boot_ranks+=1;ep=independent_endpoint(row,m)
            for name,n in ep.items():totals[m][name]+=n;group[key][m][name]+=n
        assert len(row['bootstrap_pilot_delays'])==199
        if strength==0:
            zero+=1
            assert row['samples']['local']==row['samples']['full_supported']==row['samples']['pilot_supported']
        if case['id'].endswith('_000'):
            replay=d.worker(case,p);assert {k:v for k,v in replay.items() if k!='seconds'}=={k:v for k,v in row.items() if k!='seconds'};whole+=1
            rates=fit['theta'][0]*pulse(fit['theta'][1],64,1.5)[0]+fit['theta'][2]/64
            for draw in [0,198]:
                bo,_=simulate_two_dead_times(rates,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);bs,be=numeric_statistics(bo,g['cycles'],64,lag);warm=refit_from_start(bs,be,fit['theta'],1.5,'duration');multi=fit_null(bs,be,1.5);gap=multi['loglik']-warm['loglik'];largest_gap=max(largest_gap,gap);strong+=1
                assert gap<1e-5,('Strong fit improves warm fit materially',case['id'],draw,gap)
        if (i+1)%64==0:print('Audited observed contexts and endpoints',i+1,'/640; whole199-path workers',whole,flush=True)
    while not (d.OUT/'completion.json').exists():time.sleep(2)
    assert rows==d.read();d.check()
    complete=json.loads((d.OUT/'completion.json').read_text());assert complete['status']=='COMPLETE' and complete['cases']==640 and complete['protocol_sha256']==d.sha(d.P)
    if not (d.OUT/'analysis.json').exists():d.analyze()
    before={name:d.sha(d.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']};assert complete['records_sha256']==before['records.sqlite3']
    analysis=json.loads((d.OUT/'analysis.json').read_text());assert totals==analysis['totals'] and zero==analysis['zero_context_support_cases']
    for item in analysis['per_regime_family']:
        gi=next(i for i,g in enumerate(p['regimes']) if g['name']==item['regime']);assert group[(gi,item['family'])]==item['methods']
    x,l,b=[totals[m] for m in ['pilot_supported','local','spatial_prior']]
    rule=dict(strict_more_correct_than_local=x['correct']>l['correct'],strict_fewer_errors_than_local=x['wrong']+x['false']<l['wrong']+l['false'],no_more_wrong_than_local=x['wrong']<=l['wrong'],no_more_null_than_local=x['false']<=l['false'],at_least_spatial_correct=x['correct']>=b['correct'],no_more_spatial_wrong=x['wrong']<=b['wrong'],no_more_spatial_null=x['false']<=b['false'],at_least_one_strict_spatial=x['correct']>b['correct'] or x['wrong']<b['wrong'] or x['false']<b['false'])
    assert rule==analysis['primary_rule'] and all(rule.values())==analysis['continue_to_fresh_confirmation']
    paired={}
    for control in ['local','spatial_prior']:
        wins=losses=fixes=new_errors=0
        for row in rows:
            a=independent_endpoint(row,'pilot_supported');b=independent_endpoint(row,control);wins+=int(a['correct']>b['correct']);losses+=int(a['correct']<b['correct']);ae=a['wrong']+a['false'];be=b['wrong']+b['false'];fixes+=int(ae<be);new_errors+=int(ae>be)
        paired[control]=dict(correct_gains=wins,correct_losses=losses,error_fixes=fixes,new_errors=new_errors,correct_mcnemar_exact_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.,error_mcnemar_exact_p=binomtest(fixes,fixes+new_errors).pvalue if fixes+new_errors else 1.)
    assert all(d.sha(d.OUT/name)==h for name,h in before.items())
    r=dict(status='PASS',all_observed_queries_and_16_context_paths_regenerated=len(rows),all_observed_statistics_recomputed=True,all_query_safe_exposures_directly_integrated=True,max_direct_exposure_gap=max_exposure_gap,all_query_safe_interval_starts_checked_against_latent_recovery=safe_intervals,latent_recovery_diagnostic_tolerance=2e-8,all_stored_bootstrap_pvalues_independently_reranked=boot_ranks,whole_199_path_workers_replayed=whole,bootstrap_paths_replayed=whole*199,preselected_multistart_checks=strong,largest_multistart_loglik_improvement=largest_gap,query_events=events,zero_context_support_cases=zero,totals=totals,paired=paired,rate_intervals={m:dict(correct=interval(v['correct'],256),wrong=interval(v['wrong'],256),false=interval(v['false'],384)) for m,v in totals.items()},primary_rule=rule,source_output_sha256=before,seconds=time.perf_counter()-start,scope='All640observations/contexts/statistics/endpoints and2560rankpvalues independently recounted;28outcome-independent complete workers(5572bootstrap paths) exactly replayed;56stronger null-fit checks. Not a replay of all127360bootstrap fits. Development only;plug-in composite-null calibration remains approximate;not efficacy confirmation.')
    (d.OUT/'integrity.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
