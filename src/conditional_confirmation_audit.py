"""All prospective records refit; independent endpoints and error-scope checks."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from flint import arb
import conditional_joint_confirmation as s
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float


def verify(row):
    actual=s.worker(row['case']);actual.pop('elapsed_seconds');expected=dict(row);expected.pop('elapsed_seconds');assert actual==expected
    c=row['case'];t=row['truth'];good_boxes=total_boxes=physical_witnesses=0
    cg=exact_gaussian(256,t['calibration_delay'],t['calibration_width'],False)
    A=sum(cg[27:38],arb(0));F=sum(cg[43:54],arb(0))/A;sig=arb(.01)*A;beta=arb(.005)/256
    calrate=sig*(F+arb(c['eta']))+11*beta
    if c['secondary']==0:
        qg=exact_gaussian(256,t['primary'],t['width'],False);qs=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
        amp=arb(c['primary'])/sum(qg,arb(0));rates=[(amp*(x+arb(c['eta'])*y)+arb(c['background'])/256)/1000 for x,y in zip(qg,qs)]
        query_true_upper=certified_log_likelihood(row['query'],rates,1000,'first',upper=True)
        ci=row['methods']['background_ci']
        if ci['bound'] is not None and row['confidence']['eta_upper']>=c['eta']:
            assert ci['bound']['upper']>=query_true_upper;physical_witnesses+=1
    for fold in row['methods']['joint']['folds']:
        cf=fold['nuisance_confidence'];total_boxes+=1
        compatible=not cf['informative'] or (arb(cf['signal_main_lower'])<=sig<=arb(cf['signal_main_upper']) and
            arb(cf['background_lower'])<=beta<=arb(cf['background_upper']) and arb(cf['forward_leakage_lower'])<=F<=arb(cf['forward_leakage_upper']))
        good_boxes+=compatible
        if compatible and c['secondary']==0 and 'bound' in fold:
            k,r=fold['test_lobe_count'],fold['test_lobe_risk'];ll=arb(0)
            if k:ll+=k*(-(-calrate).expm1()).log()
            ll-=(r-k)*calrate;ll+=arb(query_true_upper)
            assert arb(fold['bound']['upper'])>=ll;physical_witnesses+=1
    # Ratios are confidence restricted; these are padded p-value UPPERS,
    # not an assertion of ordinary unconditional e-values.
    for method in ['background_ci','joint']:
        rep=row['reports'][method];lower=rep['log_ratio_lower']
        p=arb(1) if lower is None else arb(1)/1000+(-arb(lower)).exp()
        p_upper=1. if not p<1 else min(1.,outward_float(p))
        if rep['declared']:assert arb(p_upper)<=arb(1)/100
    return good_boxes,total_boxes,physical_witnesses


def main():
    start=time.perf_counter();p=s.check_protocol();sha=s.bgdev.original.sha;rows=s.read_records()
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={name:sha(s.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    done=good=boxes=witnesses=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,r) for r in rows]):
            a,b,c=f.result();good+=a;boxes+=b;witnesses+=c;done+=1
            if done%96==0:print('Audited prospective coupled',done,'of',len(rows),flush=True)
    a=json.loads((s.OUT/'analysis.json').read_text());totals={}
    def correct(r,m):
        v=r['reports'][m];t=r['truth']
        return bool(r['case']['secondary']>0 and v['declared'] and abs(v['primary']-t['primary'])<=2 and abs(v['secondary']-t['secondary'])<=2)
    for method in s.METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==group]
            v=dict(correct=sum(correct(r,method) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][method]['declared'] and not correct(r,method) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][method]['declared'] for r in rs),declared=sum(r['reports'][method]['declared'] for r in rs),n=len(rs))
            assert v==a['groups'][method][group];groups[group]=v
        totals[method]=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals'];pos=[r for r in rows if r['case']['secondary']>0]
    gains=sum(correct(r,'joint') and not correct(r,'background_ci') for r in pos);losses=sum(correct(r,'background_ci') and not correct(r,'joint') for r in pos)
    rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates'];boot=np.zeros(rep)
    for group in ['far_two','near_two']:
        for stratum in range(24):
            d=np.array([int(correct(r,'joint'))-int(correct(r,'background_ci')) for r in pos if r['case']['group']==group and r['case']['stratum']==stratum]);assert len(d)==8
            boot+=rng.choice(d,size=(rep,8),replace=True).sum(1)/len(pos)
    ci=np.quantile(boot,[.025,.975]).tolist()
    assert a['paired']==dict(gains=gains,losses=losses,rate_difference=(gains-losses)/len(pos),stratified_percentile95_ci=ci)
    j,b=totals['joint'],totals['background_ci'];rule=dict(correct_gain=ci[0]>0,no_more_wrong=j['wrong']<=b['wrong'],no_more_false=j['false']<=b['false'])
    assert a['success_rule']==rule and a['all_prespecified_success_criteria']==all(rule.values())
    for m in s.METHODS:
        count=dict(screened=sum(r['methods'][m]['certified_screened_out'] for r in rows),
            boxes=sum(sum(f.get('bound',{}).get('boxes_evaluated',0) for f in r['methods'][m]['folds']) if m=='joint' else (r['methods'][m]['bound'] or {}).get('boxes_evaluated',0) for r in rows))
        assert count==a['computation'][m]
    assert a['median_detected_calibration_photons']==float(np.median([sum(r['calibration_histogram']) for r in rows]))
    assert all(sha(s.OUT/name)==digest for name,digest in snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(s.P),records_fully_refit=done,training_nuisance_boxes=boxes,
        boxes_containing_physical_truth=good,applicable_unscreened_physical_null_witness_checks=witnesses,
        group_endpoint_recounts=12,paired_bootstrap_reproduced=True,padded_pvalue_bounds_checked=True,
        output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All prospective data,calibration splits,MLEs,predictors,sparse bounds,screening,full covers,decisions,locations refit; all endpoints and primary criteria independently recounted. Failed primary is retained; not sensor validity or Q1 proof.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
