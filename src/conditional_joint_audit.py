"""All pilot refits, direct physical-null witnesses and independent endpoints."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from flint import arb
import conditional_joint_development as s
from envelope_evidence import exact_gaussian,certified_log_likelihood


def verify(pair,saved):
    actual=s.worker(pair);actual.pop('elapsed_seconds');expected=dict(saved);expected.pop('elapsed_seconds');assert actual==expected
    c=saved['case'];t=saved['truth'];compatible=0;witnesses=0
    g=exact_gaussian(256,t['calibration_delay'],t['calibration_width'],False)
    a=sum(g[27:38],arb(0));forward=sum(g[43:54],arb(0))/a
    signal=arb(.01)*a;beta=arb(.005)/256
    rate=signal*(forward+arb(c['eta']))+11*beta
    for fold in saved['folds']:
        cf=fold['nuisance_confidence']
        good=not cf['informative'] or (arb(cf['signal_main_lower'])<=signal<=arb(cf['signal_main_upper']) and
            arb(cf['background_lower'])<=beta<=arb(cf['background_upper']) and
            arb(cf['forward_leakage_lower'])<=forward<=arb(cf['forward_leakage_upper']))
        compatible+=good
        if good and c['secondary']==0:
            k,r=fold['test_lobe_count'],fold['test_lobe_risk'];ll=arb(0)
            if k:ll+=k*(-(-rate).expm1()).log()
            ll-=(r-k)*rate
            qg=exact_gaussian(256,t['primary'],t['width'],False)
            qs=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
            amp=arb(c['primary'])/sum(qg,arb(0))
            qr=[(amp*(x+arb(c['eta'])*y)+arb(c['background'])/256)/1000 for x,y in zip(qg,qs)]
            h,_,_=s.base.original.simulate_case(c)
            ll+=arb(certified_log_likelihood(h[4,4],qr,1000,'first',upper=True))
            assert arb(fold['bound']['upper'])>=ll;witnesses+=1
    return compatible,witnesses


def main():
    start=time.perf_counter();p=s.check_protocol();sha=s.base.original.sha
    rows=s.base.read_records(s.OUT);parents={r['id']:r for r in s.base.read_records(s.base.OUT)}
    originals={r['id']:r for r in s.base.read_records(s.base.original.OUT)}
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={name:sha(s.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    done=compatible=witnesses=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,(parents[r['id']],originals[r['id']]),r) for r in rows]):
            a,b=f.result();compatible+=a;witnesses+=b;done+=1
            if done%8==0:print('Audited coupled pilot',done,'of',len(rows),flush=True)
    a=json.loads((s.OUT/'analysis.json').read_text());totals={}
    def good(r,m):
        x=r['reports'][m];t=r['truth']
        return r['case']['secondary']>0 and x['declared'] and abs(x['primary']-t['primary'])<=2 and abs(x['secondary']-t['secondary'])<=2
    for m in s.METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==group]
            v=dict(correct=sum(good(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not good(r,m) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert v==a['groups'][m][group];groups[group]=v
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals'];j,b=totals['joint'],totals['background_ci_100000']
    rule=dict(more_correct=j['correct']>b['correct'],no_more_wrong=j['wrong']<=b['wrong'],no_more_false=j['false']<=b['false'])
    assert rule==a['primary_rule'] and all(rule.values())==a['continue_to_expanded_development']
    folds=[f for r in rows for f in r['folds']]
    assert a['computation']==dict(total_boxes=sum(f['bound']['boxes_evaluated'] for f in folds),median_boxes=float(np.median([f['bound']['boxes_evaluated'] for f in folds])),median_cal_predictive_minus_saturated=float(np.median([f['calibration_predictive_minus_saturated'] for f in folds])))
    assert all(sha(s.OUT/name)==digest for name,digest in snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(s.P),records_fully_refit=done,training_confidence_boxes_containing_physical_truth=compatible,total_training_boxes=len(folds),
        applicable_physical_null_joint_witness_checks=witnesses,group_endpoint_recounts=20,output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All splits/predictors/conditional calibration densities/CI endpoints/full joint covers/averages/locations and endpoints checked. Error guarantee is marginal over calibration/query/split noise,not conditional on the entire realized calibration.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
