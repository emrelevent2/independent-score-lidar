"""Regenerate every confirmation predictor/numerator; separate endpoint audit."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,sqlite3,time
import numpy as np
from scipy.stats import beta,binomtest
from flint import arb
import shoulder_confirmation as study
from guided_alternatives import guided_pool,fit_pool
from coverage_alternative import coverage_components
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from calibration_confidence_v2 import lobe_confidence
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float,continuous_null_bound
from shoulder_envelope import shoulder_null_bound


def verify(r):
    h,cal,truth=study.base.simulate_case(r['case']);query=h[4,4]
    assert truth==r['truth'] and query.tolist()==r['query']
    assert hashlib.sha256(h.tobytes()).hexdigest()==r['raw_histogram_sha256']
    assert cal[2].tolist()==r['calibration_histogram']
    confidence=lobe_confidence(cal[2],100000);assert confidence==r['confidence'];cap=confidence['eta_upper']
    pool,meta=guided_pool(h,4,4);old=fit_pool(pool);old.update(meta)
    uncal=donor_proposal(h);calibrated=donor_proposal(h,cap=cap)
    proposals=dict(old_gaussian=old,new_uncalibrated=uncal,cal_100000=calibrated)
    assert proposals==r['proposals']
    om=coverage_components(old);om['spatial']=np.arange(len(om['rates']))<4
    numerators=dict(old_gaussian=numeric_and_sparse(query,om),new_uncalibrated=numeric_and_sparse(query,components(uncal)),
                    cal_100000=numeric_and_sparse(query,components(calibrated,cap=cap)))
    assert numerators==r['numerators']
    witnesses=0;refits=0
    if r['case']['secondary']==0:
        t=r['truth'];c=r['case'];g=exact_gaussian(256,t['primary'],t['width'],False)
        side=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
        a=arb(c['primary'])/sum(g,arb(0))
        rates=[(a*(x+arb(c['eta'])*y)+arb(c['background'])/256)/1000 for x,y in zip(g,side)]
        ll=certified_log_likelihood(query,rates,1000,'first',upper=True)
        for method in study.METHODS:
            applicable=(c['eta']==0 if method=='old_gaussian' else True if method=='new_uncalibrated' else c['eta']<=cap)
            if applicable:assert ll<=r['bounds'][method]['upper'];witnesses+=1
    for method in study.METHODS:
        num=numerators[method];bound=r['bounds'][method];report=r['reports'][method]
        threshold=outward_float((arb(1000)/9).log() if method=='cal_100000' else arb(100).log())
        score=outward_float(arb(num['lower'])-arb(bound['upper']),False)
        assert report==dict(declared=score>=threshold,log_e_lower=score,log_threshold_upper=threshold,
            primary=proposals[method]['primary'],secondary=num['location'],head=num['head'],
            numerator_lower=num['lower'],null_upper=bound['upper'])
    if r['id'].endswith(('_0000','_0191')):
        b=continuous_null_bound(query,fwhm_range=(4.5,5.5),initial_cells=32,max_boxes=256,normalized_envelope=False)
        assert b==r['bounds']['old_gaussian'];refits+=1
        for method,limit in [('new_uncalibrated',.25),('cal_100000',cap)]:
            threshold=r['reports'][method]['log_threshold_upper']
            level=outward_float(arb(numerators[method]['lower'])-arb(threshold),False)
            assert shoulder_null_bound(query,cap=limit,decision_level=level)==r['bounds'][method];refits+=1
    return witnesses,refits


def main():
    start=time.perf_counter();p=study.check_protocol();OUT=study.OUT;sha=study.base.sha
    db=sqlite3.connect(OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={name:sha(OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    witnesses=refits=done=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,r) for r in rows]):
            w,b=f.result();witnesses+=w;refits+=b;done+=1
            if done%96==0:print('Audited confirmation',done,'of',len(rows),flush=True)
    a=json.loads((OUT/'analysis.json').read_text());endpoints={}
    def good(r,method):
        s=r['reports'][method]
        return bool(r['case']['secondary']>0 and s['declared'] and abs(s['primary']-r['truth']['primary'])<=2 and abs(s['secondary']-r['truth']['secondary'])<=2)
    for method in study.METHODS:
        endpoints[method]={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==group]
            e=dict(correct=sum(good(r,method) for r in rs),
                wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][method]['declared'] and not good(r,method) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][method]['declared'] for r in rs),
                declared=sum(r['reports'][method]['declared'] for r in rs),n=len(rs))
            assert e==a['methods'][method][group];endpoints[method][group]=e
    totals={m:dict(correct=sum(endpoints[m][g]['correct'] for g in ['near_two','far_two']),
                   wrong_location=sum(endpoints[m][g]['wrong_positive_location'] for g in ['near_two','far_two']),
                   false_objects=sum(endpoints[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])) for m in study.METHODS}
    assert totals==a['totals']
    positives=[r for r in rows if r['case']['secondary']>0]
    rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates'];boot=np.zeros(rep)
    for group in ['far_two','near_two']:
        for stratum in range(24):
            d=np.array([int(good(r,'cal_100000'))-int(good(r,'old_gaussian')) for r in positives if r['case']['group']==group and r['case']['stratum']==stratum])
            assert len(d)==8;boot+=rng.choice(d,size=(rep,8),replace=True).sum(1)/len(positives)
    ci=np.quantile(boot,[.025,.975]).tolist();assert ci==a['paired_correct']['stratified_percentile_95_ci']
    gains=sum(good(r,'cal_100000') and not good(r,'old_gaussian') for r in positives)
    losses=sum(good(r,'old_gaussian') and not good(r,'cal_100000') for r in positives)
    assert (gains,losses)==(a['paired_correct']['gains'],a['paired_correct']['losses'])
    assert (gains-losses)/len(positives)==a['paired_correct']['rate_difference']
    exact_p=float(binomtest(gains,gains+losses,.5,alternative='greater').pvalue) if gains+losses else 1.
    assert exact_p==a['paired_correct']['exact_one_sided_mcnemar_p']
    primary='cal_100000';control='old_gaussian'
    success=dict(paired_correct_gain=ci[0]>0,false_object_reduction=totals[primary]['false_objects']<totals[control]['false_objects'],
        wrong_location_reduction=totals[primary]['wrong_location']<totals[control]['wrong_location'])
    assert success==a['success'] and all(success.values())==a['all_prespecified_success_criteria']
    assert a['calibration']==dict(detected_photons_median=float(np.median([r['confidence']['detected'] for r in rows])),
        informative=sum(r['confidence']['informative'] for r in rows),realized_undercoverage=sum(r['confidence']['eta_upper']<r['case']['eta'] for r in rows))
    f=totals['cal_100000']['false_objects']
    assert a['null_empirical_95_one_sided_upper']==(float(beta.ppf(.95,f+1,384-f)) if f<384 else 1.)
    assert all(sha(OUT/name)==value for name,value in snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(study.P),records_regenerated=done,
        applicable_exact_null_witness_checks=witnesses,complete_cover_refits=refits,
        group_endpoint_recounts=12,paired_bootstrap_reproduced=True,output_sha256=snapshot,
        seconds=time.perf_counter()-start,
        scope='Every raw data/calibration/donor/proper sparse numerator/decision/location regenerated,all endpoints and primary CI separately recounted,eight fixed full-cover examples. Not sensor-law validation or SOTA comparison.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
