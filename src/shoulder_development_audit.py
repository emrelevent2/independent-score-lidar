"""Post-freeze verifier: regenerate donors/calibration and recount endpoints."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,sqlite3,time
import numpy as np
from flint import arb
from shoulder_development import (ROOT,P,OUT,METHODS,sha,check_protocol,simulate_case)
from guided_alternatives import guided_pool,fit_pool
from coverage_alternative import coverage_components
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from calibration_confidence import lobe_confidence
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float,continuous_null_bound
from shoulder_envelope import shoulder_null_bound


def verify_record(r):
    import hashlib
    h,cal,truth=simulate_case(r['case']);query=h[4,4]
    assert truth==r['truth'] and query.tolist()==r['query']
    assert hashlib.sha256(h.tobytes()).hexdigest()==r['raw_histogram_sha256']
    assert [x.tolist() for x in cal]==r['calibration_histograms']
    confidence=[lobe_confidence(x,n) for x,n in zip(cal,[1000,10000,100000])]
    assert confidence==r['confidence']
    pool,meta=guided_pool(h,4,4);old=fit_pool(pool);old.update(meta);new=donor_proposal(h)
    assert dict(old=old,new=new)==r['proposals']
    om=coverage_components(old);om['spatial']=np.arange(len(om['rates']))<4
    assert numeric_and_sparse(query,om)==r['numerators']['old']
    assert numeric_and_sparse(query,components(new))==r['numerators']['new']
    checked_null_witnesses=0
    # Exact physical-null likelihood is independently below every applicable
    # complete-cover upper. It cannot validate misspecified Gaussian controls.
    if r['case']['secondary']==0:
        g=exact_gaussian(256,truth['primary'],truth['width'],False)
        side=exact_gaussian(272,arb(truth['primary'])+16,truth['width'],False)[:256]
        total=sum(g,arb(0));a=arb(r['case']['primary'])/total
        rates=[(a*(x+arb(r['case']['eta'])*y)+arb(r['case']['background'])/256)/1000
               for x,y in zip(g,side)]
        ll=certified_log_likelihood(query,rates,1000,'first',upper=True)
        for cap,b in r['shoulder_bounds'].items():
            if float(cap)>=r['case']['eta']:
                assert ll<=b['upper'];checked_null_witnesses+=1
        if r['case']['eta']==0:assert ll<=r['gaussian_bound']['upper'];checked_null_witnesses+=1
    threshold=outward_float(arb(100).log());cal_threshold=outward_float((arb(1000)/9).log())
    for method in METHODS:
        a=r['reports'][method];num=r['numerators']['old' if method=='old_gaussian' else 'new']
        proposal=r['proposals']['old' if method=='old_gaussian' else 'new']
        if method.endswith('gaussian'):cap=0.;bound=r['gaussian_bound'];cut=threshold
        else:
            cap=.25 if method=='new_uncalibrated' else confidence[[1000,10000,100000].index(int(method.split('_')[1]))]['eta_upper']
            bound=r['shoulder_bounds'][str(cap)];cut=threshold if method=='new_uncalibrated' else cal_threshold
        score=outward_float(arb(num['lower'])-arb(bound['upper']),False)
        assert a['log_e_lower']==score and a['declared']==(score>=cut)
        assert a['log_threshold_upper']==cut and a['eta_cap']==cap
        assert a['numerator_lower']==num['lower'] and a['null_upper']==bound['upper']
        assert a['primary']==proposal['primary'] and a['secondary']==num['location'] and a['head']==num['head']
    refits=0
    # Two fixed examples per family, selected by ID before viewing outcomes.
    if r['id'].endswith(('_00','_23')):
        g=continuous_null_bound(query,fwhm_range=(4.5,5.5),max_boxes=256,initial_cells=32,normalized_envelope=False)
        assert g==r['gaussian_bound'];refits+=1
        level=outward_float(arb(r['numerators']['new']['lower'])-arb(cal_threshold),False)
        for cap,b in r['shoulder_bounds'].items():
            assert shoulder_null_bound(query,cap=float(cap),decision_level=level)==b;refits+=1
    return checked_null_witnesses,refits


def main():
    start=time.perf_counter();p=check_protocol()
    db=sqlite3.connect(OUT/'records.sqlite3')
    rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert {r['id'] for r in rows}=={c['id'] for c in p['cases']}
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    frozen_snapshot={name:sha(OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    witnesses=refits=done=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify_record,r) for r in rows]):
            w,b=f.result();witnesses+=w;refits+=b;done+=1
            if done%24==0:print('Audited',done,'of',len(rows),flush=True)
    analysis=json.loads((OUT/'analysis.json').read_text());endpoints={}
    for method in METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            s=dict(correct=0,wrong_positive_location=0,false_second_object=0,declared=0,n=0)
            for r in rows:
                if r['case']['group']!=group:continue
                report=r['reports'][method];declared=bool(report['declared']);positive=r['case']['secondary']>0
                correct=positive and declared and abs(report['primary']-r['truth']['primary'])<=2 and abs(report['secondary']-r['truth']['secondary'])<=2
                s['n']+=1;s['declared']+=declared;s['correct']+=correct
                s['wrong_positive_location']+=positive and declared and not correct
                s['false_second_object']+=(not positive) and declared
            assert s==analysis['methods'][method][group];groups[group]=s
        endpoints[method]=groups
    correct=lambda m:sum(endpoints[m][g]['correct'] for g in ['far_two','near_two'])
    false=lambda m:sum(endpoints[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])
    criteria=dict(more_correct=correct('cal_10000')>correct('old_gaussian'),
        fewer_false_second_objects=false('cal_10000')<false('old_gaussian'),
        calibration_benefit=endpoints['cal_10000']['near_two']['correct']>endpoints['new_uncalibrated']['near_two']['correct'])
    assert criteria==analysis['criteria'] and all(criteria.values())==analysis['continue_to_fresh_confirmation']
    for j,n in enumerate([1000,10000,100000]):
        assert analysis['calibration'][str(n)]==dict(
            eta_upper_median=float(np.median([r['confidence'][j]['eta_upper'] for r in rows])),
            informative=sum(r['confidence'][j]['informative'] for r in rows),
            realized_undercoverage=sum(r['confidence'][j]['eta_upper']<r['case']['eta'] for r in rows),
            detected_photons_median=float(np.median([r['confidence'][j]['detected'] for r in rows])))
    for base,saved in analysis['paired'].items():
        gains=losses=0
        for r in rows:
            def good(method):
                s=r['reports'][method]
                return bool(r['case']['secondary']>0 and s['declared'] and abs(s['primary']-r['truth']['primary'])<=2 and abs(s['secondary']-r['truth']['secondary'])<=2)
            a,b=good('cal_10000'),good(base);gains+=a and not b;losses+=b and not a
        assert saved==dict(correct_gains=gains,correct_losses=losses)
    assert all(sha(OUT/name)==digest for name,digest in frozen_snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(P),record_regenerations=done,
        applicable_exact_null_witness_checks=witnesses,complete_bound_refits=refits,
        group_endpoint_recounts=24,output_snapshot_sha256=frozen_snapshot,
        seconds=time.perf_counter()-start,
        scope='All raw-data regeneration hashes,exact calibration endpoints,donor fits,proper mixtures,sparse numerators,decisions,locations,group endpoints,calibration summaries; eight fixed full-cover examples. Not validation of an actual sensor or independent efficacy.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
