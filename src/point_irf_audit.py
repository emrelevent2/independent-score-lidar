"""All-record point-control replay plus independent feasible/null endpoints."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,sqlite3,time
from flint import arb
import point_irf_comparison as study
from fixed_irf_control import estimated_eta,fixed_proposal,fixed_components,fixed_null_bound
from shoulder_alternative import numeric_and_sparse
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float


def verify(r,parent):
    h,_,truth=study.parent.base.simulate_case(r['case']);query=h[4,4]
    assert truth==r['truth'] and r['case']==parent['case']
    assert hashlib.sha256(h.tobytes()).hexdigest()==r['raw_histogram_sha256']==parent['raw_histogram_sha256']
    eta=estimated_eta(parent['calibration_histogram']);assert eta==r['eta_hat']
    q=fixed_proposal(h,eta);assert q==r['proposal']
    num=numeric_and_sparse(query,fixed_components(q,eta));assert num==r['numerator']
    cut=outward_float(arb(100).log());score=outward_float(arb(num['lower'])-arb(r['bound']['upper']),False)
    assert r['reports']['point_irf']==dict(declared=score>=cut,primary=q['primary'],secondary=num['location'],head=num['head'],
        log_e_lower=score,log_threshold_upper=cut,numerator_lower=num['lower'],null_upper=r['bound']['upper'])
    # An independent feasible witness in the ASSUMED point-IRF family. This
    # deliberately makes no assertion that eta_hat equals the sensor truth.
    g=exact_gaussian(256,q['primary'],q['fitted_fwhm'],False)
    side=exact_gaussian(272,arb(q['primary'])+16,q['fitted_fwhm'],False)[:256]
    a=arb(q['primary_incident'])/sum(g,arb(0));bg=arb(q['background_incident'])
    rates=[(a*(x+arb(eta)*y)+bg/256)/1000 for x,y in zip(g,side)]
    ll=certified_log_likelihood(query,rates,1000,'first',upper=True)
    assert ll<=r['bound']['upper']
    full=0
    if r['id'].endswith(('_0000','_0191')):
        level=outward_float(arb(num['lower'])-arb(cut),False)
        assert fixed_null_bound(query,eta,decision_level=level)==r['bound'];full=1
    return full


def main():
    start=time.perf_counter();p=study.check_protocol();OUT=study.OUT;sha=study.parent.base.sha
    db=sqlite3.connect(OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    db=sqlite3.connect(study.parent.OUT/'records.sqlite3');parents={r['id']:r for r in (json.loads(x[0]) for x in db.execute('SELECT data FROM records'))};db.close()
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    complete=json.loads((OUT/'completion.json').read_text());assert sha(study.parent.OUT/'records.sqlite3')==complete['parent_sqlite_sha256']
    snapshot={name:sha(OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    done=full=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,r,parents[r['id']]) for r in rows]):
            full+=f.result();done+=1
            if done%192==0:print('Audited point IRF',done,'of',len(rows),flush=True)
    a=json.loads((OUT/'analysis.json').read_text());groups={}
    for group in ['null_plain','null_lobe','far_two','near_two']:
        rs=[r for r in rows if r['case']['group']==group]
        def good(r):
            s=r['reports']['point_irf'];t=r['truth']
            return r['case']['secondary']>0 and s['declared'] and abs(s['primary']-t['primary'])<=2 and abs(s['secondary']-t['secondary'])<=2
        v=dict(correct=sum(good(r) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports']['point_irf']['declared'] and not good(r) for r in rs),
            false_second_object=sum(r['case']['secondary']==0 and r['reports']['point_irf']['declared'] for r in rs),
            declared=sum(r['reports']['point_irf']['declared'] for r in rs),n=len(rs))
        assert v==a['groups'][group];groups[group]=v
    totals=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),
        wrong_location=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),
        false_objects=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals'];assert all(sha(OUT/name)==value for name,value in snapshot.items())
    result=dict(status='PASS',records_replayed=done,assumed_model_feasible_witness_checks=done,
        fixed_cover_refits=full,group_endpoint_recounts=4,protocol_sha256=sha(study.P),output_sha256=snapshot,
        seconds=time.perf_counter()-start,scope='Numerical assumed-point-IRF and reproducibility checks; no physical-null calibration coverage assertion.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
