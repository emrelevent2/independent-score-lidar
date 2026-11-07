"""Full selected-diagnostic refit and independent cohort endpoint recount."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
from flint import arb
import conditional_joint_diagnostic as s


def verify(item,saved):
    actual=s.worker(item);actual.pop('elapsed_seconds');row=dict(saved);row.pop('elapsed_seconds');assert actual==row
    # Convert the confidence-restricted ratio to its confidence-padded p-value.
    raw=arb(1)/1000+(-arb(saved['report']['log_e_lower'])).exp()
    p_upper=1. if not raw<1 else min(1.,s.outward_float(raw))
    if saved['report']['declared']:assert arb(p_upper)<=arb(1)/100
    return p_upper


def main():
    start=time.perf_counter();p=s.check_protocol();sha=s.parent.base.sha;rows=s.read(s.OUT)
    assert {r['id'] for r in rows}=={r['id'] for r in p['selection']}
    items={r['id']:r for r in p['selection']};snapshot={name:sha(s.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    done=0;pvalues={}
    with ProcessPoolExecutor(max_workers=4) as ex:
        jobs={ex.submit(verify,items[r['id']],r):r['id'] for r in rows}
        for f in as_completed(jobs):
            pvalues[jobs[f]]=f.result();done+=1
            if done%24==0:print('Audited selected diagnostic',done,'of',len(rows),flush=True)
    a=json.loads((s.OUT/'analysis.json').read_text());parents={r['id']:r for r in s.read(s.parent.OUT)};points={r['id']:r for r in s.read(s.point.OUT)}
    for cohort in ['repair','regression','null_sanity']:
        subset=[r for r in rows if r['cohort']==cohort]
        for method in ['old_ci','point','joint']:
            correct=wrong=false=declared=0
            for r in subset:
                rep=r['report'] if method=='joint' else parents[r['id']]['reports']['cal_100000'] if method=='old_ci' else points[r['id']]['reports']['point_irf']
                ok=r['case']['secondary']>0 and rep['declared'] and abs(rep['primary']-r['truth']['primary'])<=2 and abs(rep['secondary']-r['truth']['secondary'])<=2
                correct+=ok;wrong+=r['case']['secondary']>0 and rep['declared'] and not ok;false+=r['case']['secondary']==0 and rep['declared'];declared+=rep['declared']
            assert a['cohorts'][cohort][method]==dict(correct=correct,wrong=wrong,false=false,declared=declared,n=len(subset))
    assert all(sha(s.OUT/name)==digest for name,digest in snapshot.items())
    (s.OUT/'confidence_padded_pvalues.json').write_text(json.dumps(dict(delta_union='1/1000',ordinary_unconditional_e_value_claim=False,upper_pvalues=pvalues),indent=2))
    result=dict(status='PASS',protocol_sha256=sha(s.P),records_fully_refit=done,cohort_endpoint_recounts=9,
        confidence_padded_pvalues_checked=True,output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='Selected diagnostic reproduction only; no unbiased net recall/FPR or independent confirmation.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
