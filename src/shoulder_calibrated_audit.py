"""Full V2 refit plus separate endpoint/coverage recount and source audit."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,sqlite3,time
import numpy as np
from flint import arb
import shoulder_calibrated_development as study
from envelope_evidence import exact_gaussian,certified_log_likelihood


def verify(parent,saved):
    actual=study.worker(parent)
    actual.pop('elapsed_seconds');expected=dict(saved);expected.pop('elapsed_seconds')
    assert actual==expected,saved['id']
    witnesses=0;c=saved['case'];t=saved['truth']
    if c['secondary']==0:
        g=exact_gaussian(256,t['primary'],t['width'],False)
        side=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
        a=arb(c['primary'])/sum(g,arb(0))
        rates=[(a*(x+arb(c['eta'])*y)+arb(c['background'])/256)/1000 for x,y in zip(g,side)]
        ll=certified_log_likelihood(saved['query'],rates,1000,'first',upper=True)
        for key,bound in saved['shoulder_bounds'].items():
            if saved['confidence'][key]['eta_upper']>=c['eta']:
                assert ll<=bound['upper'];witnesses+=1
    return witnesses


def main():
    start=time.perf_counter();p=study.check_protocol();OUT=study.OUT;sha=study.base.sha
    db=sqlite3.connect(OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    parents={r['id']:r for r in study.parent_records()}
    assert set(parents)=={r['id'] for r in rows}
    snapshot={name:sha(OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    witnesses=done=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        jobs=[ex.submit(verify,parents[r['id']],r) for r in rows]
        for f in as_completed(jobs):
            witnesses+=f.result();done+=1
            if done%24==0:print('Audited calibrated V2',done,'of',len(rows),flush=True)
    a=json.loads((OUT/'analysis.json').read_text());groups={}
    def correct(r,method):
        v=r['reports'][method]
        return bool(r['case']['secondary']>0 and v['declared'] and abs(v['primary']-r['truth']['primary'])<=2 and abs(v['secondary']-r['truth']['secondary'])<=2)
    for method in study.METHODS:
        groups[method]={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==group]
            count=dict(correct=sum(correct(r,method) for r in rs),
                wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][method]['declared'] and not correct(r,method) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][method]['declared'] for r in rs),
                declared=sum(r['reports'][method]['declared'] for r in rs),n=len(rs))
            assert count==a['methods'][method][group];groups[method][group]=count
    numcorrect=lambda m:sum(groups[m][g]['correct'] for g in ['near_two','far_two'])
    numfalse=lambda m:sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])
    criteria=dict(more_correct=numcorrect('cal_10000')>numcorrect('old_gaussian'),
        fewer_false_second_objects=numfalse('cal_10000')<numfalse('old_gaussian'),
        calibration_benefit=groups['cal_10000']['near_two']['correct']>groups['new_uncalibrated']['near_two']['correct'])
    assert criteria==a['criteria'] and all(criteria.values())==a['continue_to_fresh_confirmation']
    for control,pair in a['paired'].items():
        assert pair==dict(correct_gains=sum(correct(r,'cal_10000') and not correct(r,control) for r in rows),
                          correct_losses=sum(correct(r,control) and not correct(r,'cal_10000') for r in rows))
    for n in [1000,10000,100000]:
        key=f'cal_{n}';cf=[r['confidence'][key] for r in rows]
        summary=dict(informative=sum(v['informative'] for v in cf),eta_upper_median=float(np.median([v['eta_upper'] for v in cf])),
            realized_undercoverage=sum(r['confidence'][key]['eta_upper']<r['case']['eta'] for r in rows),
            detected_photons_median=float(np.median([v['detected'] for v in cf])))
        assert summary==a['calibration'][str(n)]
    assert all(sha(OUT/name)==value for name,value in snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(study.P),records_fully_refit=done,
        applicable_exact_null_witness_checks=witnesses,group_endpoint_recounts=24,
        output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All regenerated data/donor fits/independent calibration intervals/proper sparse numerators/full continuous null covers/decisions plus separate endpoints,paired changes,calibration summaries. Closed development, not actual sensor validation or independent efficacy.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
