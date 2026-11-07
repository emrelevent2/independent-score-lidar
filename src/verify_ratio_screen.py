"""Check new reject-only certificate against all frozen diagnostic outputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
import numpy as np
from flint import arb
import conditional_joint_diagnostic as d
from certified_ratio_screen import prepare_fold,average_log
from envelope_evidence import outward_float


def verify(row):
    h,_,_=d.parent.base.simulate_case(row['case'])
    assert hashlib.sha256(h.tobytes()).hexdigest()==row['raw_histogram_sha256']
    a=np.array(row['first_calibration_half']);b=np.array(row['second_calibration_half'])
    fs=[prepare_fold(h,a,b,50000,50000),prepare_fold(h,b,a,50000,50000)]
    for new,old in zip(fs,row['folds']):
        for key in ['training_fit','nuisance_confidence','calibration_numerator','query_numerator','proposal','joint_numerator_lower']:
            assert new[key]==old[key]
        assert new['screen']['joint_null_witness_lower']<=old['bound']['upper']
        assert new['screen']['log_ratio_upper']>=old['log_e_lower']
    upper=average_log([f['screen']['log_ratio_upper'] for f in fs],True)
    assert upper>=row['report']['log_e_lower']
    screened=upper<outward_float((arb(1000)/9).log(),False)
    if screened:assert not row['report']['declared']
    return dict(cohort=row['cohort'],screened=screened,
                saved_cover_boxes=sum(f['bound']['boxes_evaluated'] for f in row['folds']) if screened else 0)


def main():
    start=time.perf_counter();d.check_protocol();rows=d.read(d.OUT);assert json.loads((d.OUT/'integrity.json').read_text())['status']=='PASS'
    summary={c:dict(n=0,screened=0,saved_cover_boxes=0) for c in ['repair','regression','null_sanity']}
    done=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,r) for r in rows]):
            r=f.result();s=summary[r['cohort']];s['n']+=1;s['screened']+=r['screened'];s['saved_cover_boxes']+=r['saved_cover_boxes'];done+=1
    result=dict(status='PASS',records_checked=done,folds_checked=2*done,cohorts=summary,
        diagnostic_protocol_sha256=d.parent.base.sha(d.P),screen_source_sha256=d.parent.base.sha(d.ROOT/'src/certified_ratio_screen.py'),
        seconds=time.perf_counter()-start,
        scope='All predictor values unchanged,all null witnesses below saved joint cover uppers,all sparse-ratio uppers above saved lower,zero accepted decisions removed. Selected diagnostic only; box counts are work avoided,not a measured speedup.')
    (d.OUT/'screen_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
