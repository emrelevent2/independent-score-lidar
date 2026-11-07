"""All diagnostic endpoints/calibration draws,plus fixed full replays."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from flint import arb
import calibration_information_diagnostic as s
from ordered_audit import correct
from physical_confirmation_audit import check_witness
from pilot import first_detection_probabilities
from envelope_evidence import outward_float


def replay(pair):
    old,r=pair;actual=s.worker(old);expected=dict(r);actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,r['id'];return r['id']


def main():
    start=time.perf_counter();p=s.check();rows=s.read();old={r['id']:r for r in s.parent.read()};sha=s.parent.base.original.sha
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};a=json.loads((s.OUT/'analysis.json').read_text());covered=witnesses=0
    for r in rows:
        o=old[r['id']];assert r['case']==o['case'] and r['truth']==o['truth'] and r['raw_histogram_sha256']==o['raw_histogram_sha256']
        assert r['reports']['point_ci_100k']==o['reports']['point_ci'] and r['reports']['point_100k']==o['reports']['point']
        c=r['case'];t=r['truth'];bins=np.arange(256);d=t['calibration_delay'];w=t['calibration_width'];eta=c['eta']
        rate=.01*(np.exp(-4*np.log(2)*((bins-d)/w)**2)+eta*np.exp(-4*np.log(2)*((bins-(d+16))/w)**2))+.005/256
        extra=np.random.default_rng(r['extra_seed']).multinomial(900000,first_detection_probabilities(rate))[:-1]
        assert extra.tolist()==r['extra_histogram'] and (extra+np.array(o['calibration_histogram'])).tolist()==r['calibration_histogram_1m']
        assert sum(r['calibration_histogram_1m'])<=1000000 and len(extra)==256
        cf=r['confidence_1m'];covered+=cf['eta_upper']>=eta
        assert r['known_eta_predictor']['fit']['eta']==eta
        h,_,_=s.parent.base.original.simulate_case(c)
        for m in ['point_ci_1m','point_1m','known_eta']:
            v=r['methods'][m];witness=v['refined_null_witness'];check_witness(h[4,4],witness);witnesses+=1
            if v['bound'] is not None:assert v['bound']['upper']>=witness['lower']
        rep=r['reports']['point_ci_1m']
        if rep['declared']:assert outward_float(arb(1)/1000+(-arb(rep['log_ratio_lower'])).exp())<=.01
    totals={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==g]
            ep=dict(correct=sum(correct(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rs),false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert ep==a['groups'][m][g];groups[g]=ep
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals'] and a['median_detected_calibration_photons_1m']==float(np.median([sum(r['calibration_histogram_1m']) for r in rows]))
    chosen=[r for r in rows if int(r['id'].rsplit('_',1)[1]) in [0,7,23]];ids=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,(old[r['id']],r)) for r in chosen]):ids.append(f.result())
    assert all(sha(s.OUT/n)==v for n,v in snapshot.items())
    v=dict(status='PASS',records=len(rows),independent_extra_calibration_draws_regenerated=len(rows),eta_intervals_covering_truth=covered,refined_witnesses_recomputed=witnesses,independent_group_recounts=20,full_worker_replays=len(ids),replay_ids=sorted(ids),output_sha256=snapshot,seconds=time.perf_counter()-start,scope='All768extra calibration draws/counts/provenance and endpoints checked;12fixedfullreplays,notall768cover refits. CLOSED10xexposure diagnostic,notnewalgorithm efficacy.')
    (s.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':main()
