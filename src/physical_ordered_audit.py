"""Independent counts and fixed worker replays for physical calibration Q."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
from flint import arb
import physical_ordered_development as s
from ordered_audit import correct,physical


def replay(pair):
    old,row=pair;actual=s.worker(old);expected=dict(row)
    actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,row['id']
    return row['id']


def main():
    start=time.perf_counter();p=s.check();rows=s.read();old={r['id']:r for r in s.parent.read()};sha=s.parent.base.original.sha
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']}
    a=json.loads((s.OUT/'analysis.json').read_text());good=eta=witness=0
    for r in rows:
        o=old[r['id']];assert r['case']==o['case'] and r['truth']==o['truth'] and r['raw_histogram_sha256']==o['raw_histogram_sha256']
        for m in ['grid_ci','point_ci','point']:assert r['reports'][m]==o['reports'][m]
        assert r['reports']['ordered_beta']==o['reports']['ordered']
        nc=dict(r['new']['nuisance_confidence']);oc=dict(o['methods']['ordered']['nuisance_confidence']);nc.pop('scope');oc.pop('scope');assert nc==oc
        wrapper=dict(o);wrapper['methods']=dict(o['methods']);wrapper['methods']['ordered']=r['new']
        wrapper['reports']=dict(o['reports']);wrapper['reports']['ordered']=r['reports']['physical']
        x,y,z=physical(wrapper);good+=x;eta+=y;witness+=z
        for m in s.NEW:
            rep=r['reports'][m]
            if rep['declared']:assert .001+float((-arb(rep['log_ratio_lower'])).exp())<=.01
    totals={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==g]
            ep=dict(correct=sum(correct(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rs),false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert ep==a['groups'][m][g];groups[g]=ep
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals']
    eligible=[m for m in s.NEW if all(totals[m]['correct']>totals[b]['correct'] and all(totals[m][k]<=totals[b][k] for k in ['wrong','false']) for b in ['grid_ci','point_ci'])]
    selected=max(eligible,key=lambda m:totals[m]['correct']) if eligible else None
    assert eligible==a['eligible'] and selected==a['selected']
    chosen=[r for r in rows if int(r['id'].rsplit('_',1)[1]) in [0,7,23]];ids=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,(old[r['id']],r)) for r in chosen]):ids.append(f.result())
    assert all(sha(s.OUT/n)==v for n,v in snapshot.items())
    result=dict(status='PASS',records=len(rows),confidence_boxes_covering_truth=good,eta_intervals_covering_truth=eta,applicable_null_witnesses=witness,full_worker_replays=len(ids),replay_ids=sorted(ids),independent_group_recounts=24,selection_reproduced=True,output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All saved decisions/groups,source/data provenance,CF equivalence,physical witnesses and fixed12full replays. Not all96worker refits. Legacy nuisance helper scope string says training-half; actual caller passes FULL calibration atdelta=.001,as protocol requires.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
