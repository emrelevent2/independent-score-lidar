"""Prospective physical-predictor data, endpoints, witnesses and fixed replays."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,hashlib
import numpy as np
from flint import arb
import physical_ordered_confirmation as s
from ordered_audit import correct,physical
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float


def replay(row):
    actual=s.worker(row['case']);expected=dict(row)
    actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,row['id']
    return row['id']


def check_witness(q,w):
    a,b,d,width,eta=w['parameters']
    assert a>=0 and b>=0 and 0<=d<=255 and 4.5<=width<=5.5 and 0<=eta<=.25
    g=exact_gaussian(256,d,width,False);side=exact_gaussian(272,arb(d)+16,width,False)[:256]
    rates=[(arb(a)*(x+arb(eta)*y)+arb(b)/256)/1000 for x,y in zip(g,side)]
    assert certified_log_likelihood(q,rates,1000,'first')==w['lower']


def main():
    start=time.perf_counter();p=s.check();rows=s.read();sha=s.base.original.sha
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};a=json.loads((s.OUT/'analysis.json').read_text())
    good=eta=witnesses=refined=0
    for r in rows:
        h,cal,truth=s.base.original.simulate_case(r['case']);assert truth==r['truth'] and hashlib.sha256(h.tobytes()).hexdigest()==r['raw_histogram_sha256'] and cal[2].tolist()==r['calibration_histogram']
        wrapper=dict(r);wrapper['methods']=dict(r['methods']);wrapper['methods']['ordered']=r['methods']['physical'];wrapper['reports']=dict(r['reports']);wrapper['reports']['ordered']=r['reports']['physical']
        x,y,z=physical(wrapper);good+=x;eta+=y;witnesses+=z
        for m in ['grid_ci','point_ci','point']:
            w=r['methods'][m]['refined_null_witness'];check_witness(h[4,4],w);refined+=1
            cap=r['interval_confidence']['eta_upper'];pe=r['full_calibration_predictor']['fit']['eta']
            assert w['parameters'][4]==pe if m=='point' else 0<=w['parameters'][4]<=cap
            bound=r['methods'][m]['bound']
            if bound is not None:assert bound['upper']>=w['lower']
        for f in r['methods']['physical']['folds']:
            w=f['query_witness']
            if 'parameters' in w:check_witness(h[4,4],w);refined+=1
        rep=r['reports']['physical']
        if rep['declared']:assert outward_float(arb(1)/1000+(-arb(rep['log_ratio_lower'])).exp())<=.01
    totals={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==g]
            ep=dict(correct=sum(correct(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rs),false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert ep==a['groups'][m][g];groups[g]=ep
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals'];pos=[r for r in rows if r['case']['secondary']>0];rng=np.random.default_rng(p['settings']['bootstrap_seed']);reps=p['settings']['bootstrap_replicates'];boot=np.zeros(reps)
    gains=sum(correct(r,'physical') and not correct(r,'grid_ci') for r in pos);losses=sum(correct(r,'grid_ci') and not correct(r,'physical') for r in pos)
    for group in ['far_two','near_two']:
        for st in range(24):
            d=np.array([int(correct(r,'physical'))-int(correct(r,'grid_ci')) for r in pos if r['case']['group']==group and r['case']['stratum']==st]);assert len(d)==8
            boot+=rng.choice(d,size=(reps,8),replace=True).sum(1)/len(pos)
    ci=np.quantile(boot,[.025,.975]).tolist();assert a['paired']==dict(gains=gains,losses=losses,rate_difference=(gains-losses)/len(pos),stratified_percentile95_ci=ci)
    x,y=totals['physical'],totals['grid_ci'];rule=dict(correct_gain=ci[0]>0,no_more_wrong=x['wrong']<=y['wrong'],no_more_false=x['false']<=y['false']);assert a['success_rule']==rule and a['all_prespecified_success_criteria']==all(rule.values())
    chosen=[r for r in rows if int(r['id'].rsplit('_',1)[1]) in [0,7,23,63,127,191]];ids=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,r) for r in chosen]):ids.append(f.result())
    assert all(sha(s.OUT/n)==v for n,v in snapshot.items())
    result=dict(status='PASS',records_regenerated=len(rows),confidence_boxes_covering_truth=good,eta_intervals_covering_truth=eta,applicable_null_witness_checks=witnesses,refined_feasible_witnesses_recomputed=refined,full_worker_replays=len(ids),replay_ids=sorted(ids),group_recounts=16,paired_bootstrap_reproduced=True,primary_criteria_reproduced=True,output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All raw observations/calibration/truth regenerated,all endpoints/CI/bootstrap recounted,all saved refined physical witnesses recomputed;24fixedfullworker replays,NOTall768fullcover refits. Screened nulls have no evaluated cover.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
