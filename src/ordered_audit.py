"""Independent recounts, physical witnesses and preselected worker replays."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
from flint import arb
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float


def correct(r,m):
    v=r['reports'][m];t=r['truth']
    return bool(r['case']['secondary']>0 and v['declared'] and abs(v['primary']-t['primary'])<=2 and abs(v['secondary']-t['secondary'])<=2)


def physical(row):
    t=row['truth'];c=row['case'];eta=c['eta']
    g=exact_gaussian(256,t['calibration_delay'],t['calibration_width'],False)
    A=sum(g[27:38],arb(0));sig=arb(.01)*A;beta=arb(.005)/256;F=sum(g[43:54],arb(0))/A
    cf=row['methods']['ordered']['nuisance_confidence']
    good=not cf['informative'] or (arb(cf['signal_main_lower'])<=sig<=arb(cf['signal_main_upper']) and
        arb(cf['background_lower'])<=beta<=arb(cf['background_upper']) and arb(cf['forward_leakage_lower'])<=F<=arb(cf['forward_leakage_upper']))
    eta_good=row['interval_confidence']['eta_upper']>=eta
    witnesses=0
    if c['secondary']==0:
        qg=exact_gaussian(256,t['primary'],t['width'],False);qs=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
        amp=arb(c['primary'])/sum(qg,arb(0));rates=[(amp*(x+arb(eta)*y)+arb(c['background'])/256)/1000 for x,y in zip(qg,qs)]
        from shoulder_development import simulate_case
        h,_,_=simulate_case(c);ll=certified_log_likelihood(h[4,4],rates,1000,'first',upper=True)
        for m in ['grid_ci','point_ci']:
            bound=row['methods'][m]['bound']
            if eta_good and bound is not None:assert bound['upper']>=ll;witnesses+=1
        if good:
            calrate=sig*(arb(eta)+F)+11*beta
            for fold in row['methods']['ordered']['folds']:
                if 'bound' not in fold:continue
                k,r=fold['test_lobe_count'],fold['test_lobe_risk'];value=arb(ll)-(r-k)*calrate
                if k:value+=k*(-(-calrate).expm1()).log()
                assert arb(fold['bound']['upper'])>=value;witnesses+=1
    for m in ['grid_ci','point_ci','ordered']:
        rep=row['reports'][m];l=rep['log_ratio_lower']
        if rep['declared']:
            p=arb(1)/1000+(-arb(l)).exp();assert outward_float(p)<=.01
    return int(good),int(eta_good),witnesses


def replay(pair):
    module,row=pair
    if module=='development':import ordered_development as s
    else:import ordered_confirmation as s
    actual=s.worker(row['case']);expected=dict(row)
    actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,row['id']
    return row['id']


def main(kind):
    if kind=='development':import ordered_development as s
    else:import ordered_confirmation as s
    start=time.perf_counter();p=s.check();rows=s.read();sha=s.base.original.sha
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']}
    a=json.loads((s.OUT/'analysis.json').read_text());good=egood=wit=0
    for row in rows:
        v=physical(row);good+=v[0];egood+=v[1];wit+=v[2]
    totals={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==g]
            ep=dict(correct=sum(correct(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert a['groups'][m][g]==ep;groups[g]=ep
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals']
    # Chosen solely by fixed stratum indices,not outcomes; exact full replays.
    chosen=[r for r in rows if int(r['case']['id'].rsplit('_',1)[1]) in [0,7,23]]
    done=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,(kind,r)) for r in chosen]):done.append(f.result())
    if kind=='development':
        def beats(x,y):return totals[x]['correct']>totals[y]['correct'] and all(totals[x][k]<=totals[y][k] for k in ['wrong','false'])
        selected='point_ci' if beats('point_ci','grid_ci') else None
        if selected and beats('ordered','point_ci'):selected='ordered'
        assert selected==a['selected']
    else:
        selected=p['selected_method'];rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates'];pos=[r for r in rows if r['case']['secondary']>0];boot=np.zeros(rep)
        gains=sum(correct(r,selected) and not correct(r,'grid_ci') for r in pos);losses=sum(correct(r,'grid_ci') and not correct(r,selected) for r in pos)
        for g in ['far_two','near_two']:
            for st in range(24):
                d=np.array([int(correct(r,selected))-int(correct(r,'grid_ci')) for r in pos if r['case']['group']==g and r['case']['stratum']==st]);assert len(d)==8
                boot+=rng.choice(d,size=(rep,8),replace=True).sum(1)/len(pos)
        ci=np.quantile(boot,[.025,.975]).tolist()
        assert a['paired']==dict(gains=gains,losses=losses,rate_difference=(gains-losses)/len(pos),stratified_percentile95_ci=ci)
        x,y=totals[selected],totals['grid_ci'];rule=dict(correct_gain=ci[0]>0,no_more_wrong=x['wrong']<=y['wrong'],no_more_false=x['false']<=y['false'])
        assert rule==a['success_rule'] and all(rule.values())==a['all_prespecified_success_criteria']
    assert all(sha(s.OUT/n)==v for n,v in snapshot.items())
    result=dict(status='PASS',records=len(rows),confidence_boxes_covering_truth=good,eta_intervals_covering_truth=egood,applicable_physical_null_witnesses=wit,
        full_worker_replays=len(done),replay_ids=sorted(done),independent_group_recounts=4*len(s.METHODS),output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='All saved endpoints,confidence physical truths and applicable null witnesses checked; full worker replays on fixed0,7,23indices per group,not every record. All numerical kernels have separate unit tests.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['development','confirmation']);main(ap.parse_args().kind)
