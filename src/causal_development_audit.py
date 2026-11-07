"""Closed development verification without claiming all cover computations refit."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,importlib,json,time,hashlib
import numpy as np
from flint import arb
from ordered_audit import correct
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float

MODULES=dict(causal='causal_calibration_development',two='causal_two_window_development',ambiguity='ambiguity_prior_development')


def replay(args):
    which,row,old=args;s=importlib.import_module(MODULES[which]);actual=s.worker(row['case'] if which=='causal' else old);expected=dict(row)
    actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,row['id'];return row['id']


def main(which):
    s=importlib.import_module(MODULES[which]);start=time.perf_counter();p=s.check();rows=s.read();original=s.base.original if which=='causal' else s.parent.base.original;sha=original.sha
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    if which=='two':
        add=json.loads((s.ROOT/'CAUSAL_TWO_WINDOW_DEPENDENCY_ADDENDUM.json').read_text());assert all(sha(s.ROOT/n)==d for n,d in add['full_source_graph'].items())
    parents={} if which=='causal' else {r['id']:r for r in s.parent.read()}
    hashes={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};a=json.loads((s.OUT/'analysis.json').read_text());good=eta_good=cal_witnesses=null_checks=0
    for row in rows:
        h,cal,t=original.simulate_case(row['case']);cal=cal[1];assert t==row['truth'] and hashlib.sha256(h.tobytes()).hexdigest()==row['raw_histogram_sha256']
        if which!='causal':assert cal.tolist()==parents[row['id']]['calibration_histogram']
        if which=='causal':
            from causal_calibration import score_lobe
            method=row['methods']['causal'];pred=method['calibration_predictor'];donor=method['calibration_prior_donor'];assert score_lobe(cal[:43],10000,donor['eta_donor'],int(cal[43:54].sum()))==pred
            cf=method['nuisance_confidence'];interval=row['interval_confidence'];checked=['causal'];joint_methods=['causal'];assert cal.tolist()==row['calibration_histogram']
        elif which=='two':
            from causal_calibration_v2 import score_lobe,score_dark,joint_numerator,two_window_upper
            old=parents[row['id']];donor=old['methods']['causal']['calibration_prior_donor'];pred=row['calibration_predictors'];qk=score_lobe(cal[:43],10000,donor['eta_donor'],int(cal[43:54].sum()));qd=score_dark(cal[:128],10000,int(cal[128:].sum()))
            assert pred==dict(lobe=qk,dark=qd,joint=joint_numerator(qk,qd));cf=row['nuisance_confidence'];interval=old['interval_confidence'];checked=['one_window','two_windows'];joint_methods=checked
        else:
            old=parents[row['id']];cf=old['methods']['causal']['nuisance_confidence'];interval=old['interval_confidence'];checked=['ambiguity_ci'];joint_methods=[]
        c=row['case'];eta=c['eta'];eta_good+=interval['eta_upper']>=eta
        cg=exact_gaussian(256,t['calibration_delay'],t['calibration_width'],False);cs=exact_gaussian(272,arb(t['calibration_delay'])+16,t['calibration_width'],False)[:256]
        A=sum(cg[27:38],arb(0));sigma=arb(.01)*A;beta=arb(.005)/256;F=sum(cg[43:54],arb(0))/A
        compatible=not cf['informative'] or (arb(cf['signal_main_lower'])<=sigma<=arb(cf['signal_main_upper']) and arb(cf['background_lower'])<=beta<=arb(cf['background_upper']) and arb(cf['forward_leakage_lower'])<=F<=arb(cf['forward_leakage_upper']));good+=compatible
        k=int(cal[43:54].sum());R=int(10000-cal[:43].sum());d=int(cal[128:].sum());RD=int(10000-cal[:128].sum())
        H=arb(.01)*sum((x+arb(eta)*y for x,y in zip(cg[43:54],cs[43:54])),arb(0))+11*beta;HD=arb(.01)*sum((x+arb(eta)*y for x,y in zip(cg[128:],cs[128:])),arb(0))+128*beta
        llk=-(R-k)*H;lld=-(RD-d)*HD
        if k:llk+=k*(-(-H).expm1()).log()
        if d:lld+=d*(-(-HD).expm1()).log()
        if which=='two' and compatible:assert arb(two_window_upper(k,R,d,RD,eta,eta,cf)['upper'])>=llk+lld;cal_witnesses+=1
        if not c['secondary']:
            qg=exact_gaussian(256,t['primary'],t['width'],False);qs=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256];amp=arb(c['primary'])/sum(qg,arb(0));rate=[(amp*(x+arb(eta)*y)+arb(c['background'])/256)/1000 for x,y in zip(qg,qs)];ql=arb(certified_log_likelihood(h[4,4],rate,1000,'first',upper=True))
            if compatible:
                for m in joint_methods:
                    bound=row['methods'][m]['bound']
                    if bound is not None:assert arb(bound['upper'])>=ql+llk+(lld if m=='two_windows' else arb(0));null_checks+=1
            if which=='ambiguity' and interval['eta_upper']>=eta and row['methods']['ambiguity_ci']['bound'] is not None:assert arb(row['methods']['ambiguity_ci']['bound']['upper'])>=ql;null_checks+=1
        for m in checked:
            rep=row['reports'][m]
            if rep['declared']:assert outward_float(arb(1)/1000+(-arb(rep['log_ratio_lower'])).exp())<=.01
    totals={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==g];v=dict(correct=sum(correct(r,m) for r in rs),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rs),false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs));assert v==a['groups'][m][g];groups[g]=v
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals']
    primary={'causal':'causal','two':'two_windows','ambiguity':'ambiguity_ci'}[which];x=totals[primary]
    def rule(b):return dict(more_correct=x['correct']>totals[b]['correct'],no_more_wrong=x['wrong']<=totals[b]['wrong'],no_more_false=x['false']<=totals[b]['false'])
    expected=rule('point_ci') if which=='ambiguity' else {b:rule(b) for b in ['grid_ci','point_ci']};assert expected==a['rule']
    chosen=[r for r in rows if int(r['id'].rsplit('_',1)[1]) in [0,7,23]];ids=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,(which,r,parents.get(r['id']))) for r in chosen]):ids.append(f.result())
    assert all(sha(s.OUT/n)==d for n,d in hashes.items())
    v=dict(status='PASS',records=len(rows),predictors_and_raw_data_regenerated=len(rows),confidence_boxes_containing_truth=good,eta_intervals_containing_truth=eta_good,calibration_point_witness_checks=cal_witnesses,applicable_null_cover_checks=null_checks,independent_group_recounts=4*len(s.METHODS),full_worker_replays=len(ids),replay_ids=sorted(ids),output_sha256=hashes,seconds=time.perf_counter()-start,scope='All data/predictors/physicalconfidence truths/groupcounts/criteria checked,12fixedfullworker replays;NOTall96cover refits. Closedmodeldevelopment only.')
    (s.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('which',choices=list(MODULES));main(p.parse_args().which)
