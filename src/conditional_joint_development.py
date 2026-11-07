"""Closed 24-scene feasibility pilot: coupled calibration/query evidence."""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time
import numpy as np
from flint import arb
import background_lobe_development as base
from joint_calibration_model import split_cycles,fit_training
from conditional_calibration import nuisance_confidence,predictive_log_interval,calibration_upper,saturated_binomial
from calibration_confidence import window_counts
from fixed_irf_control import fixed_proposal,fixed_components
from shoulder_alternative import numeric_and_sparse
from joint_calibration_envelope import shared_response_cover
from envelope_evidence import outward_float

ROOT=base.ROOT;P=ROOT/'CONDITIONAL_JOINT_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/conditional_joint_development_v1'
METHODS=['old_gaussian','old_ci_100000','background_ci_100000','point_100000','joint']
INDICES=(0,3,7,12,17,23)


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in names:
                path=module.split('.')[0]+'.py'
                if (ROOT/'src'/path).exists() and path not in seen:pending.append(path)
    return {f'src/{name}':base.original.sha(ROOT/'src'/name) for name in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    parent=base.check_protocol();assert json.loads((base.OUT/'integrity.json').read_text())['status']=='PASS'
    cases=[c for c in parent['cases'] if int(c['id'].rsplit('_',1)[1]) in INDICES]
    p=dict(status='FROZEN_BEFORE_ANY_COUPLED_PILOT_OUTCOMES',sources=source_graph(),cases=cases,
        parent_protocol_sha256=base.original.sha(base.P),parent_sqlite_sha256=base.original.sha(base.OUT/'records.sqlite3'),
        settings=dict(calibration_cycles=100000,query_cycles=1000,confidence_failure_per_training_fold='1/2000',
            query_evidence_budget='9/1000',total_bound='1/100',folds=2,average_weights=['1/2','1/2'],
            split_seed_base=2026113010,initial_eta_cells=8,initial_delay_cells=16,max_joint_boxes=1024),
        primary='Twofold mean coupled evidence versus background_ci_100000 on24closed development scenes; same original100k calibration counts randomly split at fixed cycle count INCLUDING no-event category.',
        rule=dict(all_required=True,more_correct='strictly more correct pairs than background_ci_100000',
            no_more_wrong='wrong positive locations no greater',no_more_false='physical-null declarations no greater'),
        construction='Each fold: training main/dark windows bound signal/background (four endpoints,deltafold/4); other-half lobe count scored conditionally on risk R. Proper Jeffreys beta-binomial predictor for K|R, multiplied by proper donor/calibration-trained query mixture. Null shares eta across heldoutcalibration and query; complete eta/delay/width cover. Efold values averaged,not maximized. Union of both confidence failures plus Markov budget gives.01 under stated model.',
        stopping='Each fold may stop once its certified evidence exceeds2/(9/1000); final average still checked at1/(9/1000). Location taken from fold with larger evidence lower,with no localization theorem.',
        attribution='Known confidence-set,conditional-likelihood,proper-predictor and e-value averaging arguments; no new general statistical principle claimed.',
        exclusions='Pilotonly; no independent efficacy,matched runtime,correlated detector noise,calibration drift,unknown shift,real-sensor validation,SOTAorQ1claim. Six-bin full-joint prototype source is unrun; this pilot scores only the heldout lobe count conditionally.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.original.sha(P),len(cases),'cases',len(p['sources']),'sources',flush=True)


def check_protocol():
    p=json.loads(P.read_text());base.check_protocol()
    for path,digest in p['sources'].items():assert base.original.sha(ROOT/path)==digest,path
    assert base.original.sha(base.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    return p


def score_fold(h,train,test,ntrain,ntest):
    fitted=fit_training(train,ntrain);confidence=nuisance_confidence(train,ntrain)
    kt,rt=window_counts(train,ntrain,43,53);k,r=window_counts(test,ntest,43,53)
    calnum=predictive_log_interval(kt,rt,k,r)
    proposal=fixed_proposal(h,fitted['eta']);qnum=numeric_and_sparse(h[4,4],fixed_components(proposal,fitted['eta']))
    jointnum=outward_float(arb(calnum['lower'])+arb(qnum['lower']),False)
    threshold=outward_float((arb(2000)/9).log())
    level=outward_float(arb(jointnum)-arb(threshold),False)
    provider=lambda e0,e1:calibration_upper(k,r,e0,e1,confidence)
    bound=shared_response_cover(h[4,4],provider,decision_level=level,max_boxes=1024)
    log_e=outward_float(arb(jointnum)-arb(bound['upper']),False)
    return dict(training_fit=fitted,nuisance_confidence=confidence,
        training_lobe_count=kt,training_lobe_risk=rt,test_lobe_count=k,test_lobe_risk=r,
        calibration_numerator=calnum,query_numerator=qnum,proposal=proposal,joint_numerator_lower=jointnum,
        bound=bound,log_e_lower=log_e,calibration_predictive_minus_saturated=outward_float(arb(calnum['lower'])-saturated_binomial(k,r),False))


def worker(pair):
    parent,original=pair;c=parent['case'];start=time.perf_counter()
    h,cal,truth=base.original.simulate_case(c)
    assert hashlib.sha256(h.tobytes()).hexdigest()==parent['raw_histogram_sha256']
    # Each original index gives a fixed independent auxiliary randomization.
    index=['null_plain','null_lobe','far_two','near_two'].index(c['group'])*100+int(c['id'].rsplit('_',1)[1])
    seed=2026113010+index
    a,b,na,nb=split_cycles(cal[2],100000,seed)
    folds=[score_fold(h,a,b,na,nb),score_fold(h,b,a,nb,na)]
    shift=max(f['log_e_lower'] for f in folds)
    avg=arb(shift)+(sum(((arb(f['log_e_lower'])-arb(shift)).exp() for f in folds),arb(0))/2).log()
    lower=outward_float(avg,False);cut=outward_float((arb(1000)/9).log());winner=int(np.argmax([f['log_e_lower'] for f in folds]));win=folds[winner]
    report=dict(declared=lower>=cut,log_e_lower=lower,log_threshold_upper=cut,
        primary=win['proposal']['primary'],secondary=win['query_numerator']['location'],head=win['query_numerator']['head'],reported_fold=winner)
    reports={m:parent['reports'][m] for m in METHODS[:-1]};reports['joint']=report
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=parent['raw_histogram_sha256'],
        calibration_split_seed=seed,first_calibration_half=a.tolist(),second_calibration_half=b.tolist(),folds=folds,reports=reports,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def run():
    p=check_protocol();start=time.perf_counter();db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    parent={r['id']:r for r in base.read_records(base.OUT)};original={r['id']:r for r in base.read_records(base.original.OUT)}
    pending=[(parent[c['id']],original[c['id']]) for c in p['cases'] if c['id'] not in ids]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,pair) for pair in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%4==0:print('Coupled pilot',len(ids),'of',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),invocation_seconds=time.perf_counter()-start,
        protocol_sha256=base.original.sha(P),sqlite_sha256=base.original.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check_protocol();rows=base.read_records(OUT);assert len(rows)==len(p['cases']);totals={};groups={}
    for method in METHODS:
        groups[method]={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            e=[base.original.endpoint(r,method) for r in rows if r['case']['group']==group]
            groups[method][group]={k:sum(v[k] for v in e) for k in e[0]}
        totals[method]=dict(correct=sum(groups[method][g]['correct'] for g in ['near_two','far_two']),
            wrong=sum(groups[method][g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[method][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    j,b=totals['joint'],totals['background_ci_100000']
    rule=dict(more_correct=j['correct']>b['correct'],no_more_wrong=j['wrong']<=b['wrong'],no_more_false=j['false']<=b['false'])
    allfolds=[f for r in rows for f in r['folds']]
    result=dict(status='CLOSED_FEASIBILITY_PILOT_COMPLETE',totals=totals,groups=groups,primary_rule=rule,
        continue_to_expanded_development=all(rule.values()),protocol_sha256=base.original.sha(P),
        computation=dict(total_boxes=sum(f['bound']['boxes_evaluated'] for f in allfolds),
            median_boxes=float(np.median([f['bound']['boxes_evaluated'] for f in allfolds])),
            median_cal_predictive_minus_saturated=float(np.median([f['calibration_predictive_minus_saturated'] for f in allfolds]))),
        limitations=p['exclusions'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
