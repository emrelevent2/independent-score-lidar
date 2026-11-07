"""V2 development: calibration constrains predictor AND null before scoring."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time
import numpy as np
from flint import arb
import shoulder_development as base
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from calibration_confidence_v2 import lobe_confidence
from shoulder_envelope import shoulder_null_bound
from envelope_evidence import outward_float

ROOT=base.ROOT;P=ROOT/'SHOULDER_CALIBRATED_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/shoulder_calibrated_development_v2';PARENT=base.OUT
METHODS=base.METHODS


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else
                     [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:pending.append(local)
    return {f'src/{n}':base.sha(ROOT/'src'/n) for n in sorted(seen)}


def parent_records():
    db=sqlite3.connect(PARENT/'records.sqlite3')
    rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    return rows


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    parent=base.check_protocol();assert json.loads((PARENT/'integrity.json').read_text())['status']=='PASS'
    p=dict(status='FROZEN_BEFORE_V2_OUTCOMES_ON_CLOSED_V1_DEVELOPMENT',sources=source_graph(),
        parent_protocol_sha256=base.sha(base.P),parent_sqlite_sha256=base.sha(PARENT/'records.sqlite3'),
        parent_analysis_sha256=base.sha(PARENT/'analysis.json'),cases=parent['cases'],settings=parent['settings'],
        primary_comparison=parent['primary_comparison'],continuation_rule=parent['continuation_rule'],
        changes='For each fixed calibration budget, refit donors with eta<=independent calibration U; construct the proper three-eta predictor with the same cap U; then score the query against that capped continuous null. Improved conservative binomial root bracketing. Other controls frozen parent outputs.',
        independence='Q may depend on calibration and donors, never the queried counts. Conditional proper predictive distribution plus null confidence-set union bound; known general principle.',
        exclusions=parent['exclusions'],no_primary_switch=True,
        design='Same closed96 development scenes; this is not new/independent efficacy. Fresh confirmation required if all continuation criteria pass.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.sha(P),len(p['sources']),'sources',flush=True)


def check_protocol():
    p=json.loads(P.read_text());base.check_protocol()
    for path,digest in p['sources'].items():assert base.sha(ROOT/path)==digest,path
    assert base.sha(PARENT/'records.sqlite3')==p['parent_sqlite_sha256']
    assert base.sha(PARENT/'analysis.json')==p['parent_analysis_sha256']
    return p


def worker(parent):
    start=time.perf_counter();c=parent['case'];h,cal,truth=base.simulate_case(c);query=h[4,4]
    assert hashlib.sha256(h.tobytes()).hexdigest()==parent['raw_histogram_sha256']
    assert query.tolist()==parent['query']
    cut=outward_float((arb(1000)/9).log());reports={m:parent['reports'][m] for m in METHODS[:3]}
    confidence={};proposals={};numerators={};bounds={}
    for x,n in zip(cal,[1000,10000,100000]):
        key=f'cal_{n}';cf=lobe_confidence(x,n);cap=cf['eta_upper'];confidence[key]=cf
        q=donor_proposal(h,cap=cap);num=numeric_and_sparse(query,components(q,cap=cap))
        level=outward_float(arb(num['lower'])-arb(cut),False)
        b=shoulder_null_bound(query,cap=cap,decision_level=level)
        score=outward_float(arb(num['lower'])-arb(b['upper']),False)
        reports[key]=dict(declared=score>=cut,log_e_lower=score,log_threshold_upper=cut,
            primary=q['primary'],secondary=num['location'],head=num['head'],
            numerator_lower=num['lower'],null_upper=b['upper'],eta_cap=cap)
        proposals[key]=q;numerators[key]=num;bounds[key]=b
    return dict(id=c['id'],case=c,truth=truth,query=query.tolist(),raw_histogram_sha256=parent['raw_histogram_sha256'],
        confidence=confidence,proposals=proposals,numerators=numerators,shoulder_bounds=bounds,reports=reports,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, data TEXT NOT NULL)');db.commit();return db


def run(workers):
    p=check_protocol();start=time.perf_counter();db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    pending=[r for r in parent_records() if r['id'] not in ids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(worker,r) for r in pending]):
            row=f.result();db.execute('INSERT INTO records VALUES (?,?)',(row['id'],json.dumps(row)));db.commit();ids.add(row['id'])
            if len(ids)%12==0:print('Completed calibrated V2',len(ids),'of96',flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),
        invocation_seconds=time.perf_counter()-start,protocol_sha256=base.sha(P),sqlite_sha256=base.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check_protocol();db=connect();rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert len(rows)==len(p['cases']);summary={}
    for method in METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            e=[base.endpoint(r,method) for r in rows if r['case']['group']==group]
            groups[group]={k:sum(x[k] for x in e) for k in e[0]}
        summary[method]=groups
    correct=lambda m:sum(summary[m][g]['correct'] for g in ['far_two','near_two'])
    false=lambda m:sum(summary[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])
    criteria=dict(more_correct=correct('cal_10000')>correct('old_gaussian'),
        fewer_false_second_objects=false('cal_10000')<false('old_gaussian'),
        calibration_benefit=summary['cal_10000']['near_two']['correct']>summary['new_uncalibrated']['near_two']['correct'])
    paired={}
    for control in ['old_gaussian','new_gaussian','new_uncalibrated']:
        gain=loss=0
        for r in rows:
            a,b=base.endpoint(r,'cal_10000')['correct'],base.endpoint(r,control)['correct'];gain+=a>b;loss+=b>a
        paired[control]=dict(correct_gains=gain,correct_losses=loss)
    calibration={}
    for n in [1000,10000,100000]:
        key=f'cal_{n}';cf=[r['confidence'][key] for r in rows]
        calibration[str(n)]=dict(informative=sum(x['informative'] for x in cf),eta_upper_median=float(np.median([x['eta_upper'] for x in cf])),
            realized_undercoverage=sum(r['confidence'][key]['eta_upper']<r['case']['eta'] for r in rows),
            detected_photons_median=float(np.median([x['detected'] for x in cf])))
    result=dict(status='CLOSED_DEVELOPMENT_V2_COMPLETE',methods=summary,criteria=criteria,
        continue_to_fresh_confirmation=all(criteria.values()),paired=paired,calibration=calibration,
        primary='cal_10000 fixed before V2 outcomes, conditional donor-only predictor',
        limitations=p['exclusions'],protocol_sha256=base.sha(P))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers)
    else:analyze()
