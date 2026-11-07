"""Closed development of background-aware calibration; no primary switch."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time
import numpy as np
from flint import arb
import shoulder_development as original
import shoulder_calibrated_development as parent
from background_lobe_confidence import lobe_confidence
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from shoulder_envelope import shoulder_null_bound
from fixed_irf_control import estimated_eta,fixed_proposal,fixed_components,fixed_null_bound
from envelope_evidence import outward_float

ROOT=original.ROOT;P=ROOT/'BACKGROUND_LOBE_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/background_lobe_development_v1'
BUDGETS=[1000,10000,100000]
METHODS=['old_gaussian','new_uncalibrated']+[f'{m}_{n}' for m in ['old_ci','background_ci','point'] for n in BUDGETS]


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                path=module.split('.')[0]+'.py'
                if (ROOT/'src'/path).exists() and path not in seen:pending.append(path)
    return {f'src/{name}':original.sha(ROOT/'src'/name) for name in sorted(seen)}


def read_records(folder):
    db=sqlite3.connect(folder/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rows


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    p=parent.check_protocol()
    assert json.loads((parent.OUT/'integrity.json').read_text())['status']=='PASS'
    protocol=dict(status='FROZEN_BEFORE_ANY_BACKGROUND_AWARE_DEVELOPMENT_OUTCOMES',sources=source_graph(),
        original_sqlite_sha256=original.sha(original.OUT/'records.sqlite3'),parent_sqlite_sha256=original.sha(parent.OUT/'records.sqlite3'),
        parent_protocol_sha256=original.sha(parent.P),cases=p['cases'],
        settings=dict(primary_calibration_cycles=100000,diagnostic_calibration_cycles=[1000,10000],
            total_error_budget='1/100',calibration_failure_budget='1/1000',query_budget='9/1000',
            one_sided_calibration_endpoint_budget='1/5000',endpoints=5,initial_cells=32,max_boxes=256),
        primary='Background-aware100k versus frozen three-endpoint100k on closed96 development scenes; same query,donors,calibration and optical model.',
        continue_rule=dict(all_required=True,more_correct='strictly more correct than old_ci_100000',
            no_more_wrong='wrong positive locations no greater than old_ci_100000',
            no_more_false='physical-null declarations no greater than old_ci_100000'),
        secondary='Point-IRF controls use each identical calibration budget; other budgets are diagnostic and cannot replace the primary. Report lower/upper response-bound changes and all family endpoints.',
        rationale='Bound both sides of uniform background using dark window,accounting for nonzero response leakage by a main-window amplitude upper. Five conditional-binomial endpoints share unchanged totaldelta; do not assume uniform interval tightening.',
        limitations='Closed development,not confirmation; Gaussian control misspecified on lobe scenes; no sensor validity,unknown-shift,correlated-afterpulse,universalSOTAorQ1claim.')
    P.write_text(json.dumps(protocol,indent=2));print('Frozen',original.sha(P),len(protocol['sources']),'sources',flush=True)


def check_protocol():
    p=json.loads(P.read_text());parent.check_protocol()
    for path,digest in p['sources'].items():assert original.sha(ROOT/path)==digest,path
    assert original.sha(original.OUT/'records.sqlite3')==p['original_sqlite_sha256']
    assert original.sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    return p


def worker(pair):
    old,pred=pair;start=time.perf_counter();c=old['case']
    h,cal,truth=original.simulate_case(c);query=h[4,4]
    assert hashlib.sha256(h.tobytes()).hexdigest()==old['raw_histogram_sha256']==pred['raw_histogram_sha256']
    reports={m:pred['reports'][m] for m in ['old_gaussian','new_uncalibrated']}
    reports.update({f'old_ci_{n}':pred['reports'][f'cal_{n}'] for n in BUDGETS})
    confidence={};proposals={};numerators={};bounds={};point_estimates={}
    cut=outward_float((arb(1000)/9).log());pcut=outward_float(arb(100).log())
    for j,n in enumerate(BUDGETS):
        key=f'background_ci_{n}';cf=lobe_confidence(cal[j],n);cap=cf['eta_upper'];confidence[key]=cf
        q=donor_proposal(h,cap=cap);num=numeric_and_sparse(query,components(q,cap=cap))
        level=outward_float(arb(num['lower'])-arb(cut),False)
        b=shoulder_null_bound(query,cap=cap,decision_level=level)
        score=outward_float(arb(num['lower'])-arb(b['upper']),False)
        reports[key]=dict(declared=score>=cut,log_e_lower=score,log_threshold_upper=cut,
            primary=q['primary'],secondary=num['location'],head=num['head'],numerator_lower=num['lower'],null_upper=b['upper'],eta_cap=cap)
        proposals[key]=q;numerators[key]=num;bounds[key]=b
        key=f'point_{n}';eta=estimated_eta(cal[j],n);point_estimates[key]=eta
        q=fixed_proposal(h,eta);num=numeric_and_sparse(query,fixed_components(q,eta))
        level=outward_float(arb(num['lower'])-arb(pcut),False);b=fixed_null_bound(query,eta,decision_level=level)
        score=outward_float(arb(num['lower'])-arb(b['upper']),False)
        reports[key]=dict(declared=score>=pcut,log_e_lower=score,log_threshold_upper=pcut,
            primary=q['primary'],secondary=num['location'],head=num['head'],numerator_lower=num['lower'],null_upper=b['upper'],eta_fixed=eta)
        proposals[key]=q;numerators[key]=num;bounds[key]=b
    return dict(id=c['id'],case=c,truth=truth,query=query.tolist(),raw_histogram_sha256=old['raw_histogram_sha256'],
        confidence=confidence,point_estimates=point_estimates,proposals=proposals,numerators=numerators,bounds=bounds,reports=reports,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def run():
    p=check_protocol();start=time.perf_counter();db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    parents={r['id']:r for r in read_records(parent.OUT)}
    pending=[(r,parents[r['id']]) for r in read_records(original.OUT) if r['id'] not in ids]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,pair) for pair in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Background-aware',len(ids),'of',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),invocation_seconds=time.perf_counter()-start,
        protocol_sha256=original.sha(P),sqlite_sha256=original.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check_protocol();rows=read_records(OUT);assert len(rows)==len(p['cases']);summary={};totals={}
    parents={r['id']:r for r in read_records(parent.OUT)}
    for method in METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            ep=[original.endpoint(r,method) for r in rows if r['case']['group']==group]
            groups[group]={k:sum(e[k] for e in ep) for k in ep[0]}
        summary[method]=groups
        totals[method]=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),
            wrong=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),
            false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    primary='background_ci_100000';control='old_ci_100000'
    rule=dict(more_correct=totals[primary]['correct']>totals[control]['correct'],
        no_more_wrong=totals[primary]['wrong']<=totals[control]['wrong'],no_more_false=totals[primary]['false']<=totals[control]['false'])
    calibration={};paired={}
    for n in BUDGETS:
        key=f'background_ci_{n}';old_key=f'cal_{n}'
        diffs=[r['confidence'][key]['eta_upper']-parents[r['id']]['confidence'][old_key]['eta_upper'] for r in rows]
        calibration[str(n)]=dict(tighter=sum(d<0 for d in diffs),looser=sum(d>0 for d in diffs),same=sum(d==0 for d in diffs),
            median_upper=float(np.median([r['confidence'][key]['eta_upper'] for r in rows])),
            realized_undercoverage=sum(r['confidence'][key]['eta_upper']<r['case']['eta'] for r in rows),
            positive_background_lower=sum(r['confidence'][key].get('background_per_bin_lower',0)>0 for r in rows))
        for control in [f'old_ci_{n}',f'point_{n}']:
            a=[original.endpoint(r,key)['correct'] for r in rows];b=[original.endpoint(r,control)['correct'] for r in rows]
            paired[f'{key}_vs_{control}']=dict(gains=sum(x>y for x,y in zip(a,b)),losses=sum(y>x for x,y in zip(a,b)))
    result=dict(status='CLOSED_DEVELOPMENT_COMPLETE',totals=totals,groups=summary,calibration=calibration,paired=paired,
        primary_rule=rule,continue_to_new_confirmation=all(rule.values()),protocol_sha256=original.sha(P),limitations=p['limitations'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
