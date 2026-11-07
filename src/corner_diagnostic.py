"""Selected eight-case bound diagnosis,not an efficacy experiment."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
from flint import arb
import conditional_joint_confirmation as s
import joint_bound_diagnostic as witnesses
from coupled_corner_envelope import coupled_cover
from certified_ratio_screen import average_log
from envelope_evidence import outward_float

P=s.ROOT/'COUPLED_CORNER_DIAGNOSTIC_PROTOCOL.json';OUT=s.ROOT/'results/coupled_corner_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    old=s.check_protocol();sources=dict(old['sources'])
    for name in ['corner_diagnostic.py','coupled_corner_profile.py','coupled_corner_envelope.py','joint_plateau_envelope.py','joint_cone_envelope.py']:
        sources['src/'+name]=s.bgdev.original.sha(s.ROOT/'src'/name)
    ids=[r['id'] for r in json.loads((witnesses.OUT/'analysis.json').read_text())['records']]
    p=dict(status='FROZEN_BEFORE_CORNER_DIAGNOSTIC',sources=sources,selected_ids=ids,parent_sqlite_sha256=s.bgdev.original.sha(s.OUT/'records.sqlite3'),
        settings=dict(max_boxes=1024,initial_amplitude=2,initial_eta=4,initial_delay=16),
        question='Same8CI-correct/joint-missed closedcases: cana-eta coupling in checkedconcave tangents tighten the old1024-box bound? SameQ,CF,threshold;four-coordinate cover,amplitude tail retained to infinity.',
        inference='Outcome-selected numerical diagnostic only;not new accuracy/FPR evidence. Compare standalone1024 and mincombined old/newVALIDnullbounds;combined uses extra work,not same-budget speed claim.',
        attribution='Concave tangent majorization,bilinear rectangle corner maximization and branch-and-bound are known tools. No claim of new general theorem. Application-specific computational benefit unproved.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',s.bgdev.original.sha(P),flush=True)


def check():
    p=json.loads(P.read_text());s.check_protocol();sha=s.bgdev.original.sha
    for n,d in p['sources'].items():assert sha(s.ROOT/n)==d,n
    assert sha(s.OUT/'records.sqlite3')==p['parent_sqlite_sha256'];return p


def worker(row):
    query=np.array(row['query']);folds=[]
    for f in row['methods']['joint']['folds']:
        level=outward_float(arb(f['joint_numerator_lower'])-(arb(2000)/9).log(),False)
        bound=coupled_cover(query,f['test_lobe_count'],f['test_lobe_risk'],f['nuisance_confidence'],max_boxes=1024,decision_level=level)
        combined=min(bound['upper'],f['bound']['upper'])
        folds.append(dict(bound=bound,bare_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(bound['upper']),False),combined_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(combined),False)))
    reports={};cut=outward_float((arb(1000)/9).log())
    for label,key in [('bare','bare_log_ratio'),('combined','combined_log_ratio')]:
        values=[f[key] for f in folds];lower=average_log(values);winner=int(np.argmax(values));q=row['methods']['joint']['folds'][winner]
        declared=lower>=cut;correct=bool(declared and abs(q['proposal']['primary']-row['truth']['primary'])<=2 and abs(q['query_numerator']['location']-row['truth']['secondary'])<=2)
        reports[label]=dict(declared=declared,correct=correct,wrong=bool(declared and not correct),log_ratio_lower=lower)
    return dict(id=row['id'],folds=folds,reports=reports)


def run():
    p=check();rows={r['id']:r for r in s.read_records()};proof={r['id']:r for r in json.loads((witnesses.OUT/'analysis.json').read_text())['records']};records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,rows[id]) for id in p['selected_ids']]):
            r=f.result();w=proof[r['id']]
            if w['further_null_refinement_ruled_out']:assert not any(v['declared'] for v in r['reports'].values())
            for a,b in zip(r['folds'],w['folds']):assert a['bound']['upper']>=b['feasible_joint_loglik_lower']
            records.append(r);print('Corner bound diagnostic',len(records),'/',len(p['selected_ids']),flush=True)
    records.sort(key=lambda r:r['id']);OUT.mkdir(parents=True,exist_ok=True)
    v=dict(status='SELECTED_CORNER_DIAGNOSTIC_COMPLETE',records=records,totals={m:dict(correct=sum(r['reports'][m]['correct'] for r in records),wrong=sum(r['reports'][m]['wrong'] for r in records)) for m in ['bare','combined']},seconds=time.perf_counter()-start,protocol_sha256=s.bgdev.original.sha(P),scope=p['inference']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args()
    if args.action=='freeze':freeze()
    else:run()
