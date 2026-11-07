"""Same selected bound misses; geometry-contracting tangent prototype."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
from flint import arb
import corner_diagnostic as parent
from contracted_corner_envelope import contracted_cover
from certified_ratio_screen import average_log
from envelope_evidence import outward_float

ROOT=parent.s.ROOT;P=ROOT/'CONTRACTED_CORNER_DIAGNOSTIC_PROTOCOL.json';OUT=ROOT/'results/contracted_corner_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    old=parent.check();sources=dict(old['sources'])
    for n in ['contracted_diagnostic.py','contracted_corner_envelope.py','query_likelihood_contractor.py']:
        sources['src/'+n]=parent.s.bgdev.original.sha(ROOT/'src'/n)
    p=dict(status='FROZEN_BEFORE_CONTRACTED_DIAGNOSTIC',sources=sources,selected_ids=old['selected_ids'],parent_sqlite_sha256=old['parent_sqlite_sha256'],settings=dict(max_cover_boxes=1024,initial_eta=8,initial_delay=16),
        mechanism='Queryzero-bin likelihood bounds background; checked outside-window profile plus aggregated binomial deviance bounds primary amplitude pergeometry; discarded parameters remain covered by decision-level upper floor; all amplitudetailtoinfinity retained. Coupled concave tangent optimized over boundedbackground and amplitude/eta corners.',
        scope='Same8outcome-selected CLOSED numerical misses. Not independent accuracy.1024covercells pluscachedoutside profiles/rootchecks,NOTmatchedprofilecountorCPUbudget toold1024. All numerical bounds must dominate saved certified feasible witnesses.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.s.bgdev.original.sha(P),flush=True)


def check():
    p=json.loads(P.read_text());parent.check();sha=parent.s.bgdev.original.sha
    for n,d in p['sources'].items():assert sha(ROOT/n)==d,n
    assert sha(parent.s.OUT/'records.sqlite3')==p['parent_sqlite_sha256'];return p


def worker(row):
    folds=[];query=np.array(row['query'])
    for f in row['methods']['joint']['folds']:
        level=outward_float(arb(f['joint_numerator_lower'])-(arb(2000)/9).log(),False)
        b=contracted_cover(query,f['test_lobe_count'],f['test_lobe_risk'],f['nuisance_confidence'],max_boxes=1024,decision_level=level)
        combined=min(b['upper'],f['bound']['upper'])
        folds.append(dict(bound=b,bare_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(b['upper']),False),combined_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(combined),False)))
    reports={};cut=outward_float((arb(1000)/9).log())
    for label,key in [('bare','bare_log_ratio'),('combined','combined_log_ratio')]:
        vals=[f[key] for f in folds];lower=average_log(vals);winner=int(np.argmax(vals));q=row['methods']['joint']['folds'][winner]
        declared=lower>=cut;correct=bool(declared and abs(q['proposal']['primary']-row['truth']['primary'])<=2 and abs(q['query_numerator']['location']-row['truth']['secondary'])<=2)
        reports[label]=dict(declared=declared,correct=correct,wrong=bool(declared and not correct),log_ratio_lower=lower)
    return dict(id=row['id'],folds=folds,reports=reports)


def run():
    p=check();rows={r['id']:r for r in parent.s.read_records()};proof={r['id']:r for r in json.loads((parent.witnesses.OUT/'analysis.json').read_text())['records']};records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,rows[id]) for id in p['selected_ids']]):
            r=f.result();w=proof[r['id']]
            if w['further_null_refinement_ruled_out']:assert not any(v['declared'] for v in r['reports'].values())
            for a,b in zip(r['folds'],w['folds']):assert a['bound']['upper']>=b['feasible_joint_loglik_lower']
            records.append(r);print('Contracted bound diagnostic',len(records),'/',len(p['selected_ids']),flush=True)
    records.sort(key=lambda r:r['id']);OUT.mkdir(parents=True,exist_ok=True)
    v=dict(status='SELECTED_CONTRACTED_DIAGNOSTIC_COMPLETE',records=records,totals={m:dict(correct=sum(r['reports'][m]['correct'] for r in records),wrong=sum(r['reports'][m]['wrong'] for r in records)) for m in ['bare','combined']},seconds=time.perf_counter()-start,protocol_sha256=parent.s.bgdev.original.sha(P),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args()
    if args.action=='freeze':freeze()
    else:run()
