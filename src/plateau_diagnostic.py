"""Same eight closed numerical misses; no new scientific efficacy cohort."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
from flint import arb
import conditional_joint_confirmation as s
import joint_bound_diagnostic as w
from joint_plateau_envelope import plateau_shared_cover
from certified_ratio_screen import average_log
from envelope_evidence import outward_float

P=s.ROOT/'JOINT_PLATEAU_DIAGNOSTIC_PROTOCOL.json';OUT=s.ROOT/'results/joint_plateau_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    p=s.check_protocol();sources=dict(p['sources'])
    for name in ['plateau_diagnostic.py','joint_plateau_envelope.py','joint_cone_envelope.py']:
        sources['src/'+name]=s.bgdev.original.sha(s.ROOT/'src'/name)
    ids=[r['id'] for r in json.loads((w.OUT/'analysis.json').read_text())['records']]
    doc=dict(status='FROZEN_BEFORE_PLATEAU_SCORES',sources=sources,selected_ids=ids,
        parent_sqlite_sha256=s.bgdev.original.sha(s.OUT/'records.sqlite3'),max_boxes=1024,
        purpose='Same8closedCI-correct/joint-missedcases; neweta-endpointcone with certifiedcalplateauaware splitpriority. SameQ,CF,threshold,budget.',
        scope='Outcome-selectednumericaldiagnosisonly; combiningnullUPPERboundsforsamefamilyneedsnoadditionalalpha;noindependentaccuracyclaim.')
    P.write_text(json.dumps(doc,indent=2));print('Frozen',s.bgdev.original.sha(P),flush=True)


def worker(row):
    query=np.array(row['query']);folds=[]
    for f in row['methods']['joint']['folds']:
        level=outward_float(arb(f['joint_numerator_lower'])-(arb(2000)/9).log(),False)
        b=plateau_shared_cover(query,f['test_lobe_count'],f['test_lobe_risk'],f['nuisance_confidence'],max_boxes=1024,decision_level=level)
        combo=min(b['upper'],f['bound']['upper'])
        folds.append(dict(bound=b,bare_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(b['upper']),False),
            combined_log_ratio=outward_float(arb(f['joint_numerator_lower'])-arb(combo),False)))
    reports={};cut=outward_float((arb(1000)/9).log())
    for label,key in [('bare','bare_log_ratio'),('combined','combined_log_ratio')]:
        vals=[f[key] for f in folds];lower=average_log(vals);winner=int(np.argmax(vals));q=row['methods']['joint']['folds'][winner]
        declared=lower>=cut;correct=bool(declared and abs(q['proposal']['primary']-row['truth']['primary'])<=2 and abs(q['query_numerator']['location']-row['truth']['secondary'])<=2)
        reports[label]=dict(declared=declared,correct=correct,wrong=bool(declared and not correct),log_ratio_lower=lower)
    return dict(id=row['id'],folds=folds,reports=reports)


def run():
    p=json.loads(P.read_text());s.check_protocol();sha=s.bgdev.original.sha
    for path,digest in p['sources'].items():assert sha(s.ROOT/path)==digest,path
    assert sha(s.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    rows={r['id']:r for r in s.read_records()};proof={r['id']:r for r in json.loads((w.OUT/'analysis.json').read_text())['records']}
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,rows[id]) for id in p['selected_ids']]):
            r=f.result()
            if proof[r['id']]['further_null_refinement_ruled_out']:assert not any(x['declared'] for x in r['reports'].values())
            for new,old in zip(r['folds'],proof[r['id']]['folds']):assert new['bound']['upper']>=old['feasible_joint_loglik_lower']
            records.append(r);print('Plateau diagnostic',len(records),'of',len(p['selected_ids']),flush=True)
    records.sort(key=lambda r:r['id']);OUT.mkdir(parents=True,exist_ok=True)
    doc=dict(status='SELECTED_PLATEAU_DIAGNOSTIC_COMPLETE',records=records,
        totals={m:dict(correct=sum(r['reports'][m]['correct'] for r in records),wrong=sum(r['reports'][m]['wrong'] for r in records)) for m in ['bare','combined']},
        seconds=time.perf_counter()-start,protocol_sha256=sha(P),scope=p['scope'])
    (OUT/'analysis.json').write_text(json.dumps(doc,indent=2));print(json.dumps({k:v for k,v in doc.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);a=parser.parse_args()
    if a.action=='freeze':freeze()
    else:run()
