"""Outcome-selected failure analysis on a CLOSED cohort, not new efficacy.

The preceding24-scene pilot failed its strict-more-correct rule. This separate
diagnostic keeps the method frozen and asks whether it can repair the known
point-versus-confidence gap. Regression/null sanity samples are also retained.
Do not combine these selected strata into an accuracy or false-alarm estimate.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
import numpy as np
from flint import arb
import conditional_joint_development as method
import shoulder_confirmation as parent
import point_irf_comparison as point
from joint_calibration_model import split_cycles
from envelope_evidence import outward_float

ROOT=parent.ROOT;P=ROOT/'CONDITIONAL_JOINT_DIAGNOSTIC_PROTOCOL.json'
OUT=ROOT/'results/conditional_joint_diagnostic_v1'


def read(folder):
    db=sqlite3.connect(folder/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rows


def correct(r,report):
    return bool(r['case']['secondary']>0 and report['declared'] and abs(report['primary']-r['truth']['primary'])<=2 and abs(report['secondary']-r['truth']['secondary'])<=2)


def selection():
    rows=read(parent.OUT);points={r['id']:r for r in read(point.OUT)};chosen={}
    for r in rows:
        old=correct(r,r['reports']['cal_100000']);pc=correct(r,points[r['id']]['reports']['point_irf'])
        if pc and not old:chosen[r['id']]='repair'
    for group in ['far_two','near_two']:
        candidates=[r for r in rows if r['case']['group']==group and correct(r,r['reports']['cal_100000'])]
        for r in candidates[:12]:chosen[r['id']]='regression'
    for group in ['null_plain','null_lobe']:
        for r in [r for r in rows if r['case']['group']==group][:12]:chosen[r['id']]='null_sanity'
    return [dict(id=r['id'],cohort=chosen[r['id']],case=r['case']) for r in rows if r['id'] in chosen]


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    m=method.check_protocol();parent.check_protocol();pp=point.check_protocol()
    assert json.loads((method.OUT/'integrity.json').read_text())['status']=='PASS'
    a=json.loads((method.OUT/'analysis.json').read_text());assert not a['continue_to_expanded_development']
    sources=dict(m['sources']);sources.update(pp['sources']);sources['src/conditional_joint_diagnostic.py']=parent.base.sha(Path(__file__))
    cases=selection()
    p=dict(status='FROZEN_BEFORE_ANY_JOINT_SCORES_ON_SELECTED_CLOSED768_COHORT',sources=sources,selection=cases,
        parent_protocol_sha256=parent.base.sha(parent.P),parent_sqlite_sha256=parent.base.sha(parent.OUT/'records.sqlite3'),
        point_sqlite_sha256=parent.base.sha(point.OUT/'records.sqlite3'),pilot_protocol_sha256=parent.base.sha(method.P),
        rationale='Preserve failed24casepilot. Separate diagnostic of all point-correct/confidence-missed cases,24old-confidence-correct regression cases and24physical-null sanity cases; outcome selection explicit.',
        settings=dict(calibration_cycles=100000,split_seed_base=2026120100,folds=2,max_joint_boxes=1024),
        inference='Descriptive repair/regression/sanity counts ONLY; no unbiased net recall,false-alarm rate,independent efficacy or successful primary claim. No method/threshold change. Do not count these samples as new independent scenes.',
        next_step='Only evidence of useful repair with tolerable regression would motivate a new prospective protocol; this diagnostic does not itself establish that result.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.sha(P),len(cases),'selected records',flush=True)


def check_protocol():
    p=json.loads(P.read_text());method.check_protocol();parent.check_protocol();point.check_protocol()
    for path,digest in p['sources'].items():assert parent.base.sha(ROOT/path)==digest,path
    assert parent.base.sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    assert parent.base.sha(point.OUT/'records.sqlite3')==p['point_sqlite_sha256']
    assert selection()==p['selection'];return p


def worker(item):
    c=item['case'];start=time.perf_counter();h,cal,truth=parent.base.simulate_case(c)
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=2026120100+100000*g+i
    a,b,na,nb=split_cycles(cal[2],100000,seed)
    folds=[method.score_fold(h,a,b,na,nb),method.score_fold(h,b,a,nb,na)]
    shift=max(f['log_e_lower'] for f in folds)
    avg=arb(shift)+(sum(((arb(f['log_e_lower'])-arb(shift)).exp() for f in folds),arb(0))/2).log()
    lower=outward_float(avg,False);cut=outward_float((arb(1000)/9).log());winner=int(np.argmax([f['log_e_lower'] for f in folds]));w=folds[winner]
    report=dict(declared=lower>=cut,log_e_lower=lower,log_threshold_upper=cut,primary=w['proposal']['primary'],
        secondary=w['query_numerator']['location'],head=w['query_numerator']['head'],reported_fold=winner)
    return dict(id=item['id'],cohort=item['cohort'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
        calibration_split_seed=seed,first_calibration_half=a.tolist(),second_calibration_half=b.tolist(),folds=folds,report=report,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def run():
    p=check_protocol();start=time.perf_counter();db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    pending=[r for r in p['selection'] if r['id'] not in ids]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,r) for r in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Failure diagnostic',len(ids),'of',len(p['selection']),flush=True)
    assert len(ids)==len(p['selection']);db.close()
    completion=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),invocation_seconds=time.perf_counter()-start,
        protocol_sha256=parent.base.sha(P),sqlite_sha256=parent.base.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(completion,indent=2));print(completion,flush=True)


def analyze():
    p=check_protocol();rows=read(OUT);assert len(rows)==len(p['selection']);parents={r['id']:r for r in read(parent.OUT)}
    points={r['id']:r for r in read(point.OUT)};result={}
    for cohort in ['repair','regression','null_sanity']:
        subset=[r for r in rows if r['cohort']==cohort];summary={}
        for m in ['old_ci','point','joint']:
            good=wrong=declared=false=0
            for r in subset:
                rep=(r['report'] if m=='joint' else parents[r['id']]['reports']['cal_100000'] if m=='old_ci' else points[r['id']]['reports']['point_irf'])
                ok=correct(r,rep);good+=ok;declared+=rep['declared'];wrong+=r['case']['secondary']>0 and rep['declared'] and not ok;false+=r['case']['secondary']==0 and rep['declared']
            summary[m]=dict(correct=good,wrong=wrong,false=false,declared=declared,n=len(subset))
        result[cohort]=summary
    a=dict(status='SELECTED_CLOSED_FAILURE_ANALYSIS_COMPLETE',cohorts=result,protocol_sha256=parent.base.sha(P),inference=p['inference'])
    (OUT/'analysis.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
