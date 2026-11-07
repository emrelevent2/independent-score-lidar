"""Closed location-only reassessment against the strong response-aware fit."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,json,sqlite3,time,hashlib
import numpy as np
import physical_ordered_confirmation as parent
from fixed_irf_control import fixed_components
from continuous_location import window_location,profile_location

ROOT=parent.ROOT;P=ROOT/'CONTINUOUS_LOCATION_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/continuous_location_development_v1'
METHODS=[f'{b}_{e}' for b in ['point','ci'] for e in ['old','window','profile']]


def graph():
    todo=[Path(__file__).name];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for n in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for mod in names:
                path=mod.split('.')[0]+'.py'
                if (ROOT/'src'/path).exists() and path not in seen:todo.append(path)
    return {f'src/{n}':parent.base.original.sha(ROOT/'src'/n) for n in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    old=parent.check();assert json.loads((parent.OUT/'integrity.json').read_text())['status']=='PASS'
    p=dict(status='FROZEN_BEFORE_LOCALIZATION_REASSESSMENT',sources=graph(),cases=old['cases'],parent_protocol_sha256=parent.base.original.sha(parent.P),parent_sqlite_sha256=parent.base.original.sha(parent.OUT/'records.sqlite3'),
        question='Can better location estimation repair strong-control errors WITHOUT changing any existence decision,query/calibration budget,or assuming designed16/64separation?',
        candidates=dict(window='Maximum working-posterior mass within existing±2metric; global posterior uniformly smoothed over unit-width delay cells; exact piecewise-linear window maximization; spatial proposal remains atom; ties prefer legacy',profile='For global heads only: local numerical two-pulse first-event fit of amplitudes/background/delay/width; donor primary interval±1.5,secondary intervallegacy±5.5,width4.5..5.5,admissibleseparation8;spatial heads retained'),
        settings=dict(calibration_cycles=100000,query_cycles=1000,location_tolerance=2),
        primary='Strong point-control correct/wrong location counts; confidence-interval variant reported separately. All768closedqueries,not just known errors.',
        selection='Strictly more correct AND fewer wrong than point_old,with CIwrong count <=ci_old. Eligible tie:largest point correct,thenpreferwindow. No independent efficacy claim; any selected method requires new frozen queries.',
        attribution='Working-posterior expected-loss decisions and numerical maximum likelihood are standard. No new statistical principle claimed. Location-only heuristic cannot change analytic existence guarantee but does not create location coverage.',
        exclusions='Closeddevelopment,not independentconfirmation;noQ1/SOTA,sensorvalidity,posteriorcalibration orglobalMLE claim. Previously observed oneerror motivates investigation,not subsetselection.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.original.sha(P),len(p['sources']),'sources',flush=True)


def check():
    p=json.loads(P.read_text());parent.check();assert parent.base.original.sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    for n,d in p['sources'].items():assert parent.base.original.sha(ROOT/n)==d,n
    return p


def worker(old):
    start=time.perf_counter();c=old['case'];h,_,truth=parent.base.original.simulate_case(c)
    assert hashlib.sha256(h.tobytes()).hexdigest()==old['raw_histogram_sha256'] and truth==old['truth']
    pred=old['full_calibration_predictor'];q=pred['proposal'];eta=pred['fit']['eta'];num=pred['numerator'];models=fixed_components(q,eta)
    window=window_location(h[4,4],models,num['location'])
    used=any(old['reports'][m]['declared'] for m in ['point','point_ci'])
    profile=profile_location(h[4,4],q,eta,num['location'],num['head']) if used else dict(secondary=num['location'],used=False,reason='No existence declaration in either comparator')
    reports={}
    for b,m in [('point','point'),('ci','point_ci')]:
        for e in ['old','window','profile']:
            rep=dict(old['reports'][m]);rep['secondary']=rep['secondary'] if e=='old' else (window if e=='window' else profile)['secondary'];reports[f'{b}_{e}']=rep
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=old['raw_histogram_sha256'],window=window,profile=profile,reports=reports,elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={r[0] for r in db.execute('SELECT id FROM records')};old={r['id']:r for r in parent.read()};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,old[c['id']]) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%192==0:print('Location reassessment',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=parent.base.original.sha(P),sqlite_sha256=parent.base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);totals={};groups={};errors={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[parent.base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
        err=np.array([r['reports'][m]['secondary']-r['truth']['secondary'] for r in rs if r['case']['secondary']>0 and r['reports'][m]['declared']]);errors[m]=dict(n=len(err),mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))))
    eligible=[e for e in ['window','profile'] if totals[f'point_{e}']['correct']>totals['point_old']['correct'] and totals[f'point_{e}']['wrong']<totals['point_old']['wrong'] and totals[f'ci_{e}']['wrong']<=totals['ci_old']['wrong']]
    selected=max(eligible,key=lambda e:totals[f'point_{e}']['correct']) if eligible else None
    for b in ['point','ci']:
        for r in rs:
            for e in ['window','profile']:
                a=dict(r['reports'][f'{b}_old']);v=dict(r['reports'][f'{b}_{e}']);a.pop('secondary');v.pop('secondary');assert a==v
    v=dict(status='CLOSED_LOCALIZATION_REASSESSMENT_COMPLETE',totals=totals,groups=groups,declared_positive_errors=errors,eligible=eligible,selected=selected,existence_reports_identical=True,protocol_sha256=parent.base.original.sha(P),scope=p['exclusions']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:analyze()
