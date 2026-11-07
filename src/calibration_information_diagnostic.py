"""CLOSED cohort information diagnostic,not an algorithm improvement trial."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,json,sqlite3,time,hashlib
import numpy as np
import physical_ordered_confirmation as parent
from pilot import first_detection_probabilities
from ordered_calibration import query_predictor
from background_lobe_confidence import lobe_confidence
from fixed_irf_control import fixed_proposal,fixed_components
from shoulder_alternative import numeric_and_sparse
from certified_ratio_screen import sparse_numerator_upper

ROOT=parent.ROOT;P=ROOT/'CALIBRATION_INFORMATION_DIAGNOSTIC_PROTOCOL.json';OUT=ROOT/'results/calibration_information_diagnostic_v1'
METHODS=['point_ci_100k','point_100k','point_ci_1m','point_1m','known_eta']


def graph():
    todo=[Path(__file__).name];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for n in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for mod in names:
                local=mod.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:todo.append(local)
    return {f'src/{n}':parent.base.original.sha(ROOT/'src'/n) for n in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic protocol')
    old=parent.check();assert json.loads((parent.OUT/'integrity.json').read_text())['status']=='PASS'
    p=dict(status='FROZEN_BEFORE_EXTRA_CALIBRATION_DRAWS',sources=graph(),cases=old['cases'],parent_protocol_sha256=parent.base.original.sha(parent.P),parent_sqlite_sha256=parent.base.original.sha(parent.OUT/'records.sqlite3'),
        settings=dict(calibration_cycles_old=100000,calibration_cycles_new=1000000,extra_independent_cycles=900000,extra_seed_base=10036120600,query_cycles=1000,query_boxes=8192),
        question='On ALL768CLOSED queries,how much of the loss comes from finite calibration uncertainty versus point estimation error? Add independent calibration cycles only,keep query/donors unchanged,compare fixed known-eta control.',
        construction='Original100k calibration plus900k independent multinomial cycles under same simulated optical response. New pointCI uses all1m calibration,delta=.001,queryalpha=.009. Point1m fixes estimate as exact,alpha=.01. Knowneta uses simulationtruthONLY as explicitly privileged reference,alpha=.01;not deployable.',
        exclusions='CLOSED information diagnostic,not independent efficacy;10x calibration exposure,not fair same-acquisition-budget algorithm gain;known-eta control NOT optimal detector/power ceiling;no real sensor,adaptive-stopping,FDR/FPGA/Q1claim.',
        report='All methods/groups/errors and detected calibration photons. Do not turn extra-photon gains into algorithm novelty. No outcome-based subset selection.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.original.sha(P),len(p['sources']),'sources',len(p['cases']),'cases',flush=True)


def check():
    p=json.loads(P.read_text());parent.check()
    assert parent.base.original.sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    for n,d in p['sources'].items():assert parent.base.original.sha(ROOT/n)==d,n
    return p


def worker(old):
    start=time.perf_counter();c=old['case'];h,cal,truth=parent.base.original.simulate_case(c)
    assert truth==old['truth'] and hashlib.sha256(h.tobytes()).hexdigest()==old['raw_histogram_sha256']
    t=np.arange(256);d=truth['calibration_delay'];w=truth['calibration_width'];eta=c['eta']
    rate=.01*(np.exp(-4*np.log(2)*((t-d)/w)**2)+eta*np.exp(-4*np.log(2)*((t-(d+16))/w)**2))+.005/256
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=10036120600+100000*g+i
    extra=np.random.default_rng(seed).multinomial(900000,first_detection_probabilities(rate))[:-1];newcal=cal[2]+extra
    cf=lobe_confidence(newcal,1000000);pred=query_predictor(h,newcal,1000000)
    methods=dict(point_ci_1m=parent.strong_control(h,pred,cf['eta_upper']),point_1m=parent.strong_control(h,pred,cf['eta_upper'],True))
    q=fixed_proposal(h,eta);model=fixed_components(q,eta);num=numeric_and_sparse(h[4,4],model)
    known=dict(fit=dict(eta=eta,scope='KnownsimulationetaONLY,privilegednondeployablecomparison'),proposal=q,numerator=num,numerator_upper=sparse_numerator_upper(h[4,4],model,num['indices']))
    methods['known_eta']=parent.strong_control(h,known,eta,True)
    reports=dict(point_ci_100k=old['reports']['point_ci'],point_100k=old['reports']['point']);reports.update({m:v['report'] for m,v in methods.items()})
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=old['raw_histogram_sha256'],extra_seed=seed,extra_histogram=extra.tolist(),calibration_histogram_1m=newcal.tolist(),
        confidence_1m=cf,predictor_1m=pred,known_eta_predictor=known,methods=methods,reports=reports,elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={x[0] for x in db.execute('SELECT id FROM records')};old={r['id']:r for r in parent.read()};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,old[c['id']]) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%96==0:print('Calibrationinformation',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=parent.base.original.sha(P),sqlite_sha256=parent.base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);groups={};totals={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[parent.base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    v=dict(status='CLOSED_INFORMATION_DIAGNOSTIC_COMPLETE',totals=totals,groups=groups,median_detected_calibration_photons_1m=float(np.median([sum(r['calibration_histogram_1m']) for r in rs])),protocol_sha256=parent.base.original.sha(P),scope=p['exclusions'])
    (OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:analyze()
