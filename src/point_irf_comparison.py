"""Prespecified point-IRF comparator on the same prospective acquisitions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
from flint import arb
import shoulder_confirmation as parent
from fixed_irf_control import estimated_eta,fixed_proposal,fixed_components,fixed_null_bound
from shoulder_alternative import numeric_and_sparse
from envelope_evidence import outward_float

ROOT=parent.ROOT;P=ROOT/'POINT_IRF_COMPARISON_PROTOCOL.json';OUT=ROOT/'results/point_irf_comparison_v1'


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    p=parent.check_protocol();sources=dict(p['sources'])
    for name in ['point_irf_comparison.py','fixed_irf_control.py']:
        sources['src/'+name]=parent.base.sha(ROOT/'src'/name)
    # Parent records are not queried here; freeze while only completion counts
    # are known. No point-control outcomes or confirmation aggregates inspected.
    protocol=dict(status='FROZEN_BEFORE_PARENT_AGGREGATES_OR_POINT_CONTROL_OUTCOMES',sources=sources,
        parent_protocol_sha256=parent.base.sha(parent.P),cases=p['cases'],
        method='Estimate eta from the independent100k-cycle single-surface calibration using fixed constrained IRF fit. Donors fit two surfaces with eta fixed to that estimate. Proper coverage mixture at exactly fixed eta; continuous fixed-eta null with32initial/256boxes; alpha.01.',
        comparison='Diagnostic stronger response-aware comparator on identical prospective cases/calibration budget; does not replace frozen primary endpoints or thresholds.',
        statistical_scope='Arb bounds certify the assumed fixed-eta likelihood. Treating its estimated eta as known does not have the finite-calibration physical-null error guarantee. No claim that the proposed method must beat this control in power.',
        no_tuning=True)
    P.write_text(json.dumps(protocol,indent=2));print('Frozen',parent.base.sha(P),flush=True)


def check_protocol():
    p=json.loads(P.read_text());parent.check_protocol()
    assert parent.base.sha(parent.P)==p['parent_protocol_sha256']
    for path,digest in p['sources'].items():assert parent.base.sha(ROOT/path)==digest,path
    return p


def worker(row):
    h,_,truth=parent.base.simulate_case(row['case']);query=h[4,4]
    assert hashlib.sha256(h.tobytes()).hexdigest()==row['raw_histogram_sha256']
    eta=estimated_eta(row['calibration_histogram']);q=fixed_proposal(h,eta)
    num=numeric_and_sparse(query,fixed_components(q,eta));cut=outward_float(arb(100).log())
    level=outward_float(arb(num['lower'])-arb(cut),False)
    bound=fixed_null_bound(query,eta,decision_level=level)
    score=outward_float(arb(num['lower'])-arb(bound['upper']),False)
    report=dict(declared=score>=cut,primary=q['primary'],secondary=num['location'],head=num['head'],
        log_e_lower=score,log_threshold_upper=cut,numerator_lower=num['lower'],null_upper=bound['upper'])
    return dict(id=row['id'],case=row['case'],truth=truth,eta_hat=eta,proposal=q,numerator=num,bound=bound,
        reports=dict(point_irf=report),raw_histogram_sha256=row['raw_histogram_sha256'])


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def run():
    p=check_protocol();assert json.loads((parent.OUT/'completion.json').read_text())['status']=='COMPLETE'
    start=time.perf_counter();db0=sqlite3.connect(parent.OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db0.execute('SELECT data FROM records ORDER BY id')];db0.close()
    assert len(rows)==len(p['cases']);db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    pending=[r for r in rows if r['id'] not in ids]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,r) for r in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%96==0:print('Point IRF',len(ids),'of',len(rows),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),
        invocation_seconds=time.perf_counter()-start,protocol_sha256=parent.base.sha(P),
        parent_sqlite_sha256=parent.base.sha(parent.OUT/'records.sqlite3'),sqlite_sha256=parent.base.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    check_protocol();db=connect();rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert len(rows)==768;groups={}
    for group in ['null_plain','null_lobe','far_two','near_two']:
        ep=[parent.base.endpoint(r,'point_irf') for r in rows if r['case']['group']==group]
        groups[group]={k:sum(e[k] for e in ep) for k in ep[0]}
    totals=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),
        wrong_location=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),
        false_objects=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    result=dict(status='COMPLETE',groups=groups,totals=totals,protocol_sha256=parent.base.sha(P),
        scope='Prospectively specified diagnostic comparator; point calibration uncertainty ignored; no physical-null finite-calibration error guarantee asserted.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
