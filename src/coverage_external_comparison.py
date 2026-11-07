"""Current fixed coverage method on the existing 579-case transfer cohort.

Retains all external outputs unchanged. This closed cohort is not a new
prospective efficacy test or an in-distribution reproduction of those models.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,math,sqlite3,time
import numpy as np
from coverage_confirmation import verify as coverage_verify
from coverage_evidence import evaluate_coverage
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/guided_confirmation_v1';OLD=ROOT/'results/guided_confirmation_v1'
OUT=ROOT/'results/coverage_external_v1';P=ROOT/'COVERAGE_EXTERNAL_COMPARISON_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    assert not P.exists() and not OUT.exists();method=coverage_verify()
    cohort=json.loads((ROOT/'POINTCLOUD_COMPARATOR_PROTOCOL.json').read_text())['cases']
    m=json.loads((DATA/'manifest.json').read_text());cases=[c for c in m['cases'] if c['id'] in cohort];assert len(cases)==579
    p=dict(status='Frozen before current coverage outputs on this closed,previously evaluated transfer cohort.',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        own_source_sha256=sha(Path(__file__)),coverage_protocol_sha256=sha(ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json'),settings=method['settings'],
        data_manifest_sha256=sha(DATA/'manifest.json'),baseline_analysis_sha256=sha(OLD/'analysis.json'),
        comparator_protocol_sha256=sha(ROOT/'POINTCLOUD_COMPARATOR_PROTOCOL.json'),cases=cases,
        proposal_record_sha256={c['id']:sha(OLD/(c['id']+'.json')) for c in cases},
        alpha_empirical=.005,alpha_analytic=.01,notes=['No new training,prediction,reporter or preprocessing for published controls.',
            'Own mixture and certificate settings unchanged from fresh confirmation; no transfer tuning.',
            'Every empirical method uses same199calibration/200testnull/180positive acquisitions,own strict rank cutoff floored at0.',
            'This is an added comparison on a closed transfer cohort,not independent validation or published-method SOTA reproduction.',
            'SQLite atomic records avoid duplicating hundreds of individual files under storage pressure.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen current579case coverage comparison',flush=True)


def verify():
    p=json.loads(P.read_text());assert p['own_source_sha256']==sha(Path(__file__));coverage_verify()
    assert sha(ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json')==p['coverage_protocol_sha256']
    assert sha(DATA/'manifest.json')==p['data_manifest_sha256'] and sha(OLD/'analysis.json')==p['baseline_analysis_sha256']
    assert sha(ROOT/'POINTCLOUD_COMPARATOR_PROTOCOL.json')==p['comparator_protocol_sha256'];return p


def worker(c,settings,digest):
    path=DATA/(c['id']+'.npz');assert sha(path)==c['sha256']
    assert sha(OLD/(c['id']+'.json'))==digest
    r=json.loads((OLD/(c['id']+'.json')).read_text());q=r['reports']['guided_fast']['proposal']
    h=np.load(path)['histogram'][8,8];e=evaluate_coverage(h,q,settings)
    return dict(id=c['id'],proposal=q,evidence=e,analytic=e['coverage_report'],
        empirical=dict(e['coverage_report'],score=e['coverage_fast_score']))


def run(workers):
    p=verify();OUT.mkdir(exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,protocol TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
    present={row[0] for row in db.execute('SELECT id FROM records')};todo=[c for c in p['cases'] if c['id'] not in present];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,p['settings'],p['proposal_record_sha256'][c['id']]) for c in todo]
        for i,job in enumerate(as_completed(jobs),1):
            r=job.result();raw=json.dumps(r,allow_nan=False);digest=hashlib.sha256(raw.encode()).hexdigest()
            db.execute('INSERT INTO records VALUES(?,?,?,?)',(r['id'],sha(P),raw,digest));db.commit()
            if i%100==0:print('Coverage external',i,'/',len(todo),round(time.perf_counter()-t,1),flush=True)
    assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0]==579;db.close()
    (OUT/'completion.json').write_text(json.dumps(dict(cases=579,protocol_sha256=sha(P),elapsed_seconds=time.perf_counter()-t),indent=2))


def analyze():
    p=verify();db=sqlite3.connect(OUT/'records.sqlite3');rows={}
    for identifier,digest,raw,d in db.execute('SELECT * FROM records'):
        assert digest==sha(P) and hashlib.sha256(raw.encode()).hexdigest()==d;rows[identifier]=json.loads(raw)
    db.close();assert len(rows)==579
    cal=[c for c in p['cases'] if c['split']=='calibration'];null=[c for c in p['cases'] if c['split']=='evaluation' and not c['metadata']['positive']];pos=[c for c in p['cases'] if c['metadata']['positive']]
    assert (len(cal),len(null),len(pos))==(199,200,180)
    cut=max(0.,sorted(rows[c['id']]['empirical']['score'] for c in cal)[math.ceil(200*.995)-1]);summary={}
    for method in ['analytic','empirical']:
        count=dict(correct=0,wrong_depth=0,null_false=0,pair=0)
        for c in null+pos:
            r=rows[c['id']][method];declared=r['declared'] if method=='analytic' else r['score']>cut
            if not c['metadata']['positive']:count['null_false']+=declared;continue
            m=c['metadata'];good=abs(r['delay']-m['secondary_delay'])<=2
            count['correct']+=bool(declared and good);count['wrong_depth']+=bool(declared and not good)
            count['pair']+=bool(declared and min(max(abs(r['primary']-m['primary_delay']),abs(r['delay']-m['secondary_delay'])),max(abs(r['delay']-m['primary_delay']),abs(r['primary']-m['secondary_delay'])))<=2)
        summary['coverage_'+method]=dict(**count,alpha=.01 if method=='analytic' else .005,threshold=None if method=='analytic' else cut)
    old=json.loads((OLD/'analysis.json').read_text())['paired']
    for key in ['guided_fast_cal0.005','complete_guided_likelihood_cal0.005','local_likelihood_cal0.005','caspi_cal0.005','unrolled_raw_cal0.005','unrolled_coates_cal0.005']:
        if key not in old:continue
        r=old[key];s=r['summaries'];summary[key]=dict(correct=s['positive']['localized']['count'],wrong_depth=s['positive']['wrong_depth']['count'],null_false=s['null']['declared']['count'],pair=s['positive']['unordered_pair']['count'],alpha=.005,threshold=r['threshold'])
    result=dict(status='Closed579case transfer comparison; not new prospective efficacy or general SOTA reproduction.',protocol_sha256=sha(P),methods=summary,denominators=dict(calibration=199,null=200,positive=180))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    freeze() if a.action=='freeze' else run(a.workers) if a.action=='run' else analyze()
