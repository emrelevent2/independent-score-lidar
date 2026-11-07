"""A prespecified extension from128 to1024 measured single-surface blocks."""
from pathlib import Path
import sys,json,sqlite3,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import calibrated_recovery_study as study
OUT=ROOT/'results/extended_measured_nulls_v1'
P=ROOT/'EXTENDED_MEASURED_NULL_PROTOCOL_20261107.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    assert not P.exists()
    p=dict(status='FIXED_BEFORE_ADDITIONAL_MEASURED_BLOCK_OUTCOMES',source_sha256=sha(__file__),
        parent_protocol_sha256=sha(study.P),total_blocks=1024,
        rationale='The first128 measured outcomes are already known (calibrated0,censored1). Extend the same deterministic block schedule to1024 to estimate measured false-alarm frequency more precisely; no threshold,model,bank or block selection changes.',
        source='OD0 acq2. First frame after event70400 plus1000-period buffer. Nonoverlapping1000-period blocks indexed0..1023; indices128..1023 are additional. The first64 blocks used for counterfactual injections remain fixed.',
        endpoints='All1024 measured declarations,exact binomial descriptive intervals and applicability; first128 and additional896 also reported separately. Serial dependence and single-scene scope are retained; no inference of measured two-surface accuracy.',workers=12)
    P.write_text(json.dumps(p,indent=2));print('Extended measured-null protocol frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());assert sha(__file__)==p['source_sha256'];assert sha(study.P)==p['parent_protocol_sha256'];return p


def run():
    p=check();OUT.mkdir(exist_ok=True);cfg=study.setting('measured_high')
    raw=loadmat(ROOT/'data/highflux_raw/2019_01_28/FGS_td048_2019_01_28_acq2.mat')
    frames=raw['frameNum'].ravel().astype(np.int64);phase=raw['detTime'].ravel().astype(float)/cfg['unit_ns']
    first=int(frames[70400])+1000;tasks=[]
    assert first+1024000<=frames[-1]
    for index in range(128,1024):
        start=first+index*1000;lo=np.searchsorted(frames,start);hi=np.searchsorted(frames,start+1000)
        observed=(frames[lo:hi]-start)*64.+phase[lo:hi];tasks.append((index,observed))
    start=time.perf_counter()
    with sqlite3.connect(OUT/'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done={json.loads(r[0])['index'] for r in db.execute('SELECT payload FROM records')}
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            futures=[pool.submit(study.measured_worker,t) for t in tasks if t[0] not in done]
            for i,future in enumerate(as_completed(futures),1):
                row=future.result();db.execute('INSERT INTO records VALUES(?,?)',(row['id'],json.dumps(row)));db.commit()
                if i%64==0:print('Additional measured blocks',i,'/896',flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(status='COMPLETE',additional=896,seconds=time.perf_counter()-start,protocol_sha256=sha(P)),indent=2))


def analyze():
    p=check();base=json.loads((study.OUT/'analysis.json').read_text());records=[]
    for path in [study.OUT/'records.sqlite3',OUT/'records.sqlite3']:
        with sqlite3.connect(path) as db:records.extend(json.loads(r[0]) for r in db.execute('SELECT payload FROM records') if json.loads(r[0])['kind']=='public')
    assert len(records)==1024 and len({r['index'] for r in records})==1024
    assert len({r['data_sha256'] for r in records})==1024
    rows=[]
    for method in ['calibrated','censored']:
        threshold=base['thresholds']['measured_high'][method]
        selected=[r for r in records if r['methods'][method]['applicable']]
        alarms=sum(r['methods'][method]['score']>threshold for r in selected)
        early=sum(r['index']<128 and r['methods'][method]['score']>threshold for r in selected)
        rows.append(dict(method=method,records=1024,applicable=len(selected),false_alarms=alarms,
            first128_alarms=early,additional896_alarms=alarms-early,interval=study.interval(alarms,len(selected))))
    result=dict(status='COMPLETE',protocol_sha256=sha(P),parent_analysis_sha256=sha(study.OUT/'analysis.json'),rows=rows)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':globals()[sys.argv[1]]()
