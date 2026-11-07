"""New real-only development after donor-anchor failure, before held-out data."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,time
import numpy as np
from real_poisson_methods_v2 import WorkingPulse
from anchored_poisson import evaluate_anchored
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/real_poisson_development_v1'
BASE=ROOT/'results/real_poisson_development_v2';OUT=ROOT/'results/anchored_development_v1';P=ROOT/'ANCHORED_DEVELOPMENT_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing anchored protocol retained')
    assert not (ROOT/'REAL_POISSON_TEST_PROTOCOL.json').exists()
    p=dict(status='DEVELOPMENT after weak donor-only real results; query-primary anchoring has no independence guarantee.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['anchored_development.py','anchored_poisson.py','real_poisson_methods_v2.py']},
        manifest_sha256=sha(DATA/'manifest.json'),base_protocol_sha256=sha(ROOT/'REAL_POISSON_DEVELOPMENT_V2_PROTOCOL.json'),
        settings=dict(kind='gaussian',depth_sigmas=[4.,8.,16.],probabilities=[.01,.025,.05]),
        notes=['Same fixed899developmentqueries and three thinnedframes. Gaussian selected in prior response-family comparison.',
            'Original donor-only median anchor may lie between disconnected surfaces. Anchor here comes from query null fit.',
            'Three residual starts on the pooled histogram; every raw center still excluded from pooling.',
            'All test histograms remain unread. Further selection must remain developmental.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen anchored development',flush=True)


@lru_cache(maxsize=1)
def frame(i):return np.load(DATA/f'thin_{i}.npz')['histograms']


def worker(q,i,sigma,digest):
    path=OUT/f'sigma{int(sigma)}_{i}_{q["id"]}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    b=json.loads((BASE/f'gaussian_{i}_{q["id"]}.json').read_text());start=time.perf_counter()
    r=evaluate_anchored(frame(i),WorkingPulse(),q['row'],q['column'],b,sigma)
    result=dict(query_id=q['id'],frame_index=i,depth_sigma=sigma,protocol_sha256=digest,**r,elapsed_seconds=time.perf_counter()-start)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,allow_nan=False));temp.replace(path)


def run(workers):
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DATA/'manifest.json')==p['manifest_sha256'];m=json.loads((DATA/'manifest.json').read_text());OUT.mkdir(exist_ok=True)
    t=time.perf_counter();done=0;total=len(m['queries'])*9
    for i in range(3):
        assert sha(DATA/f'thin_{i}.npz')==m['inputs'][i]['sha256']
        for sigma in p['settings']['depth_sigmas']:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(worker,q,i,sigma,sha(P)) for q in m['queries']]
                for job in as_completed(jobs):
                    job.result();done+=1
                    if done%500==0:print('Anchored dev',done,'/',total,'seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(cases=total,protocol_sha256=sha(P),elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else run(args.workers)
