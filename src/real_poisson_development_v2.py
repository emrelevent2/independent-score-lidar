"""Technical rerun: same frozen real inputs, stable log-weight normalization.

V1 stopped on a real depth-boundary underflow; it is retained, not analyzed
as a completed accuracy run. No response parameter, seed or query changes.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,time
import numpy as np
from real_poisson_methods_v2 import WorkingPulse,evaluate_query
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'real_poisson_development_v1'
OUT=ROOT/'results'/'real_poisson_development_v2';P=ROOT/'REAL_POISSON_DEVELOPMENT_V2_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing technical-rerun snapshot retained')
    previous=ROOT/'REAL_POISSON_DEVELOPMENT_PROTOCOL.json';p=json.loads(previous.read_text())
    p.update(status='DEVELOPMENT technical rerun of unchanged real inputs after normalized donor weights underflowed inV1.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['real_poisson_development_v2.py','real_poisson_methods_v2.py','likelihood_baseline.py']},
        prepared_protocol_sha256=sha(previous),prepared_manifest_sha256=sha(DATA/'manifest.json'))
    p['notes'].append('Only implementation change: subtract max(log_weights) before exponentiation. Algebraically identical normalized weights; no scientific parameter, input, annotation, query or random seed changes.')
    P.write_text(json.dumps(p,indent=2));print('Frozen real developmentV2technical rerun',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DATA/'manifest.json')==p['prepared_manifest_sha256']
    return p


@lru_cache(maxsize=2)
def load_frame(index):return np.load(DATA/f'thin_{index}.npz')['histograms']


@lru_cache(maxsize=2)
def pulse(kind):return WorkingPulse(kind,np.load(DATA/'empirical_profile.npz')['profile'] if kind=='empirical' else None)


def worker(query,index,kind,digest):
    identifier=f'{kind}_{index}_{query["id"]}';path=OUT/(identifier+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    t=time.perf_counter();r=evaluate_query(load_frame(index),pulse(kind),query['row'],query['column'])
    result=dict(id=identifier,query_id=query['id'],frame_index=index,kind=kind,protocol_sha256=digest,**r,elapsed_seconds=time.perf_counter()-t)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path);return result


def run(workers,pilot=False):
    p=verify();digest=sha(P);m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==p['prepared_protocol_sha256']
    assert sha(DATA/'empirical_profile.npz')==m['profile_sha256']
    for entry in m['inputs']:assert sha(DATA/f'thin_{entry["index"]}.npz')==entry['sha256']
    queries=m['queries'];OUT.mkdir(exist_ok=True)
    if pilot:queries=[q for split in ['normalization','calibration','evaluation_one','evaluation_two'] for q in [x for x in queries if x['split']==split][:2]]
    start=time.perf_counter();done=0
    for index in range(3):
        for kind in p['settings']['kinds']:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(worker,q,index,kind,digest) for q in queries]
                for job in as_completed(jobs):
                    job.result();done+=1
                    if done%100==0:print('Real devV2',done,'/',len(queries)*6,'seconds',round(time.perf_counter()-start,1),flush=True)
    (OUT/('pilot_completion.json' if pilot else 'completion.json')).write_text(json.dumps(dict(protocol_sha256=digest,cases=done,elapsed_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','pilot','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else run(args.workers,args.action=='pilot')
