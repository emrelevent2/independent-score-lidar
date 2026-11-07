"""Fixed-model, held-out Sequence17 annotation-agreement experiment.

Freeze after development selection and before acquiring this histogram.
No acquisition cycles or complete physical-negative labels are invented.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,shutil,time,zipfile
import numpy as np
from real_poisson_methods_v2 import WorkingPulse,evaluate_query
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'data'/'real_poisson_development_v1'
DEVOUT=ROOT/'results'/'real_poisson_development_v2';SELECT=ROOT/'data'/'stb_real_test_selection'
DATA=ROOT/'data'/'real_poisson_test_v1';OUT=ROOT/'results'/'real_poisson_test_v1';P=ROOT/'REAL_POISSON_TEST_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['real_poisson_test.py','real_poisson_methods_v2.py','real_poisson_analysis_v2.py',
    'real_poisson_development_v2.py','likelihood_baseline.py','stb_remote_zip.py']


def freeze():
    if P.exists():raise RuntimeError('Existing held-out protocol retained')
    a=json.loads((DEVOUT/'analysis.json').read_text());cal=json.loads((DEVOUT/'calibration.json').read_text())
    selected=json.loads((SELECT/'selection.json').read_text())['selected'];assert len(selected)==1 and selected[0]['sequence']==17
    item=selected[0];assert sha(SELECT/item['local_numeric_file'])==item['local_sha256']
    assert not (DATA/'frame_histograms.npz').exists(),'Test histogram already present; investigate chronology'
    p=dict(status='Frozen after development selection and BEFORE held-out histogram acquisition or method outcomes.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        development_analysis_sha256=sha(DEVOUT/'analysis.json'),development_calibration_sha256=sha(DEVOUT/'calibration.json'),
        profile_sha256=sha(DEV/'empirical_profile.npz'),test_frame=item,
        selected_kinds=a['selected_kinds'],calibration=cal,
        settings=dict(probabilities=[.01,.025,.05],normalization_seed=20261113,known_one_queries=1000,two_interval_queries=500,
            alpha=[.005,.01],primary_alpha=.01,route_gate=3.5,disagreement_ratio=1.5,
            interval_tolerance_bins=2.,spatial_block_size=16,bootstrap_replicates=2000),
        endpoints=['Primary: declared two-return pair matches two known annotation intervals one-to-one, independent of brightness order.',
            'Known-one-interval: accepted-unmatched extra declaration rate, never relabeled as a physical false-positive rate.',
            'Weak-interval coverage uses the weaker full-measurement annotated interval and all actually reported returns.',
            'Primary-only interval agreement and duplicate-within-known-window declarations remain separate.',
            'Paired spatial-block bootstrap descriptive intervals conditional on this recorded test frame; no independent-scene generalization claim.'],
        notes=['All method kinds are selected independently on DEVELOPMENT by the same fixed utility; local baseline receives both-kind selection.',
            'Calibration thresholds transfer unchanged from development; held-out annotations never choose cutoffs or method parameters.',
            'Bernoulli attenuation of recorded histograms, not new low-exposure acquisitions or exact physical Poisson equivalence.',
            'Known classes1--13;14Unknown and0Unlabeled never become true negative labels.',
            'Time is evaluated in released integer bins. No P1metric conversion or P2ray/offset reuse.',
            'No physical conditional e-value guarantee. Working Poisson likelihoods and numerical multistart fits are empirical scores.',
            'Only the fixed member of the156GBarchive is fetched via bounded HTTPRange, with CRC and downloaded-memberSHA.',
            'No evaluation-driven method or threshold switching after this freeze.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen real test models',p['selected_kinds'],flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DEV/'empirical_profile.npz')==p['profile_sha256']
    assert sha(DEVOUT/'analysis.json')==p['development_analysis_sha256']
    assert sha(DEVOUT/'calibration.json')==p['development_calibration_sha256']
    return p


def prepare():
    p=verify()
    if (DATA/'manifest.json').exists():raise RuntimeError('Existing real-test inputs retained')
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);item=p['test_frame']
    labels=np.load(SELECT/item['local_numeric_file'],allow_pickle=False)['labels'];assert labels.shape==(49152,672)
    known=(labels>0)&(labels<14);starts=known&~np.pad(known[:,:-1],((0,0),(1,0)))
    ends=known&~np.pad(known[:,1:],((0,0),(0,1)));n=starts.sum(-1)
    pix,begin=np.nonzero(starts);epix,end=np.nonzero(ends);assert np.array_equal(pix,epix)
    one=np.flatnonzero(n==1);one=one[(one//256>=4)&(one//256<188)&(one%256>=4)&(one%256<252)]
    two=np.flatnonzero(n==2);j=np.searchsorted(pix,two)
    eligible=(begin[j+1]-end[j]>=8)&(two//256>=4)&(two//256<188)&(two%256>=4)&(two%256<252);two=two[eligible]
    rng=np.random.default_rng(p['settings']['normalization_seed'])
    chosen_one=rng.permutation(one)[:p['settings']['known_one_queries']];chosen_two=rng.permutation(two)[:p['settings']['two_interval_queries']]
    assert len(chosen_one)==1000 and len(chosen_two)==500
    # Coordinates are selected entirely from the fixed annotations before raw
    # counts are downloaded. The weaker-window field is evaluation metadata.
    queries=[]
    for split,ids in [('known_one',chosen_one),('two_interval',chosen_two)]:
        for index,pixel in enumerate(ids):
            j=int(np.searchsorted(pix,pixel));ks=[j] if split=='known_one' else [j,j+1]
            queries.append(dict(id=f'{split}_{index:04d}',split=split,pixel=int(pixel),row=int(pixel//256),column=int(pixel%256),windows=[[int(begin[k]),int(end[k])] for k in ks]))
    histogram_member='p1/Sequence17/'+Path(item['member']).name.replace('_semantic.npy','.txt')
    target=DATA/Path(histogram_member).name
    if target.exists():raise RuntimeError('Existing raw histogram needs explicit partial-download inspection')
    reader=RangeReader(BASE+'Histogram/p1.zip',156925687435,budget=128*1024*1024)
    with zipfile.ZipFile(reader) as z:
        info=z.getinfo(histogram_member);assert info.file_size<=150*1024*1024
        partial=target.with_suffix('.partial')
        with z.open(info) as src,partial.open('wb') as dst:
            while True:
                chunk=src.read(1024*1024)
                if not chunk:break
                dst.write(chunk)
        assert partial.stat().st_size==info.file_size;partial.replace(target)
        download=dict(member=histogram_member,crc32=info.CRC,bytes=info.file_size,sha256=sha(target),
            remote_bytes_read=reader.transferred,whole_archive_hash_verified=False)
    h=np.loadtxt(target,dtype=np.int64);assert h.shape==(49152,672) and np.all(h>=0) and h.max()<=np.iinfo(np.int32).max
    h=h.astype(np.int32).reshape(192,256,672);np.savez_compressed(DATA/'frame_histograms.npz',histograms=h)
    for q in queries:
        if len(q['windows'])==2:
            strengths=[int(h[q['row'],q['column'],a:b+1].sum()) for a,b in q['windows']]
            q['weak_window']=int(np.argmin(strengths))
    inputs=[]
    for index,prob in enumerate(p['settings']['probabilities']):
        seed=2026111300+index;child=np.random.default_rng(seed).binomial(h,prob).astype(np.int32)
        path=DATA/f'thin_{index}.npz';np.savez_compressed(path,histograms=child)
        inputs.append(dict(index=index,probability=prob,seed=seed,sha256=sha(path)))
    shutil.copyfile(DEV/'empirical_profile.npz',DATA/'empirical_profile.npz')
    manifest=dict(protocol_sha256=sha(P),queries=queries,inputs=inputs,download=download,
        frame_sha256=sha(DATA/'frame_histograms.npz'),license='CC BY-NC-SA4.0; cite Dong et al.,SP-TransientBench2026.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Prepared held-out real frame and1500queries',flush=True)


@lru_cache(maxsize=2)
def frame(index):return np.load(DATA/f'thin_{index}.npz')['histograms']


@lru_cache(maxsize=2)
def pulse(kind):return WorkingPulse(kind,np.load(DATA/'empirical_profile.npz')['profile'] if kind=='empirical' else None)


def worker(q,index,kind,digest):
    identifier=f'{kind}_{index}_{q["id"]}';path=OUT/(identifier+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    t=time.perf_counter();r=evaluate_query(frame(index),pulse(kind),q['row'],q['column'])
    result=dict(id=identifier,query_id=q['id'],frame_index=index,kind=kind,protocol_sha256=digest,**r,elapsed_seconds=time.perf_counter()-t)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(path);return result


def run(workers):
    p=verify();digest=sha(P);m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==digest
    assert sha(DATA/'empirical_profile.npz')==p['profile_sha256']
    for entry in m['inputs']:assert sha(DATA/f'thin_{entry["index"]}.npz')==entry['sha256']
    kinds=sorted(set(p['selected_kinds'].values()));total=len(m['queries'])*3*len(kinds);done=0;t=time.perf_counter()
    for index in range(3):
        for kind in kinds:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(worker,q,index,kind,digest) for q in m['queries']]
                for job in as_completed(jobs):
                    job.result();done+=1
                    if done%250==0 or done==total:print('Real held-out',done,'/',total,'seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=digest,cases=total,kinds=kinds,elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
