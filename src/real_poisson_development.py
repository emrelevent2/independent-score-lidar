"""Working-Poisson development on the selected real Sequence7 frame only.

All rates are agreement with released intervals. Unannotated returns are
not proven absent. No selected Sequence17 test histogram is accessed here.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,math,time
import numpy as np
from real_poisson_methods import WorkingPulse,evaluate_query
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'data'/'stb_multireturn_development'
REF=ROOT/'results'/'stb_multireturn_inventory_v1'/'reference_windows.npz'
DATA=ROOT/'data'/'real_poisson_development_v1';OUT=ROOT/'results'/'real_poisson_development_v1'
P=ROOT/'REAL_POISSON_DEVELOPMENT_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing development snapshot retained')
    p=dict(status='DEVELOPMENT real-count working-model comparison; not a sensor-calibrated false-positive study.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['real_poisson_development.py','real_poisson_methods.py','likelihood_baseline.py']},
        input_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in [SOURCE/'frame_histograms.npz',REF]},
        settings=dict(kinds=['gaussian','empirical'],probabilities=[.01,.025,.05],width_range=[3.,16.],
            profile_pixels=1000,normalization_pixels=200,calibration_pixels=399,evaluation_one=100,evaluation_two=200,
            alpha=[.005,.01],normalization_alpha=.01,route_gate=3.5,disagreement_ratio=1.5,profile_radius=24,selection_seed=20261112),
        notes=['No acquisition-cycle count is invented; unrestricted nonnegative count amplitudes and a working Poisson likelihood.',
            'Gaussian/empirical effective-response families allow widths3--16bins based on development annotation-width inventory, not test accuracy.',
            'Empirical pulse is estimated only from1000distinct bright known-one-interval DEVELOPMENT pixels; other query sets exclude those pixels.',
            'Unknown14/Unlabeled0never supply negative labels. One known window does not prove a single physical surface.',
            'Independent Bernoulli count thinning of recorded full histograms is controlled numerical attenuation, not newly acquired low-exposure data.',
            'All methods and both response kinds share exactly the same thinned frames and fixed query coordinates.',
            'Normalization and calibration queries are disjoint; overlap of neighboring windows remains, so no exchangeability/conditional sensor guarantee is asserted.',
            'Primary two-window endpoint matches both intervals one-to-one, independent of reported brightness order.',
            'Report weak-window agreement and unmatched declarations separately; no physical false-positive labels are invented.',
            'The selected Sequence17 test histogram remains unread. Development outcomes can select real working-model settings before a new frozen real test.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen real development',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    return p


def prepare():
    p=verify();settings=p['settings']
    for n,d in p['input_sha256'].items():assert sha(ROOT/n)==d,n
    if (DATA/'manifest.json').exists():raise RuntimeError('Existing development inputs retained')
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    h=np.load(SOURCE/'frame_histograms.npz')['histograms'];flat=h.reshape(-1,672);ref=np.load(REF)
    pix,start,end,n=ref['pixel'],ref['start'],ref['end'],ref['window_count']
    rng=np.random.default_rng(settings['selection_seed']);one=np.flatnonzero(n==1)
    one=one[(one//256>=4)&(one//256<188)&(one%256>=4)&(one%256<252)];one=rng.permutation(one)
    profile_ids=[];profiles=[];radius=settings['profile_radius']
    for pixel in one:
        j=int(np.searchsorted(pix,pixel));a,b=int(start[j]),int(end[j]);peak=a+int(np.argmax(flat[pixel,a:b+1]))
        if not(radius<=peak<672-radius and 3<=b-a+1<=25 and flat[pixel,a:b+1].sum()>=100):continue
        values=np.maximum(flat[pixel,peak-radius:peak+radius+1]-np.median(flat[pixel]),0.)
        if values.sum()<=0:continue
        profiles.append(values/values.sum());profile_ids.append(int(pixel))
        if len(profile_ids)==settings['profile_pixels']:break
    assert len(profile_ids)==settings['profile_pixels']
    profile=np.median(np.stack(profiles),axis=0);profile[profile<.002*profile.max()]=0;profile/=profile.sum()
    np.savez_compressed(DATA/'empirical_profile.npz',profile=profile,source_pixels=np.array(profile_ids))
    one=one[~np.isin(one,profile_ids)];queries=[];offset=0
    for split,count in [('normalization',settings['normalization_pixels']),('calibration',settings['calibration_pixels']),('evaluation_one',settings['evaluation_one'])]:
        for i,pixel in enumerate(one[offset:offset+count]):
            j=int(np.searchsorted(pix,pixel));queries.append(dict(id=f'{split}_{i:04d}',split=split,pixel=int(pixel),row=int(pixel//256),column=int(pixel%256),windows=[[int(start[j]),int(end[j])]]))
        offset+=count
    two=rng.permutation(ref['eligible_two_window_pixels'])[:settings['evaluation_two']]
    for i,pixel in enumerate(two):
        j=int(np.searchsorted(pix,pixel));windows=[[int(start[k]),int(end[k])] for k in [j,j+1]]
        strengths=[int(flat[pixel,a:b+1].sum()) for a,b in windows]
        queries.append(dict(id=f'evaluation_two_{i:04d}',split='evaluation_two',pixel=int(pixel),row=int(pixel//256),column=int(pixel%256),windows=windows,weak_window=int(np.argmin(strengths))))
    inputs=[]
    for index,prob in enumerate(settings['probabilities']):
        seed=2026111200+index;thinned=np.random.default_rng(seed).binomial(h,prob).astype(np.int32)
        path=DATA/f'thin_{index}.npz';np.savez_compressed(path,histograms=thinned)
        inputs.append(dict(index=index,probability=prob,seed=seed,sha256=sha(path)))
    manifest=dict(protocol_sha256=sha(P),queries=queries,inputs=inputs,profile_sha256=sha(DATA/'empirical_profile.npz'))
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Prepared',len(queries),'queries x3attenuations x2response families',flush=True)


@lru_cache(maxsize=2)
def load_frame(index):return np.load(DATA/f'thin_{index}.npz')['histograms']


@lru_cache(maxsize=2)
def pulse(kind):
    return WorkingPulse(kind,np.load(DATA/'empirical_profile.npz')['profile'] if kind=='empirical' else None)


def worker(query,index,kind,digest):
    identifier=f'{kind}_{index}_{query["id"]}';path=OUT/(identifier+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    t=time.perf_counter();r=evaluate_query(load_frame(index),pulse(kind),query['row'],query['column'])
    result=dict(id=identifier,query_id=query['id'],frame_index=index,kind=kind,protocol_sha256=digest,
        **r,elapsed_seconds=time.perf_counter()-t)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path);return result


def run(workers,pilot=False):
    p=verify();digest=sha(P);m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==digest
    assert sha(DATA/'empirical_profile.npz')==m['profile_sha256']
    for entry in m['inputs']:assert sha(DATA/f'thin_{entry["index"]}.npz')==entry['sha256']
    queries=m['queries']
    if pilot:queries=[q for split in ['normalization','calibration','evaluation_one','evaluation_two'] for q in [x for x in queries if x['split']==split][:2]]
    t=time.perf_counter();done=0
    for index in range(3):
        for kind in p['settings']['kinds']:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(worker,q,index,kind,digest) for q in queries]
                for job in as_completed(jobs):
                    job.result();done+=1
                    if done%100==0 or pilot:print('Real dev',done,'/',len(queries)*6,'seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/('pilot_completion.json' if pilot else 'completion.json')).write_text(json.dumps(dict(protocol_sha256=digest,cases=done,elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','pilot','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers,args.action=='pilot')
