"""Fixed empirical policies on two annotation-selected held-out real frames.

Freeze all methods and their development thresholds BEFORE acquiring counts.
No acquisition cycles, metric timing, or physical-negative labels are inferred.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,time,zipfile
import numpy as np
from real_poisson_methods_v2 import WorkingPulse,evaluate_query
from anchored_poisson import evaluate_anchored
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'data/real_poisson_development_v1';DEVOUT=ROOT/'results/real_poisson_development_v2'
ANCHOR=ROOT/'results/anchored_development_v1';SELECT=ROOT/'data/stb_real_test_selection_more/selection.json'
DATA=ROOT/'data/real_multiframe_test_v1';OUT=ROOT/'results/real_multiframe_test_v1';P=ROOT/'REAL_POISSON_TEST_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['real_multiframe_test.py','real_multiframe_analysis.py','real_poisson_methods_v2.py','anchored_poisson.py',
    'real_poisson_analysis_v2.py','real_poisson_development_v2.py','likelihood_baseline.py','stb_remote_zip.py']


def freeze():
    if P.exists():raise RuntimeError('Existing held-out snapshot retained')
    a=json.loads((DEVOUT/'analysis.json').read_text());anch=json.loads((ANCHOR/'analysis.json').read_text())
    selection=json.loads(SELECT.read_text());assert selection['selected'][0]['sequence']==17
    assert not DATA.exists(),'Test input directory already exists; inspect chronology'
    selected=selection['selected']
    for item in selected:assert sha(ROOT/item['label_file_relative'])==item['local_sha256']
    files=[DEVOUT/'analysis.json',DEVOUT/'calibration.json',ANCHOR/'analysis.json',ANCHOR/'calibration.json',SELECT,DEV/'empirical_profile.npz']
    p=dict(status='Frozen AFTER development choice and BEFORE any selected test histogram acquisition or outcome.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        input_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},frames=selected,
        selected_kinds=a['selected_kinds'],anchored_selected=anch['selected'],
        development_calibration=json.loads((DEVOUT/'calibration.json').read_text()),
        anchored_calibration=json.loads((ANCHOR/'calibration.json').read_text()),
        settings=dict(probabilities=[.01,.025,.05],known_one_queries=1000,two_interval_queries=500,
            coordinate_seed=20261113,attenuation_seed=2026111300,alphas=[.005,.01],primary_alpha=.01,
            route_gate=3.5,disagreement_ratio=1.5,interval_tolerance_bins=2.,spatial_block_size=16,bootstrap_replicates=2000),
        endpoints=['Primary: Sequence17 first frame, one-to-one agreement with two released known intervals, at each attenuation separately.',
            'Secondary validation: Sequence3 middle frame, selected from annotations only; all frames reported, no favorable frame choice.',
            'Known-one-interval accepted-unmatched declarations; these are not proven physical false positives.',
            'Weak-interval agreement, primary agreement, duplicate-in-known-window and declarations not matching both known intervals remain separate.',
            'Paired16x16spatial-block bootstrap differences conditional on each frame; attenuation replicates are not independent scenes.'],
        notes=['All baseline response families selected independently by the same fixed development objective.',
            'Selected anchored variant uses query primary and thus has no donor-independence/e-value guarantee.',
            'If unchanged local fallback won development, report fallback explicitly; do not claim a new effective method.',
            'Every cutoff and score normalizer transfers unchanged from development; no test-dependent threshold fitting.',
            'Counts are numerically thinned full recorded histograms, not fresh physical low-exposure captures.',
            'Known1--13intervals only. Unknown14and Unlabeled0never become negative labels. Multiple windows originate from released peak-peeling labels.',
            'Integer temporal bins only; unresolved P1timing prevents defensible metre conversion.',
            'Only two selected ZIP members fetched with bounded range, CRC and memberSHA; no full archive hash claim.',
            'Single working Poisson model, numerical multistart fits; no sensor-calibrated error guarantee or FPGA performance claim.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen held-out frames',[(r['sequence'],r['position']) for r in selected],'anchored',p['anchored_selected']['id'],flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    for n,d in p['input_sha256'].items():assert sha(ROOT/n)==d,n
    for item in p['frames']:assert sha(ROOT/item['label_file_relative'])==item['local_sha256']
    return p


def select_queries(labels,settings,frame_index):
    assert labels.shape==(49152,672) and labels.dtype==np.uint8
    known=(labels>0)&(labels<14);starts=known&~np.pad(known[:,:-1],((0,0),(1,0)));ends=known&~np.pad(known[:,1:],((0,0),(0,1)))
    n=starts.sum(-1);pix,begin=np.nonzero(starts);epix,end=np.nonzero(ends);assert np.array_equal(pix,epix)
    one=np.flatnonzero(n==1);two=np.flatnonzero(n==2);j=np.searchsorted(pix,two);two=two[begin[j+1]-end[j]>=8]
    interior=lambda a:a[(a//256>=4)&(a//256<188)&(a%256>=4)&(a%256<252)]
    rng=np.random.default_rng(settings['coordinate_seed']+100*frame_index);queries=[]
    for split,ids,count in [('known_one',interior(one),settings['known_one_queries']),('two_interval',interior(two),settings['two_interval_queries'])]:
        chosen=rng.permutation(ids)[:count];assert len(chosen)==count
        for i,pixel in enumerate(chosen):
            j=int(np.searchsorted(pix,pixel));ks=[j] if split=='known_one' else [j,j+1]
            queries.append(dict(id=f'{split}_{i:04d}',split=split,pixel=int(pixel),row=int(pixel//256),column=int(pixel%256),
                windows=[[int(begin[k]),int(end[k])] for k in ks]))
    return queries


def prepare():
    p=verify();DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);frames=[]
    for fi,item in enumerate(p['frames']):
        name=f'Sequence{item["sequence"]}_{item["position"]}';directory=DATA/name;directory.mkdir(exist_ok=True)
        if (directory/'manifest.json').exists():
            m=json.loads((directory/'manifest.json').read_text());assert m['protocol_sha256']==sha(P);frames.append(dict(name=name,manifest_sha256=sha(directory/'manifest.json')));continue
        labels=np.load(ROOT/item['label_file_relative'],allow_pickle=False)['labels'];queries=select_queries(labels,p['settings'],fi)
        coordinate_file=directory/'query_coordinates.json'
        if coordinate_file.exists():assert json.loads(coordinate_file.read_text())==queries
        else:coordinate_file.write_text(json.dumps(queries,indent=2))
        member=f'p1/Sequence{item["sequence"]}/'+Path(item['member']).name.replace('_semantic.npy','.txt');target=directory/Path(member).name
        download_file=directory/'download.json'
        if download_file.exists():
            download=json.loads(download_file.read_text());assert download['member']==member and sha(target)==download['sha256']
        else:
            assert not target.exists(),'Unexpected unrecorded complete download'
            reader=RangeReader(BASE+'Histogram/p1.zip',156925687435,budget=128*1024*1024)
            with zipfile.ZipFile(reader) as z:
                info=z.getinfo(member);assert info.file_size<=150*1024*1024;partial=target.with_suffix('.partial')
                with z.open(info) as src,partial.open('wb') as dst:
                    while True:
                        chunk=src.read(1024*1024)
                        if not chunk:break
                        dst.write(chunk)
                assert partial.stat().st_size==info.file_size;partial.replace(target)
                download=dict(member=member,crc32=info.CRC,bytes=info.file_size,sha256=sha(target),remote_bytes_read=reader.transferred,whole_archive_hash_verified=False)
            download_file.write_text(json.dumps(download,indent=2))
        h=np.loadtxt(target,dtype=np.int64);assert h.shape==(49152,672) and np.all(h>=0) and h.max()<=np.iinfo(np.int32).max
        h=h.astype(np.int32).reshape(192,256,672);np.savez_compressed(directory/'frame_histograms.npz',histograms=h)
        for q in queries:
            if len(q['windows'])==2:q['weak_window']=int(np.argmin([h[q['row'],q['column'],a:b+1].sum() for a,b in q['windows']]))
        inputs=[]
        for i,prob in enumerate(p['settings']['probabilities']):
            seed=p['settings']['attenuation_seed']+100*fi+i;child=np.random.default_rng(seed).binomial(h,prob).astype(np.int32)
            path=directory/f'thin_{i}.npz';np.savez_compressed(path,histograms=child);inputs.append(dict(index=i,probability=prob,seed=seed,sha256=sha(path)))
        manifest=dict(protocol_sha256=sha(P),frame_index=fi,sequence=item['sequence'],position=item['position'],queries=queries,inputs=inputs,
            coordinate_sha256=sha(coordinate_file),download=download,frame_sha256=sha(directory/'frame_histograms.npz'),license='CC BY-NC-SA4.0; SP-TransientBench2026.')
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2));frames.append(dict(name=name,manifest_sha256=sha(directory/'manifest.json')))
        print('Prepared',name,'queries',len(queries),'downloaded compressed bytes',download['remote_bytes_read'],flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),frames=frames),indent=2))


@lru_cache(maxsize=1)
def frame(name,i):return np.load(DATA/name/f'thin_{i}.npz')['histograms']


@lru_cache(maxsize=2)
def pulse(kind):return WorkingPulse(kind,np.load(DEV/'empirical_profile.npz')['profile'] if kind=='empirical' else None)


def worker(name,q,i,kind,setting,digest):
    path=OUT/name/f'{kind}_{i}_{q["id"]}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    t=time.perf_counter();h=frame(name,i);b=evaluate_query(h,pulse(kind),q['row'],q['column'])
    anchored=evaluate_anchored(h,pulse(kind),q['row'],q['column'],b,setting['depth_sigma']) if setting['depth_sigma'] is not None and kind==setting['kind'] else None
    r=dict(query_id=q['id'],frame_name=name,attenuation_index=i,kind=kind,protocol_sha256=digest,base=b,anchored=anchored,elapsed_seconds=time.perf_counter()-t)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(r,allow_nan=False));temp.replace(path)


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P);done=0;t=time.perf_counter();expected=0
    kinds=sorted(set(p['selected_kinds'].values())|{p['anchored_selected']['kind']})
    for f in m['frames']:
        directory=DATA/f['name'];assert sha(directory/'manifest.json')==f['manifest_sha256'];fm=json.loads((directory/'manifest.json').read_text())
        (OUT/f['name']).mkdir(exist_ok=True)
        for i in range(3):
            assert sha(directory/f'thin_{i}.npz')==fm['inputs'][i]['sha256']
            for kind in kinds:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    jobs=[pool.submit(worker,f['name'],q,i,kind,p['anchored_selected'],sha(P)) for q in fm['queries']];expected+=len(jobs)
                    for job in as_completed(jobs):
                        job.result();done+=1
                        if done%500==0:print('Real test',done,'seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=done,expected=expected,elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
