"""New-count held-out test of the fixed guarded working-Poisson location."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,time,zipfile
import numpy as np
from real_poisson_methods_v2 import WorkingPulse,evaluate_query
from real_multiframe_test import select_queries
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];SELECT=ROOT/'data/stb_guarded_selection_v1/selection.json';RULE=ROOT/'REAL_GUARDED_RULE_FREEZE.json'
DATA=ROOT/'data/guarded_real_confirmation_v1';OUT=ROOT/'results/guarded_real_confirmation_v1';P=ROOT/'GUARDED_REAL_CONFIRMATORY_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['guarded_real_confirmation.py','guarded_real_analysis.py','guarded_real_policy.py','real_poisson_methods_v2.py',
    'real_poisson_analysis_v2.py','real_poisson_development_v2.py','likelihood_baseline.py','real_multiframe_test.py',
    'anchored_poisson.py','stb_remote_zip.py','stb_guarded_selection.py']


def freeze():
    assert not P.exists() and not DATA.exists(),'Existing test/input snapshot retained'
    selection=json.loads(SELECT.read_text());assert selection['rule_sha256']==sha(RULE);rule=json.loads(RULE.read_text())
    assert selection['selected'],'No eligible new frame; counts must not be downloaded'
    for n,d in rule['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    frames=selection['selected']
    for frame in frames:assert sha(ROOT/frame['label_file_relative'])==frame['local_sha256']
    calfile=ROOT/'results/real_poisson_development_v2/calibration.json';assert sha(calfile)==rule['development_calibration_sha256']
    p=dict(status='Frozen final guarded real test BEFORE any selected count acquisition or outcome.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        selection_sha256=sha(SELECT),rule_sha256=sha(RULE),frames=frames,calibration=json.loads(calfile.read_text()),
        calibration_sha256=sha(calfile),settings=dict(probabilities=[.01,.025,.05],primary_retention=.025,
            primary_alpha=.01,alphas=[.005,.01],coordinate_seed=20261116,attenuation_seed=2026111600,
            known_one_queries=1000,two_interval_cap=500,spatial_block_size=16,bootstrap_replicates=5000,bootstrap_seed=20261117),
        endpoints=['Primary: equal-frame mean of guarded-minus-local annotated pair agreement at2.5%retention,1%nominal score cutoff.',
            'All other retentions and cutoffs secondary; report every selected frame and policy without outcome selection.',
            'Known-one accepted-unmatched,accepted-incomplete pair,weak-window agreement and primary agreement remain separate.',
            'Exact local/guarded score,threshold and declaration identity must hold for every query.',
            'Approximate conditional-frame spatial block-weighting intervals:Exp(1)weights per16x16pixel block,shared across frames of the same sequence.'],
        notes=['Same simulation-derived4bin,.5raw-score,.5blend,3.5support constants; no new real tuning.',
            'Working-Poisson support replaces the first-event count-noise proxy. No acquisition-cycle count is inferred.',
            'Existing development Gaussian normalizers and cutoffs transfer unchanged for every policy.',
            'New captures of already seen sequences are not independently new scenes; conditional frame analysis only.',
            'Labels are histogram-derived semantic intervals,not exhaustive independent physical hidden-surface truth.',
            'Numerical count thinning is not a fresh physical low-exposure capture; no image-wide guarantee or FPGA claim.',
            'No posterior best-frame/best-retention/best-method switch. An adverse or zero result is retained.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen final real test',[(f['sequence'],f['position']) for f in frames],flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(SELECT)==p['selection_sha256'] and sha(RULE)==p['rule_sha256']
    assert sha(ROOT/'results/real_poisson_development_v2/calibration.json')==p['calibration_sha256']
    for f in p['frames']:assert sha(ROOT/f['label_file_relative'])==f['local_sha256']
    return p


def query_settings(p,item):
    return dict(known_one_queries=p['settings']['known_one_queries'],two_interval_queries=min(p['settings']['two_interval_cap'],item['eligible_two_window_pixels']),
        coordinate_seed=p['settings']['coordinate_seed'])


def prepare():
    p=verify();DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);entries=[]
    assert not (DATA/'manifest.json').exists(),'Completed prepared inputs retained'
    for fi,item in enumerate(p['frames']):
        name=f'Sequence{item["sequence"]}_{item["position"]}';directory=DATA/name;directory.mkdir(exist_ok=True)
        if (directory/'manifest.json').exists():
            fm=json.loads((directory/'manifest.json').read_text());assert fm['protocol_sha256']==sha(P);entries.append(dict(name=name,manifest_sha256=sha(directory/'manifest.json')));continue
        labels=np.load(ROOT/item['label_file_relative'],allow_pickle=False)['labels'];queries=select_queries(labels,query_settings(p,item),fi)
        coordinates=directory/'query_coordinates.json'
        if coordinates.exists():assert json.loads(coordinates.read_text())==queries
        else:coordinates.write_text(json.dumps(queries,indent=2))
        member=f'p1/Sequence{item["sequence"]}/'+Path(item['member']).name.replace('_semantic.npy','.txt');target=directory/Path(member).name
        download_file=directory/'download.json'
        if download_file.exists():
            download=json.loads(download_file.read_text());assert download['member']==member and sha(target)==download['sha256']
        else:
            assert not target.exists(),'Unrecorded complete raw file requires inspection'
            reader=RangeReader(BASE+'Histogram/p1.zip',156925687435,budget=128*1024*1024)
            with zipfile.ZipFile(reader) as z:
                info=z.getinfo(member);assert info.file_size<=150*1024*1024;partial=target.with_suffix('.partial')
                with z.open(info) as src,partial.open('wb') as dst:
                    while True:
                        chunk=src.read(1024*1024)
                        if not chunk:break
                        dst.write(chunk)
                assert partial.stat().st_size==info.file_size;partial.replace(target)
                download=dict(member=member,bytes=info.file_size,crc32=info.CRC,sha256=sha(target),remote_bytes_read=reader.transferred,whole_archive_hash_verified=False)
            download_file.write_text(json.dumps(download,indent=2))
        h=np.loadtxt(target,dtype=np.int64);assert h.shape==(49152,672) and h.min()>=0 and h.max()<=np.iinfo(np.int32).max
        h=h.astype(np.int32).reshape(192,256,672);np.savez_compressed(directory/'frame_histograms.npz',histograms=h)
        for q in queries:
            if len(q['windows'])==2:q['weak_window']=int(np.argmin([h[q['row'],q['column'],a:b+1].sum() for a,b in q['windows']]))
        inputs=[]
        for i,probability in enumerate(p['settings']['probabilities']):
            seed=p['settings']['attenuation_seed']+100*fi+i;child=np.random.default_rng(seed).binomial(h,probability).astype(np.int32)
            path=directory/f'thin_{i}.npz';np.savez_compressed(path,histograms=child);inputs.append(dict(index=i,probability=probability,seed=seed,sha256=sha(path)))
        fm=dict(protocol_sha256=sha(P),sequence=item['sequence'],position=item['position'],frame_index=fi,queries=queries,inputs=inputs,
            coordinate_sha256=sha(coordinates),download=download,frame_sha256=sha(directory/'frame_histograms.npz'),
            license='SP-TransientBench,Dong et al.2026,CC BY-NC-SA4.0')
        (directory/'manifest.json').write_text(json.dumps(fm,indent=2));entries.append(dict(name=name,manifest_sha256=sha(directory/'manifest.json')))
        print('Prepared new guarded frame',name,'queries',len(queries),'compressed bytes',download['remote_bytes_read'],flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),frames=entries),indent=2))


@lru_cache(maxsize=1)
def frame(name,index):return np.load(DATA/name/f'thin_{index}.npz')['histograms']


@lru_cache(maxsize=1)
def pulse():return WorkingPulse('gaussian')


def worker(name,q,index,digest):
    path=OUT/name/f'{index}_{q["id"]}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    t=time.perf_counter();result=evaluate_query(frame(name,index),pulse(),q['row'],q['column'])
    r=dict(query_id=q['id'],frame_name=name,attenuation_index=index,protocol_sha256=digest,**result,elapsed_seconds=time.perf_counter()-t)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(r,allow_nan=False));temporary.replace(path)


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P);t=time.perf_counter();done=0;expected=0
    for entry in m['frames']:
        name=entry['name'];directory=DATA/name;assert sha(directory/'manifest.json')==entry['manifest_sha256'];fm=json.loads((directory/'manifest.json').read_text());(OUT/name).mkdir(exist_ok=True)
        for i in range(3):
            assert sha(directory/f'thin_{i}.npz')==fm['inputs'][i]['sha256']
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(worker,name,q,i,sha(P)) for q in fm['queries']];expected+=len(jobs)
                for job in as_completed(jobs):
                    job.result();done+=1
                    if done%500==0:print('Guarded new real',done,'seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=done,expected=expected,elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
