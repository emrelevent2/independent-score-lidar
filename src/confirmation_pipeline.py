"""Frozen-data pipeline: calibration, independent scenes and paired controls."""
from pathlib import Path
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.io import savemat
from scipy.ndimage import uniform_filter
from confirmatory_scenes import make_case
from continuous_alternative import alternative_at,amplitude_components
from envelope_evidence import robust_mixture_evidence
from likelihood_baseline import local_likelihood_report,fitted_flux_report
from pilot import coates
from methods import depth_guided_pool

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'confirmation_v1'
RESULTS=ROOT/'results'/'confirmation_v1'


def designs():
    out=[]
    for i in range(399):out.append(dict(id=f'cal_{i:04d}',split='calibration',kind='null',scene_seed=200000+i,noise_seed=300000+i,secondary=6.,caspi=i<199))
    for i in range(600):out.append(dict(id=f'null_{i:04d}',split='evaluation',kind='null',scene_seed=400000+i,noise_seed=500000+i,secondary=6.,caspi=i<200))
    for level_index,level in enumerate([3.,6.,10.]):
        for j in range(150):
            i=level_index*150+j
            out.append(dict(id=f'positive_{i:04d}',split='evaluation',kind='positive',scene_seed=410000+i,noise_seed=510000+i,secondary=level,caspi=j<60))
    return out


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value


def prepare():
    freeze=ROOT/'CONFIRMATORY_FREEZE.json'
    if not freeze.exists():raise RuntimeError('Freeze protocol and code before creating confirmation inputs')
    DATA.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True)
    records=[]
    for item in designs():
        path=DATA/(item['id']+'.npz')
        if path.exists():raise RuntimeError('Inputs already exist; do not regenerate completed data')
        h,metadata,_=make_case(item['scene_seed'],item['noise_seed'],item['kind'],item['secondary'])
        np.savez_compressed(path,histogram=h)
        records.append(dict(**item,metadata=metadata,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=dict(freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),cases=records,
                  input_count=len(records),status='inputs generated without detector outputs')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    selected=[r for r in records if r['caspi']]
    # Small MATLAB batches; query-only outputs avoid saving every denoised cube.
    for start in range(0,len(selected),50):
        batch=selected[start:start+50]
        h=np.stack([np.load(DATA/(r['id']+'.npz'))['histogram'] for r in batch],axis=-1)
        savemat(DATA/f'caspi_input_{start//50:02d}.mat',{'observations':h,'case_ids':np.array([r['id'] for r in batch],dtype=object)},do_compression=True)
    print('Prepared',len(records),'cases;',len(selected),'pre-specified CASPI cases',flush=True)


def worker(item):
    path=RESULTS/(item['id']+'.json')
    if path.exists():return item['id'],'retained'
    with np.load(DATA/(item['id']+'.npz')) as d:h=d['histogram']
    begin=time.perf_counter();reports={};times={}
    if item['split']=='evaluation':
        t=time.perf_counter();_,parameters=alternative_at(h,8,8,size=9)
        alt=amplitude_components(parameters)
        evidence=robust_mixture_evidence(h[8,8],alt,normalized_envelope=False)
        reports['proposed']=dict(score=evidence['log_e_lower'],delay=parameters['secondary'],
            declared=bool(evidence['declared'] and parameters['valid']),
            proposal_parameters=parameters,evidence=evidence)
        times['proposed']=time.perf_counter()-t
    t=time.perf_counter();reports['local_likelihood']=local_likelihood_report(h[8,8]);times['local_likelihood']=time.perf_counter()-t
    x=coates(h)
    inputs={'raw':x,'box3':uniform_filter(x,(3,3,1),mode='wrap')}
    for size in [3,7]:
        t=time.perf_counter();inputs[f'depth_guided_{size}']=depth_guided_pool(h,spatial_size=size);times[f'depth_guided_{size}_preprocessing']=time.perf_counter()-t
    for name,flux in inputs.items():
        t=time.perf_counter();reports[name]=fitted_flux_report(flux[8,8]);times[name+'_report']=time.perf_counter()-t
    result=clean(dict(id=item['id'],reports=reports,timings=times,total_seconds=time.perf_counter()-begin))
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(path)
    return item['id'],'computed'


def evaluate(workers=3):
    RESULTS.mkdir(exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    freeze=ROOT/'CONFIRMATORY_FREEZE.json'
    if manifest['freeze_sha256']!=hashlib.sha256(freeze.read_bytes()).hexdigest():raise RuntimeError('Freeze changed')
    frozen=json.loads(freeze.read_text())
    for name,sha in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed: '+name)
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,r) for r in manifest['cases']]
        for i,job in enumerate(as_completed(jobs),1):
            case,status=job.result()
            if i%25==0 or i==len(jobs):print(i,'/',len(jobs),case,status,'seconds',round(time.perf_counter()-begin,1),flush=True)
    (RESULTS/'completion.json').write_text(json.dumps(dict(cases=len(jobs),elapsed_seconds=time.perf_counter()-begin,
        freeze_sha256=manifest['freeze_sha256'],workers=workers),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','evaluate']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:evaluate(args.workers)
