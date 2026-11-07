"""Separately frozen final study after development of foreground-guided evidence.

Keep all original confirmation/development files unchanged. No test outcome
summary is produced by this runner. Calibrated and certified decision rules
have different guarantees and must be named separately in all reports.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,time,math,datetime
import numpy as np
from scipy.io import savemat
from scipy.ndimage import uniform_filter
from guided_confirmatory_scenes import make_case
from guided_alternatives import guided_pool,fit_pool
from continuous_alternative import alternative_at,amplitude_components
from adaptive_evidence import adaptive_evidence,mixture_interval,feasible_local_null
from guided_likelihood import guided_likelihood_report
from envelope_evidence import robust_mixture_evidence,certified_log_likelihood,outward_float
from likelihood_baseline import local_likelihood_report,fitted_flux_report
from methods import depth_guided_pool
from pilot import coates
from confirmation_pipeline import clean
from flint import arb
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'guided_confirmation_v1';OUT=ROOT/'results'/'guided_confirmation_v1'
P=ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json'
SOURCES=['guided_confirmation.py','guided_confirmatory_scenes.py','guided_alternatives.py','adaptive_evidence.py',
 'guided_likelihood.py','envelope_evidence.py','continuous_alternative.py','likelihood_baseline.py','likelihood_evidence.py',
 'pilot.py','methods.py','crossfit.py','screen.py','joint_detector.py','confirmation_pipeline.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def designs():
    out=[]
    for split,count,scene0,noise0 in [('calibration',1999,830000,930000),('evaluation',2000,832000,932000)]:
        for i in range(count):
            family=int(np.random.default_rng(scene0+i+10000000).integers(0,2))
            out.append(dict(id=f'{"cal" if split=="calibration" else "null"}_{i:04d}',split=split,kind='null',
                scene_seed=scene0+i,noise_seed=noise0+i,secondary=6.,separation_family=family,caspi=i<(199 if split=='calibration' else 200)))
    for level_index,level in enumerate([3.,6.,10.]):
        for family in range(2):
            for j in range(150):
                i=level_index*300+family*150+j
                out.append(dict(id=f'positive_{i:04d}',split='evaluation',kind='positive',scene_seed=834000+i,noise_seed=934000+i,
                    secondary=level,separation_family=family,caspi=j<30))
    return out


def freeze():
    if P.exists():raise RuntimeError('Existing final protocol retained')
    p=dict(status='Frozen before final inputs and outputs',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        matlab_sha256={n:sha(ROOT/n) for n in ['run_guided_confirmation_caspi.m','run_caspi_cycles.m']},
        settings=dict(spatial_sigma=1.5,depth_sigma=2.,donor_size=9,amplitude_factors=[.5,1.,2.,4.],
            analytic_alpha=.01,initial_cells=32,max_boxes=2048,calibration_alpha=[.005,.01],calibration_score_floor=0.),
        primary_practical_mode='guided_fast calibrated at0.005; marginal exchangeability guarantee with1999 labeled null acquisitions',
        primary_certified_mode='guided_certified at0.01; model-conditional guarantee; adaptive exact decision, no empirical threshold',
        sources_of_development=['IMPROVEMENT_DEVELOPMENT_PROTOCOL_v1.json','ADAPTIVE_DEVELOPMENT_PROTOCOL_v1.json'],
        cases=designs(),notes=[
            'All development seeds and the original confirmation are excluded from this study.',
            'Two separation strata:12--24 and48--80 bins;150 positives per photon level per stratum.',
            'Null separation families are independent random draws with the same rule in calibration and evaluation.',
            'All methods share observations; localized secondary tolerance2bins; wrong-depth declarations explicitly counted.',
            'Calibrated rules: strict score>max(0,ordered calibration score at ceil((m+1)*(1-alpha))).',
            'Calibrated modes do NOT inherit the analytic e-value conditional guarantee.',
            'CASPI fixed subset199cal,200testnull,30positives in each of6 level/separation strata (180 positives).',
            'For paired CASPI tables, recalibrate ALL calibrated methods on the SAME first199 calibration cases.',
            'Primary full-study controls: ordinary local likelihood, donor-guided likelihood, raw, box, depth-guided3/7.',
            'Uniform9 full256, uniform9 adaptive, local3 adaptive and guided full256 are declared ablations.',
            'Unavailable oracle true-rate distribution is a diagnostic only; never select or present it as a method.',
            'No test tuning. Exact rate intervals and paired acquisition comparisons; intervals are nominal unless specified.',
            'Additional2025pretrained comparator planned from pinned official source5895893e2b04ab3315b4c195189a688de04228fa.',
            'Its technical adapter must be frozen and validated before inspecting final query accuracy; pretrained transfer limitations must remain explicit.',
            'Earlier RT3D executable failures are not accuracy results and must not be reused as zero-recall comparisons.',
            'No full-image throughput, physical calibration or FPGA claim follows from this simulation study.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen',len(p['cases']),'new acquisitions',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,digest in p['source_sha256'].items():assert sha(ROOT/'src'/n)==digest,n
    for n,digest in p['matlab_sha256'].items():assert sha(ROOT/n)==digest,n
    return p


def prepare():
    p=verify()
    if (DATA/'manifest.json').exists():raise RuntimeError('Existing prepared inputs retained')
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);records=[]
    for i,item in enumerate(p['cases']):
        path=DATA/(item['id']+'.npz')
        if path.exists():raise RuntimeError('Input already exists; inspect partial preparation')
        limits=(12.,24.) if item['separation_family']==0 else (48.,80.)
        h,meta,_=make_case(item['scene_seed'],item['noise_seed'],item['kind'],item['secondary'],separation_range=limits)
        np.savez_compressed(path,histogram=h);records.append(dict(**item,metadata=meta,sha256=sha(path)))
        if (i+1)%1000==0:print('Prepared',i+1,'inputs',flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=records),indent=2))
    selected=[c for c in records if c['caspi']]
    for start in range(0,len(selected),50):
        batch=selected[start:start+50];h=np.stack([np.load(DATA/(c['id']+'.npz'))['histogram'] for c in batch],axis=-1)
        savemat(DATA/f'caspi_input_{start//50:02d}.mat',dict(observations=h,case_ids=np.asarray([c['id'] for c in batch],dtype=object)),do_compression=True)
    print('Prepared',len(records),'inputs and',len(selected),'CASPI inputs',flush=True)


def worker(item,digest,settings):
    path=OUT/(item['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    inp=DATA/(item['id']+'.npz');assert sha(inp)==item['sha256'];h=np.load(inp)['histogram'];start=time.perf_counter();reports={};timings={}
    t=time.perf_counter();pool,weights=guided_pool(h,8,8,spatial_sigma=settings['spatial_sigma'],depth_sigma=settings['depth_sigma'])
    params=fit_pool(pool);means=amplitude_components(params);timings['guided_proposal']=time.perf_counter()-t
    t=time.perf_counter()
    if item['split']=='evaluation':
        evidence=adaptive_evidence(h[8,8],means,alpha=settings['analytic_alpha'],initial_cells=settings['initial_cells'],max_boxes=settings['max_boxes'])
        witness=evidence['local_null_witness'];score=evidence['fast_calibration_score']
        reports['guided_certified']=dict(declared=bool(evidence['declared'] and params['valid']),delay=params['secondary'],evidence=evidence)
    else:
        numerator=mixture_interval(h[8,8],means);witness=feasible_local_null(h[8,8]);score=(numerator['lower']+numerator['upper'])/2-witness['lower']
    timings['guided_decision_or_score']=time.perf_counter()-t
    reports['guided_fast']=dict(score=score,delay=params['secondary'],proposal=dict(**params,**weights))
    t=time.perf_counter();reports['guided_likelihood']=guided_likelihood_report(h[8,8],params['secondary'],null_witness=witness);timings['guided_likelihood']=time.perf_counter()-t
    t=time.perf_counter();reports['local_likelihood']=local_likelihood_report(h[8,8]);timings['local_likelihood']=time.perf_counter()-t
    x=coates(h);fluxes=dict(raw=x,box3=uniform_filter(x,(3,3,1),mode='wrap'))
    for size in [3,7]:fluxes[f'depth_guided_{size}']=depth_guided_pool(h,spatial_size=size)
    for name,flux in fluxes.items():reports[name]=fitted_flux_report(flux[8,8])
    if item['split']=='evaluation':
        for size,name in [(9,'uniform9'),(3,'local3')]:
            _,p=alternative_at(h,8,8,size=size);alternatives=amplitude_components(p)
            r=adaptive_evidence(h[8,8],alternatives,max_boxes=settings['max_boxes'])
            reports[name+'_adaptive']=dict(declared=bool(r['declared'] and p['valid']),delay=p['secondary'],evidence=r)
            if size==9:
                full=robust_mixture_evidence(h[8,8],alternatives,normalized_envelope=False)
                reports['uniform9_full256']=dict(declared=bool(full['declared'] and p['valid']),delay=p['secondary'],evidence=full)
                log_e=outward_float(arb(reports['guided_certified']['evidence']['numerator_lower'])-arb(full['upper']),False)
                reports['guided_full256']=dict(declared=bool(log_e>=full['log_threshold_upper'] and params['valid']),delay=params['secondary'],score=log_e)
                # Unavailable true-rate distribution, solely a diagnostic.
                m=item['metadata'];tbin=np.arange(256)
                def pulse(center):
                    g=np.exp(-4*np.log(2)*((tbin-center)/m['fwhm'])**2);return g/g.sum()
                truth_rate=(m['primary_incident']*pulse(m['primary_delay'])+m['secondary_incident']*pulse(m['secondary_delay'])+m['background_incident']/256)/1000
                numerator=certified_log_likelihood(h[8,8],np.maximum(truth_rate,1e-12),1000,'first')
                oracle_log=outward_float(arb(numerator)-arb(full['upper']),False)
                reports['oracle_reference_full256']=dict(declared=bool(oracle_log>=full['log_threshold_upper']),delay=m['secondary_delay'],score=oracle_log,oracle=True)
    result=clean(dict(id=item['id'],protocol_sha256=digest,reports=reports,timings=timings,total_seconds=time.perf_counter()-start))
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path);return result


def calibrate(cases,settings):
    records=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases];result={}
    for name in records[0]['reports']:
        scores=sorted(-np.inf if r['reports'][name].get('score') is None else r['reports'][name]['score'] for r in records)
        result[name]={}
        for alpha in settings['calibration_alpha']:
            rank=math.ceil((len(scores)+1)*(1-alpha));threshold=max(0.,float(scores[rank-1]))
            result[name][str(alpha)]=dict(threshold=threshold,rank=rank,n=len(scores),strict_greater=True)
    return result


def evaluate(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P)
    start=time.perf_counter()
    for split in ['calibration','evaluation']:
        cases=[c for c in m['cases'] if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,sha(P),p['settings']) for c in cases]
            for i,j in enumerate(as_completed(jobs),1):
                j.result()
                if i%100==0 or i==len(jobs):print(split,i,'/',len(jobs),'seconds',round(time.perf_counter()-start,1),flush=True)
        if split=='calibration':
            thresholds=calibrate(cases,p['settings'])
            paired=calibrate([c for c in cases if c['caspi']],p['settings'])
            (OUT/'thresholds.json').write_text(json.dumps(dict(protocol_sha256=sha(P),full=thresholds,paired_caspi=paired),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(status='CPU completed; no outcome analysis yet',cases=len(m['cases']),elapsed_seconds=time.perf_counter()-start,protocol_sha256=sha(P)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','evaluate']);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='prepare':prepare()
    else:evaluate(args.workers)
