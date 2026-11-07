"""Fresh development acquisitions for sensitivity/locality improvements.

All proposal variants share exactly the same certified null bound. Oracle
curves use truth only as an explicitly unavailable reference, never as a method.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,time,math,csv
import numpy as np
from flint import arb
from scipy.stats import beta
from confirmatory_scenes import make_case
from guided_alternatives import candidates
from envelope_evidence import continuous_null_bound,certified_log_likelihood,outward_float
from likelihood_baseline import local_likelihood_report,fitted_flux_report
from pilot import coates
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'improvement_development_v1';DATA=ROOT/'data'/'improvement_development_v1'
PROTOCOL=ROOT/'IMPROVEMENT_DEVELOPMENT_PROTOCOL_v1.json'
SOURCES=['improvement_development.py','guided_alternatives.py','envelope_evidence.py',
 'continuous_alternative.py','confirmatory_scenes.py','likelihood_baseline.py','pilot.py','crossfit.py','screen.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def designs():
    rows=[]
    for i in range(399):rows.append(dict(id=f'cal_{i:04d}',split='calibration',kind='null',scene_seed=620000+i,noise_seed=720000+i,secondary=6.))
    for i in range(100):rows.append(dict(id=f'null_{i:04d}',split='evaluation',kind='null',scene_seed=621000+i,noise_seed=721000+i,secondary=6.))
    for level_index,level in enumerate([3.,6.,10.]):
        for j in range(50):
            i=level_index*50+j
            rows.append(dict(id=f'positive_{i:04d}',split='evaluation',kind='positive',scene_seed=622000+i,noise_seed=722000+i,secondary=level))
    return rows


def freeze():
    if PROTOCOL.exists():raise RuntimeError('Existing development snapshot retained')
    p=dict(status='Development snapshot before execution; NOT final held-out confirmation',
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},cases=designs(),
        settings=dict(alpha=.01,cycles=1000,initial_cells=32,max_boxes=256,gap_tolerance=.01,fwhm_range=[4.5,5.5]),
        variants=['old9','local3','guided15','guided25','mixture_local_guided','mixture_three'],
        notes=['All alternative parameters and weights exclude raw center before any preprocessing.',
            'Mixture likelihood determines declaration; most likely component only determines reported delay.',
            'Local likelihood and raw controls calibrated on399 distinct null scenes, rank396.',
            'Oracle truth-reference alternatives are unavailable diagnostics and cannot be selected as methods.',
            'These seeds are distinct from the frozen original confirmation. New final confirmation is still required.'])
    PROTOCOL.write_text(json.dumps(p,indent=2));print('Snapshotted',len(p['cases']),'development acquisitions',flush=True)


def mix(terms):
    shift=max(terms)
    return outward_float(arb(shift)+(sum(((arb(x)-arb(shift)).exp() for x in terms),arb(0))/len(terms)).log(),False)


def oracle(meta,bins=256):
    t=np.arange(bins);width=meta['fwhm']
    def g(d):
        values=np.exp(-4*np.log(2)*((t-d)/width)**2);return values/values.sum()
    base=meta['primary_incident']*g(meta['primary_delay'])+meta['background_incident']/bins
    weak=meta['secondary_incident']*g(meta['secondary_delay'])
    return [np.maximum((base+f*weak)/1000,1e-12) for f in [.5,1.,2.,4.]]


def worker(item,protocol_sha):
    file=OUT/(item['id']+'.json')
    if file.exists():
        result=json.loads(file.read_text());assert result['protocol_sha256']==protocol_sha;return result
    start=time.perf_counter();h,meta,_=make_case(item['scene_seed'],item['noise_seed'],item['kind'],item['secondary'])
    inp=DATA/(item['id']+'.npz');np.savez_compressed(inp,histogram=h)
    reports=dict(local_likelihood=local_likelihood_report(h[8,8]),raw=fitted_flux_report(coates(h[8,8])))
    bound=None
    if item['split']=='evaluation':
        bound=continuous_null_bound(h[8,8],fwhm_range=(4.5,5.5),normalized_envelope=False)
        groups=candidates(h);q=oracle(meta)
        groups['oracle_reference']=dict(means=[q[1]],component_delays=[meta['secondary_delay']],valid=True)
        groups['oracle_mixture_reference']=dict(means=q,component_delays=[meta['secondary_delay']]*4,valid=True)
        cache={};threshold=outward_float(arb(100).log())
        for name,group in groups.items():
            terms=[]
            for mean in group['means']:
                if id(mean) not in cache:cache[id(mean)]=certified_log_likelihood(h[8,8],mean,1000,'first')
                terms.append(cache[id(mean)])
            numerator=mix(terms);log_e=outward_float(arb(numerator)-arb(bound['upper']),False)
            component=int(np.argmax(terms));delay=group['component_delays'][component]
            reports[name]=dict(score=log_e,delay=delay,declared=bool(log_e>=threshold and group['valid']),
                mixture_components=len(terms),selected_component=component,
                parameters=group.get('parameters'),oracle='oracle' in name)
    # All report scores are finite on this generator. Never serialize NaN.
    result=dict(**item,metadata=meta,reports=reports,null_bound=bound,input_sha256=sha(inp),
        protocol_sha256=protocol_sha,seconds=time.perf_counter()-start)
    tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(file)
    return result


def ci(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def analyze():
    p=json.loads(PROTOCOL.read_text());records=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in p['cases']]
    evaluation=[r for r in records if r['split']=='evaluation'];methods=sorted(evaluation[0]['reports'])
    result=dict(status='Development outcomes; not final confirmation',methods={},protocol_sha256=sha(PROTOCOL))
    for name in methods:
        cal=[r['reports'][name]['score'] for r in records if r['split']=='calibration' and name in r['reports']]
        threshold=float(np.sort(cal)[math.ceil(.99*(len(cal)+1))-1]) if cal else None
        scores=[]
        for row in evaluation:
            report=row['reports'][name];truth=row['metadata'];positive=truth['positive']
            declared=bool(report['declared']) if threshold is None else report['score']>threshold
            localized=bool(declared and positive and abs(report['delay']-truth['secondary_delay'])<=2.)
            scores.append(dict(id=row['id'],positive=positive,geometry=truth['geometry'],level=truth['secondary_incident'],
                declared=declared,localized=localized,wrong=bool(declared and positive and not localized)))
        null=[r for r in scores if not r['positive']];pos=[r for r in scores if r['positive']]
        fp=sum(r['declared'] for r in null);tp=sum(r['localized'] for r in pos)
        result['methods'][name]=dict(false=fp,null_n=len(null),false_ci95=ci(fp,len(null)),localized=tp,positive_n=len(pos),
            recall=tp/len(pos),recall_ci95=ci(tp,len(pos)),wrong=sum(r['wrong'] for r in pos),threshold=threshold,
            geometry={str(g):dict(n=sum(r['geometry']==g for r in pos),localized=sum(r['geometry']==g and r['localized'] for r in pos)) for g in range(5)},
            levels={str(v):dict(n=sum(r['level']==v for r in pos),localized=sum(r['level']==v and r['localized'] for r in pos)) for v in [3.,6.,10.]},records=scores)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for name,d in result['methods'].items():print(name,'false',d['false'],'/',d['null_n'],'localized',d['localized'],'/',d['positive_n'],'wrong',d['wrong'],flush=True)


def run(workers):
    p=json.loads(PROTOCOL.read_text())
    for n,digest in p['source_sha256'].items():assert sha(ROOT/'src'/n)==digest,n
    OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,sha(PROTOCOL)) for c in p['cases']]
        for i,j in enumerate(as_completed(jobs),1):
            j.result()
            if i%50==0 or i==len(jobs):print(i,'/',len(jobs),'seconds',round(time.perf_counter()-start,1),flush=True)
    analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','analyze']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers)
    else:analyze()
