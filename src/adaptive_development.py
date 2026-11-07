"""Development audit of certified stopping and optional calibrated fast mode.

The fast calibration score is NOT an e-value. Its guarantee is marginal under
exchangeability with labeled null acquisitions, like the calibrated controls.
The analytic mode retains the model-conditional e-value threshold100.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time,argparse,math
import numpy as np
from guided_alternatives import guided_pool,fit_pool
from continuous_alternative import amplitude_components
from adaptive_evidence import adaptive_evidence
from improvement_development import ci
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'improvement_development_v1'
DATA=ROOT/'data'/'improvement_development_v1';OUT=ROOT/'results'/'adaptive_development_v1'
P=ROOT/'ADAPTIVE_DEVELOPMENT_PROTOCOL_v1.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['adaptive_development.py','adaptive_evidence.py','guided_alternatives.py','envelope_evidence.py','continuous_alternative.py','likelihood_baseline.py']


def freeze():
    if P.exists():raise RuntimeError('Existing development snapshot retained')
    original=json.loads((ROOT/'IMPROVEMENT_DEVELOPMENT_PROTOCOL_v1.json').read_text())
    out=dict(status='Development extension, not final confirmation',cases=original['cases'],
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        settings=dict(spatial_sigma=1.5,depth_sigma=2.,alpha=.01,initial_cells=32,max_boxes=2048),
        fast_mode='Calibrate q-mixture log likelihood minus feasible numerical null log likelihood; never call it analytic e-value.',
        calibrated_alpha=[.005,.01],calibrated_threshold_floor=0.,
        notes=['Same649 development observations retained byte-for-byte.',
            'Safe early rejection uses upper numerator and feasible lower null; acceptance uses lower numerator and a complete-cover upper null.',
            'Child bounds inherit valid parent caps; budget-undecided cases are not declared.',
            'Calibrated score threshold is max(0, order statistic) on399 labeled nulls; strict greater-than decision.',
            'Analytic threshold100 is unchanged. Final independent study remains required.'])
    P.write_text(json.dumps(out,indent=2));print('Snapshot retained development inputs',flush=True)


def worker(item,settings,digest):
    path=OUT/(item['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    base=json.loads((BASE/(item['id']+'.json')).read_text());inp=DATA/(item['id']+'.npz')
    assert sha(inp)==base['input_sha256'];h=np.load(inp)['histogram'];start=time.perf_counter()
    pool,weights=guided_pool(h,8,8,spatial_sigma=settings['spatial_sigma'],depth_sigma=settings['depth_sigma'])
    params=fit_pool(pool);means=amplitude_components(params)
    evidence=adaptive_evidence(h[8,8],means,alpha=settings['alpha'],initial_cells=settings['initial_cells'],max_boxes=settings['max_boxes'])
    out=dict(id=item['id'],protocol_sha256=digest,metadata=base['metadata'],parameters=dict(**params,**weights),
        evidence=evidence,declared=bool(evidence['declared'] and params['valid']),seconds=time.perf_counter()-start)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False));tmp.replace(path);return out


def analyze():
    p=json.loads(P.read_text());cases=p['cases'];rows={c['id']:json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases}
    bases={c['id']:json.loads((BASE/(c['id']+'.json')).read_text()) for c in cases}
    cal=[c for c in cases if c['split']=='calibration'];ev=[c for c in cases if c['split']=='evaluation'];methods={}
    for label in ['adaptive_guided','guided15_full256','old9']:
        methods[label]=dict(threshold=None,mode='analytic conditional model guarantee',records=[])
    for alpha in [.005,.01]:
        rank=math.ceil((len(cal)+1)*(1-alpha))
        for name in ['fast_guided','local_likelihood','raw']:
            scores=[rows[c['id']]['evidence']['fast_calibration_score'] if name=='fast_guided' else bases[c['id']]['reports'][name]['score'] for c in cal]
            threshold=max(0.,float(np.sort(scores)[rank-1]))
            methods[f'{name}_cal{alpha}']=dict(threshold=threshold,calibration_rank=rank,calibration_n=len(cal),
                mode='marginal exchangeability calibration; NOT analytic e-value guarantee',records=[])
    for case in ev:
        identifier=case['id'];r=rows[identifier];b=bases[identifier];truth=b['metadata']
        for name,m in methods.items():
            if name=='adaptive_guided':declared=r['declared'];delay=r['parameters']['secondary']
            elif name in ['guided15_full256','old9']:
                report=b['reports']['guided15' if name=='guided15_full256' else 'old9'];declared=report['declared'];delay=report['delay']
            else:
                control=name.split('_cal')[0]
                if control=='fast_guided':score=r['evidence']['fast_calibration_score'];delay=r['parameters']['secondary'];valid=r['parameters']['valid']
                else:score=b['reports'][control]['score'];delay=b['reports'][control]['delay'];valid=True
                declared=bool(score>m['threshold'] and valid)
            localized=bool(declared and truth['positive'] and abs(delay-truth['secondary_delay'])<=2.)
            m['records'].append(dict(id=identifier,positive=truth['positive'],declared=bool(declared),localized=localized,
                wrong=bool(declared and truth['positive'] and not localized),geometry=truth['geometry'],level=truth['secondary_incident']))
    for name,m in methods.items():
        null=[r for r in m['records'] if not r['positive']];pos=[r for r in m['records'] if r['positive']]
        fp=sum(r['declared'] for r in null);tp=sum(r['localized'] for r in pos)
        m.update(false=fp,null_n=len(null),localized=tp,positive_n=len(pos),wrong=sum(r['wrong'] for r in pos),
            false_ci95=ci(fp,len(null)),recall_ci95=ci(tp,len(pos)))
        print(name,'false',fp,'/',len(null),'localized',tp,'/',len(pos),'wrong',m['wrong'],flush=True)
    records=[rows[c['id']] for c in ev]
    diag=dict(n=len(records),decisions={key:sum(r['evidence']['decision']==key for r in records) for key in sorted(set(r['evidence']['decision'] for r in records))},
        boxes_quantiles=np.quantile([r['evidence']['boxes_evaluated'] for r in records],[0,.5,.9,.99,1]).tolist(),
        parallel_seconds_median=float(np.median([r['seconds'] for r in records])),note='Parallel-job timing, not isolated latency benchmark.')
    out=dict(status='Development outcomes; final independent confirmation required',protocol_sha256=sha(P),methods=methods,diagnostics=diag)
    (OUT/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(diag,flush=True)


def run(workers):
    p=json.loads(P.read_text())
    for n,digest in p['source_sha256'].items():assert sha(ROOT/'src'/n)==digest,n
    OUT.mkdir(exist_ok=True);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,p['settings'],sha(P)) for c in p['cases']]
        for i,j in enumerate(as_completed(jobs),1):
            j.result()
            if i%100==0 or i==len(jobs):print(i,'/',len(jobs),'seconds',round(time.perf_counter()-start,1),flush=True)
    analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','analyze']);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();freeze() if args.action=='freeze' else run(args.workers) if args.action=='run' else analyze()
