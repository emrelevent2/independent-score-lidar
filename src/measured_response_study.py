"""Pre-specified grid illustration on four public measured histogram files.

There are no supplied truth maps. Counts are declarations under a model, not
verified detection accuracy. Previous ROI experiments were development.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,time
import numpy as np
from flint import arb
from read_rt3d import read_rbin
from measured_envelope import MeasuredResponse,measured_alternative_at,measured_evidence
from envelope_evidence import outward_float
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'rt3d_measured'
OUT=ROOT/'results'/'measured_response_v1';PROTOCOL=ROOT/'MEASURED_RESPONSE_PROTOCOL.json'
SOURCES=['measured_response_study.py','measured_envelope.py','envelope_evidence.py','read_rt3d.py']
FILES=['4_camouflage_3_2ms.rbin','5_camouflage_0_32ms.rbin','1_mannequin_head.rbin','2_mannequin_head_no_backplane.rbin']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if PROTOCOL.exists():raise RuntimeError('Existing protocol retained; never silently refreeze')
    cases=[]
    for file in FILES:
        coords=[(r,c) for r in range(8,151,18) for c in range(8,71,18)] if 'camouflage' in file else [(r,c) for r in range(8,137,32) for c in range(8,137,32)]
        for row,col in coords:cases.append(dict(id=f'{Path(file).stem}_{row}_{col}',file=file,row=row,column=col))
    out=dict(status='Prospective measured-data illustration settings, not ground-truth confirmation',
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},data_sha256={n:sha(DATA/n) for n in FILES},
        settings=dict(stretch_range=[.9,1.1],shape_tolerance=.05,initial_cells=32,max_boxes=256,
            gap_tolerance=.01,donor_size=9,amplitude_factors=[.5,1.,2.,4.],alpha=.01),
        cases=cases,notes=['No cycle counts invented; independent Poisson model as in the source paper.',
            'No supplied ground-truth surface labels; declarations are not true-positive counts.',
            'Shape tolerance is prescribed, not a measured simultaneous calibration confidence band.',
            'Nominal supplied IRF with continuous piecewise-linear interpolation and zero padding.',
            'Grid chosen geometrically; includes previously explored areas; do not claim untouched real test.',
            'The two camouflage exposures are not assumed registered independent acquisitions.',
            'Bonferroni illustration remains conditional on all stated photon/sensor model assumptions.'])
    PROTOCOL.write_text(json.dumps(out,indent=2));print('Frozen',len(cases),'measured queries',flush=True)


@lru_cache(maxsize=4)
def load(file):
    d=read_rbin(DATA/file);return d['histograms'][0],MeasuredResponse(d['irf'])


def worker(case,settings,protocol_sha):
    path=OUT/(case['id']+'.json')
    if path.exists():
        result=json.loads(path.read_text());assert result['protocol_sha256']==protocol_sha
        return result
    start=time.perf_counter();h,response=load(case['file']);row,col=case['row'],case['column']
    alternatives,parameters=measured_alternative_at(h,response,row,col,size=settings['donor_size'],
        stretch_range=settings['stretch_range'],factors=settings['amplitude_factors'])
    evidence=measured_evidence(h[row,col],alternatives,response,alpha=settings['alpha'],
        stretch_range=settings['stretch_range'],shape_tolerance=settings['shape_tolerance'],
        initial_cells=settings['initial_cells'],max_boxes=settings['max_boxes'],gap_tolerance=settings['gap_tolerance'])
    result=dict(**case,protocol_sha256=protocol_sha,parameters=parameters,evidence=evidence,
        observed_photons=int(h[row,col].sum()),declared=bool(evidence['declared'] and parameters['valid']),
        elapsed_seconds=time.perf_counter()-start)
    np.savez_compressed(OUT/(case['id']+'_likelihood_inputs.npz'),histogram=h[row,col],alternative_means=np.stack(alternatives))
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path)
    return result


def run(workers):
    p=json.loads(PROTOCOL.read_text())
    for n,digest in p['source_sha256'].items():assert sha(ROOT/'src'/n)==digest,n
    for n,digest in p['data_sha256'].items():assert sha(DATA/n)==digest,n
    OUT.mkdir(exist_ok=True);start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,p['settings'],sha(PROTOCOL)) for c in p['cases']]
        for i,j in enumerate(as_completed(jobs),1):
            records.append(j.result())
            if i%6==0 or i==len(jobs):print(i,'/',len(jobs),'elapsed',round(time.perf_counter()-start,1),flush=True)
    summary={}
    for file in FILES:
        rows=[r for r in records if r['file']==file];threshold=outward_float((arb(len(rows))/arb('.05')).log())
        summary[file]=dict(queries=len(rows),declared_alpha01=sum(r['declared'] for r in rows),
            declared_conditional_bonferroni05=sum(r['parameters']['valid'] and r['evidence']['log_e_lower']>=threshold for r in rows),
            bonferroni_log_threshold_upper=threshold,median_photons=float(np.median([r['observed_photons'] for r in rows])),
            median_gap=float(np.median([r['evidence']['gap'] for r in rows])),
            max_gap=max(r['evidence']['gap'] for r in rows),fallback_cells=sum(r['evidence']['conservative_fallbacks'] for r in rows))
    (OUT/'summary.json').write_text(json.dumps(dict(status='Completed illustration; no ground-truth accuracy',protocol_sha256=sha(PROTOCOL),
        elapsed_seconds=time.perf_counter()-start,summaries=summary),indent=2));print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();freeze() if args.action=='freeze' else run(args.workers)
