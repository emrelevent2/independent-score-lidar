"""Closed-record development assessment before seeking new real captures."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import argparse,datetime,hashlib,json,time
import numpy as np
from real_coverage_mixture import report
from real_poisson_analysis_v2 import route,threshold,endpoints
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/real_poisson_development_v1';OLD=ROOT/'results/real_poisson_development_v2'
OUT=ROOT/'results/real_coverage_development_v1';P=ROOT/'REAL_COVERAGE_DEVELOPMENT_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['real_coverage_mixture.py','real_coverage_development.py','real_poisson_analysis_v2.py','real_poisson_development_v2.py','real_poisson_methods_v2.py','likelihood_baseline.py']


def freeze():
    assert not P.exists() and not OUT.exists(),'Preserve prior snapshots'
    m=json.loads((DATA/'manifest.json').read_text());records={}
    for i in range(3):
        for q in m['queries']:
            name=f'gaussian_{i}_{q["id"]}.json';records[name]=sha(OLD/name)
    p=dict(status='DEVELOPMENT ONLY on closed Sequence7record. Frozen before any new mixture outputs.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={n:sha(ROOT/'src'/n) for n in SOURCES},
        manifest_sha256=sha(DATA/'manifest.json'),old_record_sha256=records,
        calibration_sha256=sha(OLD/'calibration.json'),simulation_coverage_protocol_sha256=sha(ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json'),
        settings=dict(amplitudes=[1,2,4,8,16,32],spatial_factors=[.5,1,2,4],support_gate=3.5,spatial_masses=['9/10','1/10'],
            retentions=[.01,.025,.05],alphas=[.005,.01],primary_alpha=.01,primary_retention=.025),
        methods=['coverage','local','guided','routed'],
        decision='No grid search. Continue to new prospective counts only if coverage mean across retentions of pair-rate minus known-one-unmatched-rate exceeds ordinary local fitting; otherwise retain the negative transfer and diagnose the fixed donor nuisance model.',
        notes=['All weights and components fixed by query-excluded donor fit, as in simulation architecture.',
            'Working Poisson numerical score minus numerical null fit; no analytic certificate or sensor calibration claim.',
            'Existing Gaussian response widths3--16bins and donor fit unchanged. No acquisition cycles inferred.',
            'Same prepared thinnings,queries and own-calibration opportunities for all controls.',
            'Histogram-derived annotation windows are not independent exhaustive physical truth.',
            'No new physical acquisition, new-scene generalization or independent efficacy claimed on this closed cohort.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen real coverage development',len(records),flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['sources'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DATA/'manifest.json')==p['manifest_sha256'] and sha(OLD/'calibration.json')==p['calibration_sha256']
    return p


@lru_cache(maxsize=1)
def frame(i):return np.load(DATA/f'thin_{i}.npz')['histograms']


def worker(q,i,digest,old_digest):
    path=OUT/f'{i}_{q["id"]}.json'
    if path.exists():assert json.loads(path.read_text())['protocol_sha256']==digest;return
    source=OLD/f'gaussian_{i}_{q["id"]}.json';assert sha(source)==old_digest;r=json.loads(source.read_text())
    h=frame(i)[q['row'],q['column']];coverage=report(h,r['proposal'],r['null']['log_likelihood'])
    result=dict(protocol_sha256=digest,query_id=q['id'],index=i,source_sha256=old_digest,coverage=coverage)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(path)


def run(workers):
    p=verify();OUT.mkdir(exist_ok=True);m=json.loads((DATA/'manifest.json').read_text());t=time.perf_counter();done=0
    for entry in m['inputs']:
        i=entry['index'];assert sha(DATA/f'thin_{i}.npz')==entry['sha256']
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,q,i,sha(P),p['old_record_sha256'][f'gaussian_{i}_{q["id"]}.json']) for q in m['queries']]
            for job in as_completed(jobs):
                job.result();done+=1
                if done%300==0:print('Real coverage development',done,'/2697',round(time.perf_counter()-t,1),flush=True)
    assert done==2697;(OUT/'completion.json').write_text(json.dumps(dict(cases=done,protocol_sha256=sha(P),elapsed_seconds=time.perf_counter()-t),indent=2))


def analyze():
    p=verify();assert json.loads((OUT/'completion.json').read_text())['cases']==2697
    m=json.loads((DATA/'manifest.json').read_text());calibration=json.loads((OLD/'calibration.json').read_text());summary={};cuts={};utilities={k:[] for k in p['methods']}
    for i,prob in enumerate(p['settings']['retentions']):
        rows={}
        for q in m['queries']:
            r=json.loads((OLD/f'gaussian_{i}_{q["id"]}.json').read_text());new=json.loads((OUT/f'{i}_{q["id"]}.json').read_text());assert new['protocol_sha256']==sha(P)
            rows[q['id']]={'coverage':new['coverage'],**{k:route(r,calibration[f'gaussian_{i}']['normalizers'],k) for k in ['local','guided','routed']}}
        for method in p['methods']:
            scores=[rows[q['id']][method]['score'] for q in m['queries'] if q['split']=='calibration']
            for alpha in p['settings']['alphas']:
                cut=threshold(scores,alpha);key=f'{i}_{alpha}_{method}';cuts[key]=cut
                groups={g:dict(n=0,**{k:0 for k in ['declared','pair','weak_window_agreement','accepted_unmatched','primary_agreement','duplicate_known_window']}) for g in ['evaluation_one','evaluation_two']}
                for q in m['queries']:
                    if q['split'] not in groups:continue
                    r=rows[q['id']][method];e=endpoints(q,r,r['score']>cut);g=groups[q['split']];g['n']+=1
                    for field,v in e.items():g[field]+=v
                summary[key]=dict(retention=prob,alpha=alpha,method=method,threshold=cut,groups=groups)
                if alpha==.01:
                    two=groups['evaluation_two'];one=groups['evaluation_one'];utilities[method].append(two['pair']/two['n']-one['accepted_unmatched']/one['n'])
                    print(prob,method,'pair',two['pair'],'/200','unmatchedone',one['accepted_unmatched'],'/100',flush=True)
    utility={k:float(np.mean(v)) for k,v in utilities.items()};proceed=utility['coverage']>utility['local']
    result=dict(status='DEVELOPMENT on closed real record; not independent efficacy.',protocol_sha256=sha(P),methods=summary,
        mean_pair_minus_known_one_unmatched_utility=utility,proceed_to_prospective_counts=proceed,
        decision='Proceed under frozen rule' if proceed else 'Negative development transfer retained; do not consume new count frames for this unchanged method.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'calibration.json').write_text(json.dumps(cuts,indent=2));print(result['decision'],utility,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    freeze() if a.action=='freeze' else run(a.workers) if a.action=='run' else analyze()
