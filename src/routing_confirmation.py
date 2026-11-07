"""Independently frozen test of empirical spatial/local routing.

Closed earlier cohorts are development for this rule. Current sources and
settings must be frozen before any of the new observations are generated.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,math,time
import numpy as np
from routing_scenes import make_routing_case
from guided_alternatives import guided_pool,fit_pool
from complete_guided_likelihood import complete_guided_likelihood
from likelihood_baseline import local_likelihood_report
from profile_proposals import profile_at_proposals
from support_routing import route_report
from disagreement_routing import disagreement_report
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'routing_confirmation_v1';OUT=ROOT/'results'/'routing_confirmation_v1'
P=ROOT/'ROUTING_CONFIRMATORY_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['routing_confirmation.py','routing_scenes.py','support_routing.py','disagreement_routing.py',
    'guided_confirmatory_scenes.py','guided_alternatives.py','complete_guided_likelihood.py',
    'likelihood_baseline.py','profile_proposals.py','adaptive_evidence.py','envelope_evidence.py',
    'continuous_alternative.py','likelihood_evidence.py','pilot.py','crossfit.py','screen.py','methods.py','joint_detector.py']


def designs():
    cases=[]
    for split,n,s0,n0 in [('calibration',3999,1430000,2430000),('evaluation',3000,1435000,2435000)]:
        for i in range(n):
            family=int(np.random.default_rng(s0+i+10000000).integers(0,2))
            cases.append(dict(id=f'{"cal" if split=="calibration" else "null"}_{i:04d}',split=split,
                kind='null',scene_seed=s0+i,noise_seed=n0+i,secondary=6.,separation_family=family,support_family='coherent'))
    for family,n_per_stratum,s0,n0,split in [('coherent',150,1440000,2440000,'evaluation'),
            ('singleton',100,1442000,2442000,'evaluation'),('filament',50,1444000,2444000,'stress'),
            ('crossing_step',50,1444300,2444300,'stress')]:
        for li,level in enumerate([3.,6.,10.]):
            for sf in range(2):
                for j in range(n_per_stratum):
                    i=li*2*n_per_stratum+sf*n_per_stratum+j
                    cases.append(dict(id=f'{family}_{i:04d}',split=split,kind='positive',scene_seed=s0+i,
                        noise_seed=n0+i,secondary=level,separation_family=sf,support_family=family))
    return cases


def freeze():
    if P.exists():raise RuntimeError('Existing protocol retained')
    selected=json.loads((ROOT/'results'/'disagreement_development_v1'/'analysis.json').read_text())['selected']
    assert selected=='route35_cal0.005','Inspect development selection before defining final rule'
    normalizers=json.loads((ROOT/'SUPPORT_DEVELOPMENT_PROTOCOL.json').read_text())['normalizers']
    p=dict(status='Frozen before new inputs or accuracy outcomes',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        development_sources=['SUPPORT_DEVELOPMENT_PROTOCOL.json','DISAGREEMENT_DEVELOPMENT_PROTOCOL.json'],
        primary_method='routed at0.005',settings=dict(gate=3.5,normalizers=normalizers,alpha=[.005,.01],
            spatial_sigma=1.5,depth_sigma=2.,localization_tolerance_bins=2.,disagreement_ratio=1.5),cases=designs(),
        endpoints=['Coherent correct-localization gain vs ordinary local fit.',
            'Coherent recall difference vs prior spatial report; prespecified descriptive noninferiority margin -2percentagepoints.',
            'Singleton recall difference vs ordinary local fit; prespecified descriptive noninferiority margin -5percentagepoints.',
            'Report null false alarms and positive wrong locations separately in every group.',
            'Six-stratum paired bootstrap per positive family;1000two-stage calibration resamples for main comparisons.',
            'Two new geometry families are prespecified out-of-distribution stress, not pooled into primary groups.'],
        notes=['Every earlier test cohort is now development for this new routing algorithm; no earlier accuracy is an independent claim for it.',
            'Gate and score normalizers were selected from CLOSED development data and are fixed before this test.',
            'Support proxy is not a confidence bound; the complete routing score needs fresh null calibration.',
            'Calibration3999 and testnull3000 follow the same randomized null-context mixture. No test positive sets thresholds.',
            'Shared fresh calibration for all empirical controls; ranks ceil((m+1)*(1-alpha)),strict score>max(0,threshold).',
            'Main900coherent and600singleton positive families remain separate; balanced utility is descriptive only.',
            'Stress300filament and300crossing-step positives. Crossing-step true query lies on high-depth side; base separation may gain6--12bins.',
            'Strong controls: ordinary local fit, prior spatial report, query-refined guided fit; ablations: naive union, disagreement override, fixed-secondary profile.',
            'Disagreement override was weaker in development; it is an ablation, not a selected primary method.',
            'No new conditional e-value guarantee, physical sensor accuracy, whole-image throughput or FPGA claim.',
            'No outcome-based switching of the primary policy after this test.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen',len(p['cases']),'fresh acquisitions',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    return p


def prepare():
    p=verify()
    if (DATA/'manifest.json').exists():raise RuntimeError('Existing manifest retained')
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);cases=[]
    for i,c in enumerate(p['cases']):
        f=DATA/(c['id']+'.npz')
        if f.exists():raise RuntimeError('Partial preparation exists; inspect before resuming')
        h,m,_=make_routing_case(c);np.savez_compressed(f,histogram=h)
        cases.append(dict(**c,metadata=m,sha256=sha(f)))
        if (i+1)%1000==0:print('Prepared',i+1,flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=cases),indent=2))
    print('Prepared',len(cases),flush=True)


def worker(c,digest,settings):
    f=OUT/(c['id']+'.json')
    if f.exists():
        r=json.loads(f.read_text());assert r['protocol_sha256']==digest;return r
    inp=DATA/(c['id']+'.npz');assert sha(inp)==c['sha256'];h=np.load(inp)['histogram'];timings={};start=time.perf_counter()
    t=time.perf_counter();pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info);timings['proposal']=time.perf_counter()-t
    t=time.perf_counter();spatial=complete_guided_likelihood(h[8,8],q['primary'],q['secondary']);timings['guided_fit']=time.perf_counter()-t
    t=time.perf_counter();local=local_likelihood_report(h[8,8]);timings['local_fit']=time.perf_counter()-t
    reports=dict(guided_fit=spatial,local_fit=local,spatial_report=dict(spatial,delay=q['secondary']))
    reports['routed']=route_report(q,spatial,local,settings['normalizers'],'route',settings['gate'])
    reports['naive_union']=route_report(q,spatial,local,settings['normalizers'],'naive_union')
    reports['disagreement']=disagreement_report(q,spatial,local,settings['normalizers'],ratio=settings['disagreement_ratio'],gate=settings['gate'])
    t=time.perf_counter();reports['fixed_secondary']=profile_at_proposals(h[8,8],q['primary'],q['secondary'],'fixed_secondary');timings['fixed_secondary']=time.perf_counter()-t
    result=dict(id=c['id'],protocol_sha256=digest,proposal=q,reports=reports,timings=timings,total_seconds=time.perf_counter()-start)
    tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(f);return result


def calibrate(cases,settings):
    rows=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases];out={}
    for name in rows[0]['reports']:
        scores=sorted(r['reports'][name]['score'] for r in rows);out[name]={}
        for alpha in settings['alpha']:
            rank=math.ceil((len(scores)+1)*(1-alpha));out[name][str(alpha)]=dict(threshold=max(0.,float(scores[rank-1])),rank=rank,n=len(scores),strict_greater=True)
    assert out['spatial_report']==out['guided_fit'];return out


def run(workers):
    p=verify();digest=sha(P);manifest=json.loads((DATA/'manifest.json').read_text());assert manifest['protocol_sha256']==digest
    start=time.perf_counter()
    for split in ['calibration','evaluation','stress']:
        cases=[c for c in manifest['cases'] if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,digest,p['settings']) for c in cases]
            for i,j in enumerate(as_completed(jobs),1):
                j.result()
                if i%500==0 or i==len(jobs):print(split,i,'/',len(jobs),'elapsed',round(time.perf_counter()-start,1),flush=True)
        if split=='calibration':
            (OUT/'thresholds.json').write_text(json.dumps(dict(protocol_sha256=digest,methods=calibrate(cases,p['settings'])),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=digest,cases=len(manifest['cases']),elapsed_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
