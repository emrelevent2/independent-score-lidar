"""Fresh simulation confirmation of unchanged decisions + guarded locations."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,math,time
import numpy as np
from routing_scenes import make_routing_case
from guided_alternatives import guided_pool,fit_pool
from complete_guided_likelihood import complete_guided_likelihood
from likelihood_baseline import local_likelihood_report
from support_routing import route_report
from guarded_localization import guarded_report
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/guarded_confirmation_v1';OUT=ROOT/'results/guarded_confirmation_v1'
P=ROOT/'GUARDED_LOCALIZATION_CONFIRMATORY_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['guarded_confirmation.py','guarded_confirmation_analysis.py','guarded_localization.py','routing_scenes.py','guided_confirmatory_scenes.py',
    'guided_alternatives.py','complete_guided_likelihood.py','likelihood_baseline.py','support_routing.py','adaptive_evidence.py',
    'envelope_evidence.py','continuous_alternative.py','likelihood_evidence.py','pilot.py','crossfit.py','screen.py','methods.py','joint_detector.py',
    'analyze_routing_confirmation.py','routing_confirmation.py','profile_proposals.py','disagreement_routing.py']


def designs():
    cases=[]
    for split,n,s0 in [('calibration',1999,1730000),('evaluation',2000,1740000)]:
        for i in range(n):
            cases.append(dict(id=f'{"cal" if split=="calibration" else "null"}_{i:04d}',split=split,kind='null',scene_seed=s0+i,
                noise_seed=s0+i+1000000,secondary=0.,separation_family=int(np.random.default_rng(s0+i+10000000).integers(0,2)),support_family='coherent'))
    for family,n,s0 in [('coherent',150,1750000),('singleton',100,1752000),('filament',50,1754000),('crossing_step',50,1755000)]:
        for li,level in enumerate([3.,6.,10.]):
            for sf in range(2):
                for j in range(n):
                    i=li*2*n+sf*n+j
                    cases.append(dict(id=f'{family}_{i:04d}',split='evaluation',kind='positive',scene_seed=s0+i,noise_seed=s0+i+1000000,
                        secondary=level,separation_family=sf,support_family=family))
    return cases


def freeze():
    if P.exists():raise RuntimeError('Existing confirmation snapshot retained')
    a=json.loads((ROOT/'results/guarded_development_v1/analysis.json').read_text());assert a['selected']['id']!='local_fallback'
    cases=designs();old=json.loads((ROOT/'ROUTING_CONFIRMATORY_PROTOCOL.json').read_text())['cases']
    assert not ({c['scene_seed'] for c in cases}&{c['scene_seed'] for c in old})
    assert not ({c['noise_seed'] for c in cases}&{c['noise_seed'] for c in old})
    p=dict(status='Fresh6099acquisitions; frozen BEFORE inputs/outcomes. Previous9099routing cohort is development for this rule.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        development_analysis_sha256=sha(ROOT/'results/guarded_development_v1/analysis.json'),settings=a['selected']['setting'],
        normalizers=json.loads((ROOT/'SUPPORT_DEVELOPMENT_PROTOCOL.json').read_text())['normalizers'],alphas=[.005,.01],primary_alpha=.005,cases=cases,
        endpoints=['Primary: paired correct-secondary localization gain over ordinary local likelihood in900coherent queries.',
            'Wrong location, pair localization and any-channel truth agreement remain separate.',
            'Isolated600,filament300,crossing-step300: report individually. These geometries were used in development and are no longer untrained stress.',
            'Null2000: exact equality of declarations to local method, by unchanged score and threshold; not a lower false-alarm-rate claim.',
            'Six-stratum paired bootstrap,5000replicates; two-stage calibration+test bootstrap1000replicates for all groups.'],
        notes=['Local score and threshold are identical by construction; only returned secondary position changes.',
            'Support>=3.5,abs(local-donor)<=4bins,guided likelihood improvement>=0.5local,position is mean of local and donor.',
            'Pure local and previous high-sensitivity routed controls use same inputs and fresh null calibration.',
            'No model-conditional e-value or physical sensor generalization is asserted for this empirically calibrated rule.',
            'This simulation rule is not silently applied to the separate frozen real working-count study.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen guarded confirmation',len(cases),'inputs',p['settings'],flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(ROOT/'results/guarded_development_v1/analysis.json')==p['development_analysis_sha256']
    return p


def prepare():
    p=verify();DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);entries=[]
    assert not (DATA/'manifest.json').exists(),'Completed inputs retained'
    for i,c in enumerate(p['cases']):
        path=DATA/(c['id']+'.npz');assert not path.exists(),'Inspect partial preparation before overwriting'
        h,m,_=make_routing_case(c);np.savez_compressed(path,histogram=h);entries.append(dict(**c,metadata=m,sha256=sha(path)))
        if (i+1)%1000==0:print('Prepared guarded',i+1,flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=entries),indent=2))


def worker(c,digest,settings,norms):
    path=OUT/(c['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    assert sha(DATA/(c['id']+'.npz'))==c['sha256'];h=np.load(DATA/(c['id']+'.npz'))['histogram'];start=time.perf_counter()
    pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info)
    spatial=complete_guided_likelihood(h[8,8],q['primary'],q['secondary']);local=local_likelihood_report(h[8,8])
    guarded=guarded_report(q,spatial,local,settings);assert guarded['score']==local['score']
    reports=dict(local=local,guarded=guarded,previous_routed=route_report(q,spatial,local,norms,'route',3.5),
        guided=spatial,spatial=dict(spatial,delay=q['secondary']))
    r=dict(id=c['id'],protocol_sha256=digest,proposal=q,reports=reports,elapsed_seconds=time.perf_counter()-start)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,allow_nan=False));tmp.replace(path)


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P);t=time.perf_counter()
    for split in ['calibration','evaluation']:
        cases=[c for c in m['cases'] if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,sha(P),p['settings'],p['normalizers']) for c in cases]
            for i,job in enumerate(as_completed(jobs),1):
                job.result()
                if i%500==0 or i==len(jobs):print('Guarded',split,i,'/',len(jobs),'seconds',round(time.perf_counter()-t,1),flush=True)
        if split=='calibration':
            records=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases];cuts={}
            for method in records[0]['reports']:
                scores=sorted(r['reports'][method]['score'] for r in records);cuts[method]={}
                for alpha in p['alphas']:
                    rank=math.ceil((len(scores)+1)*(1-alpha));cuts[method][str(alpha)]=max(0.,scores[rank-1])
            assert cuts['guarded']==cuts['local'];(OUT/'thresholds.json').write_text(json.dumps(dict(protocol_sha256=sha(P),methods=cuts),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(m['cases']),elapsed_seconds=time.perf_counter()-t),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
