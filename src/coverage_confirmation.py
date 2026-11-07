"""Fresh confirmation of proper full-delay coverage at matched null budgets."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,math,time
import numpy as np
from routing_scenes import make_routing_case
from guided_alternatives import guided_pool,fit_pool
from coverage_evidence import evaluate_coverage
from likelihood_baseline import local_likelihood_report
from complete_guided_likelihood import complete_guided_likelihood
from support_routing import route_report
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/coverage_confirmation_v1';OUT=ROOT/'results/coverage_confirmation_v1';P=ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['coverage_confirmation.py','coverage_confirmation_analysis.py','coverage_evidence.py','coverage_alternative.py',
    'routing_scenes.py','guided_confirmatory_scenes.py','guided_alternatives.py','continuous_alternative.py',
    'adaptive_evidence.py','envelope_evidence.py','likelihood_evidence.py','likelihood_baseline.py','complete_guided_likelihood.py',
    'support_routing.py','pilot.py','crossfit.py','screen.py','methods.py','joint_detector.py','analyze_routing_confirmation.py',
    'routing_confirmation.py','profile_proposals.py','disagreement_routing.py']
EMPIRICAL=['spatial_fast','coverage_fast','local_fit','guided_fit','previous_routed']


def designs():
    cases=[]
    for split,n,start in [('calibration',1999,1930000),('evaluation',2000,1940000)]:
        for i in range(n):cases.append(dict(id=f'{"cal" if split=="calibration" else "null"}_{i:04d}',split=split,kind='null',scene_seed=start+i,
            noise_seed=start+i+1000000,secondary=0.,separation_family=int(np.random.default_rng(start+i+10000000).integers(0,2)),support_family='coherent'))
    for family,n,start in [('coherent',150,1950000),('singleton',100,1952000),('filament',50,1954000),('crossing_step',50,1955000)]:
        for li,level in enumerate([3.,6.,10.]):
            for sf in range(2):
                for j in range(n):
                    i=li*2*n+sf*n+j;cases.append(dict(id=f'{family}_{i:04d}',split='evaluation',kind='positive',scene_seed=start+i,noise_seed=start+i+1000000,
                        secondary=level,separation_family=sf,support_family=family))
    return cases


def freeze():
    assert not P.exists() and not DATA.exists(),'Existing prospective protocol/inputs retained'
    dev=ROOT/'results/coverage_development_v1/analysis.json';a=json.loads(dev.read_text());assert a['status'].startswith('DEVELOPMENT')
    cases=designs();assert len(cases)==6099
    for name in ['CONFIRMATORY_FREEZE.json','GUIDED_CONFIRMATORY_PROTOCOL.json','LOCALIZATION_CONFIRMATORY_PROTOCOL.json','ROUTING_CONFIRMATORY_PROTOCOL.json','GUARDED_LOCALIZATION_CONFIRMATORY_PROTOCOL.json']:
        old=json.loads((ROOT/name).read_text()).get('cases',[])
        if old:
            assert not ({c['scene_seed'] for c in cases}&{c['scene_seed'] for c in old})
            assert not ({c['noise_seed'] for c in cases}&{c['noise_seed'] for c in old})
    p=dict(status='FROZEN before6099new observations or any current accuracy outcome.',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},development_analysis_sha256=sha(dev),cases=cases,
        settings=dict(alpha=.01,cycles=1000,fwhm_range=[4.5,5.5],initial_cells=32,max_boxes=256,global_top_components=16,
            secondary_amplitudes=[1.,2.,4.,8.,16.,32.],support_gate=3.5,spatial_mass_high='9/10',spatial_mass_low='1/10',numerical_screen_margin=1.),
        empirical_alphas=[.005,.01],primary_empirical_alpha=.005,
        normalizers=json.loads((ROOT/'SUPPORT_DEVELOPMENT_PROTOCOL.json').read_text())['normalizers'],
        endpoints=['Primary analytic comparison at1%:isolated correct-secondary gain of coverage versus spatial-only proper mixture.',
            'Coherent correct-rate cost reported separately; prespecified descriptive noninferiority margin minus3percentagepoints.',
            'Separate null declarations,wrong-positive coordinates,role-invariant any-channel and one-to-one pair endpoints in everyfamily.',
            'Both analytic alternatives use exactlythe same256-box complete-cover null bound on each bounded query.',
            'Secondary calibrated full-mixture scores versus ordinarylocal,strongguided and previousrouter atsharedfreshnull calibration.',
            '5000six-stratum paired acquisition bootstrap;1000two-stage calibration+test resamples for empirical comparisons.'],
        notes=['No weight/threshold/grid search after the240case development result. All defaults transferred unchanged.',
            'Global prior grid andweights dependonlyon center-excluded donors; query selects evaluation subset,not the underlying distribution.',
            'Original exact rational weights are retained after sparse evaluation; no selected-component renormalization.',
            'All analytic acceptances require outward lower numerator minus complete-null upper >=outward upperlog100.',
            'Numerical candidate screen can remove declarations only; no assumption on its numerical accuracy enters analytic typeI validity.',
            'Fast scores subtract only a feasible null witness and have NOanalytic e-value guarantee.',
            'All4positive families already appear in development. No untrained-geometry or new-physical-sensor claim.',
            'Old real and synthetic cohorts are closed development for this later coverage method; no retrospective independent claim.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen proper-coverage confirmation6099cases',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(ROOT/'results/coverage_development_v1/analysis.json')==p['development_analysis_sha256'];return p


def prepare():
    p=verify();DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);assert not (DATA/'manifest.json').exists();rows=[]
    for i,c in enumerate(p['cases']):
        path=DATA/(c['id']+'.npz');assert not path.exists(),'Inspect partial preparation before any overwrite'
        h,m,_=make_routing_case(c);np.savez_compressed(path,histogram=h);rows.append(dict(**c,metadata=m,sha256=sha(path)))
        if (i+1)%1000==0:print('Prepared coverage',i+1,flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=rows),indent=2))


def worker(c,digest,settings,norms):
    path=OUT/(c['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    assert sha(DATA/(c['id']+'.npz'))==c['sha256'];h=np.load(DATA/(c['id']+'.npz'))['histogram'];start=time.perf_counter()
    pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info);evidence=evaluate_coverage(h[8,8],q,settings)
    local=local_likelihood_report(h[8,8]);guided=complete_guided_likelihood(h[8,8],q['primary'],q['secondary'])
    spatial=dict(score=evidence['old_fast_score'],primary=q['primary'],delay=q['secondary'],head='spatial')
    coverage=dict(score=evidence['coverage_fast_score'],primary=q['primary'],delay=evidence['coverage_report']['delay'],head=evidence['coverage_report']['head'])
    reports=dict(spatial_cert=evidence['old_report'],coverage_cert=evidence['coverage_report'],spatial_fast=spatial,coverage_fast=coverage,
        local_fit=local,guided_fit=guided,previous_routed=route_report(q,guided,local,norms,'route',3.5))
    r=dict(id=c['id'],protocol_sha256=digest,proposal=q,evidence=evidence,reports=reports,elapsed_seconds=time.perf_counter()-start)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(r,allow_nan=False));temp.replace(path)


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P);start=time.perf_counter()
    for split in ['calibration','evaluation']:
        cs=[c for c in m['cases'] if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,sha(P),p['settings'],p['normalizers']) for c in cs]
            for i,job in enumerate(as_completed(jobs),1):
                job.result()
                if i%500==0 or i==len(jobs):print('Coverage confirmation',split,i,'/',len(jobs),'seconds',round(time.perf_counter()-start,1),flush=True)
        if split=='calibration':
            records=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in cs];cuts={}
            for method in EMPIRICAL:
                values=sorted(r['reports'][method]['score'] for r in records);cuts[method]={}
                for alpha in p['empirical_alphas']:cuts[method][str(alpha)]=max(0.,values[math.ceil((len(values)+1)*(1-alpha))-1])
            (OUT/'thresholds.json').write_text(json.dumps(dict(protocol_sha256=sha(P),methods=cuts),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(m['cases']),elapsed_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
