"""Fresh confirmation of decoupled existence testing and donor localization.

The first guided study is now development for this structural change. Freeze
this protocol before generating any new inputs. Never overwrite prior studies.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, datetime, hashlib, json, math, time
import numpy as np
from guided_confirmatory_scenes import make_case
from pilot import first_detection_probabilities
from guided_alternatives import guided_pool, fit_pool
from continuous_alternative import amplitude_components
from adaptive_evidence import adaptive_evidence, mixture_interval, feasible_local_null
from complete_guided_likelihood import complete_guided_likelihood
from likelihood_baseline import local_likelihood_report
from profile_proposals import profile_at_proposals

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'localization_confirmation_v1'
OUT = ROOT/'results'/'localization_confirmation_v1'
P = ROOT/'LOCALIZATION_CONFIRMATORY_PROTOCOL.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES = ['localization_confirmation.py', 'guided_confirmatory_scenes.py',
    'guided_alternatives.py', 'continuous_alternative.py', 'adaptive_evidence.py',
    'complete_guided_likelihood.py', 'likelihood_baseline.py', 'profile_proposals.py',
    'envelope_evidence.py', 'likelihood_evidence.py', 'pilot.py', 'methods.py',
    'crossfit.py', 'screen.py', 'joint_detector.py']


def designs():
    cases = []
    for split, n, s0, n0 in [('calibration',1999,1130000,2130000),
                            ('evaluation',2000,1132000,2132000)]:
        for i in range(n):
            family = int(np.random.default_rng(s0+i+10000000).integers(0,2))
            cases.append(dict(id=f'{"cal" if split=="calibration" else "null"}_{i:04d}',
                split=split,kind='null',scene_seed=s0+i,noise_seed=n0+i,
                secondary=6.,separation_family=family))
    for li, level in enumerate([3.,6.,10.]):
        for family in range(2):
            for j in range(150):
                i=li*300+family*150+j
                cases.append(dict(id=f'positive_{i:04d}',split='evaluation',kind='positive',
                    scene_seed=1134000+i,noise_seed=2134000+i,
                    secondary=level,separation_family=family))
    for li, level in enumerate([3.,6.,10.]):
        for j in range(100):
            i=li*100+j
            cases.append(dict(id=f'singleton_{i:04d}',split='stress',kind='positive',
                scene_seed=1136000+i,noise_seed=2136000+i,secondary=level,
                separation_family=j%2,singleton=True))
    return cases


def freeze():
    if P.exists(): raise RuntimeError('Existing protocol retained')
    p=dict(status='Frozen before fresh inputs or outcomes',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        primary_method='spatial_report at alpha0.005',
        primary_comparator='complete_guided at alpha0.005',
        settings=dict(alpha=[.005,.01],spatial_sigma=1.5,depth_sigma=2.,
            localization_tolerance_bins=2.,initial_cells=32,max_boxes=2048),
        development_sources=['PROFILE_DEVELOPMENT_PROTOCOL.json','GUIDED_CONFIRMATORY_PROTOCOL.json'],
        cases=designs(),notes=[
            'Primary method retains the complete guided likelihood existence score exactly, but reports the donor secondary position.',
            'Both primary arms have exactly the same calibrated decisions and null false alarms; only localization differs.',
            'Secondary structural arms: fixed_secondary and fixed_both profiles. No outcome-based variant selection.',
            'Other controls: ordinary local likelihood and previously frozen guided_fast/certified.',
            'Fresh1999 null calibrations,2000 independent null tests,900 positives in six fixed photon/separation strata.',
            'Order-statistic calibration: strict score>max(0,score[ceil((m+1)*(1-alpha))]); calibrated modes require exchangeability.',
            'Every proposal excludes raw queried pixel before Coates, weighting, fitting, or location selection.',
            'Positive endpoints: correct weak return within2bins, wrong-depth declarations, any-channel localization and one-to-one two-surface localization.',
            'Report null false declarations separately from wrong positive localization; total errors is their sum.',
            'Prespecified300 singleton positives are an out-of-support stress test, not pooled into main900; use the same calibration thresholds.',
            'Singleton donor-only localization is expected to be weak. No general claim for isolated targets is authorized.',
            'Bootstrap comparisons pair acquisitions and stratify main positives by the six fixed strata; report uncertainty.',
            'No new CASPI/deep-model accuracy claims follow from this focused replication; prior closed comparisons remain separately labeled.',
            'No physical-measurement or FPGA performance claim.'])
    P.write_text(json.dumps(p,indent=2)); print('Frozen',len(p['cases']),'fresh cases',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items(): assert sha(ROOT/'src'/n)==d,n
    return p


def observation(c):
    limits=(12.,24.) if c['separation_family']==0 else (48.,80.)
    h,m,truth=make_case(c['scene_seed'],c['noise_seed'],c['kind'],c['secondary'],separation_range=limits)
    if c.get('singleton'):
        t=np.arange(256);w=truth['fwhm']
        def pulse(d):
            g=np.exp(-4*np.log(2)*((t-d[...,None])/w[...,None])**2)
            return g/g.sum(-1,keepdims=True)
        mask=np.zeros(h.shape[:2],bool);mask[8,8]=True
        rate=(truth['primary_incident'][...,None]*pulse(truth['primary_delay'])+
            (truth['secondary_incident']*mask)[...,None]*pulse(truth['secondary_delay'])+
            truth['background'][...,None]/256)/1000
        rng=np.random.default_rng(c['noise_seed'])
        for r in range(16):
            for k in range(16):h[r,k]=rng.multinomial(1000,first_detection_probabilities(rate[r,k]))[:-1]
        m['context']='isolated_singleton';m['singleton']=True
    return h,m


def prepare():
    p=verify()
    if (DATA/'manifest.json').exists(): raise RuntimeError('Existing inputs retained')
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);records=[]
    for i,c in enumerate(p['cases']):
        f=DATA/(c['id']+'.npz')
        if f.exists(): raise RuntimeError('Partial input exists; inspect before resuming')
        h,m=observation(c);np.savez_compressed(f,histogram=h)
        records.append(dict(**c,metadata=m,sha256=sha(f)))
        if (i+1)%1000==0: print('Prepared',i+1,flush=True)
    (DATA/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=records),indent=2))
    print('Prepared',len(records),'inputs',flush=True)


def worker(c,digest,settings):
    f=OUT/(c['id']+'.json')
    if f.exists():
        r=json.loads(f.read_text());assert r['protocol_sha256']==digest;return r
    inp=DATA/(c['id']+'.npz');assert sha(inp)==c['sha256']
    h=np.load(inp)['histogram'];reports={};timings={};start=time.perf_counter()
    t=time.perf_counter();pool,info=guided_pool(h,8,8,spatial_sigma=1.5,depth_sigma=2.)
    q=fit_pool(pool);timings['proposal']=time.perf_counter()-t
    t=time.perf_counter();null=feasible_local_null(h[8,8]);num=mixture_interval(h[8,8],amplitude_components(q))
    reports['guided_fast']=dict(score=(num['lower']+num['upper'])/2-null['lower'],delay=q['secondary'],primary=q['primary'])
    timings['guided_fast']=time.perf_counter()-t
    t=time.perf_counter();control=complete_guided_likelihood(h[8,8],q['primary'],q['secondary'])
    reports['complete_guided']=control
    reports['spatial_report']=dict(control,delay=q['secondary'],location_source='center-excluded donors')
    timings['complete_guided_and_spatial_report']=time.perf_counter()-t
    for mode in ['fixed_secondary','fixed_both']:
        t=time.perf_counter();reports[mode]=profile_at_proposals(h[8,8],q['primary'],q['secondary'],mode,null_witness=null)
        timings[mode]=time.perf_counter()-t
    t=time.perf_counter();reports['local_likelihood']=local_likelihood_report(h[8,8]);timings['local_likelihood']=time.perf_counter()-t
    if c['split']!='calibration':
        t=time.perf_counter();e=adaptive_evidence(h[8,8],amplitude_components(q),alpha=.01,
            initial_cells=settings['initial_cells'],max_boxes=settings['max_boxes'])
        reports['guided_certified']=dict(declared=bool(e['declared'] and q['valid']),delay=q['secondary'],primary=q['primary'],evidence=e)
        timings['guided_certified']=time.perf_counter()-t
    result=dict(id=c['id'],protocol_sha256=digest,proposal=dict(**q,**info),reports=reports,
        timings=timings,total_seconds=time.perf_counter()-start)
    tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(f)
    return result


def calibrate(cases,settings):
    rows=[json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases];out={}
    for name in rows[0]['reports']:
        scores=sorted(r['reports'][name]['score'] for r in rows);out[name]={}
        for alpha in settings['alpha']:
            rank=math.ceil((len(scores)+1)*(1-alpha));out[name][str(alpha)]=dict(
                threshold=max(0.,float(scores[rank-1])),rank=rank,n=len(scores),strict_greater=True)
    assert out['complete_guided']==out['spatial_report']
    return out


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P)
    start=time.perf_counter()
    for split in ['calibration','evaluation','stress']:
        cases=[c for c in m['cases'] if c['split']==split]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,c,sha(P),p['settings']) for c in cases]
            for i,j in enumerate(as_completed(jobs),1):
                j.result()
                if i%500==0 or i==len(jobs): print(split,i,'/',len(jobs),'elapsed',round(time.perf_counter()-start,1),flush=True)
        if split=='calibration':
            (OUT/'thresholds.json').write_text(json.dumps(dict(protocol_sha256=sha(P),methods=calibrate(cases,p['settings'])),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(m['cases']),elapsed_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run']);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    freeze() if args.action=='freeze' else prepare() if args.action=='prepare' else run(args.workers)
