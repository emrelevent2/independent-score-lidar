"""Closed-cohort development of a coverage-preserving proper alternative.

Analytic declarations require a certified sparse-mixture lower numerator and
the SAME complete-cover null upper bound for both alternatives. A numerical
screen only saves computation; it cannot authorize a declaration.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,time
import numpy as np
from flint import arb
from adaptive_evidence import mixture_interval,feasible_local_null
from envelope_evidence import continuous_null_bound,outward_float
from coverage_alternative import coverage_components,numeric_scores,selected_indices,sparse_numerator_lower,location_report
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/routing_confirmation_v1';BASE=ROOT/'results/routing_confirmation_v1'
OUT=ROOT/'results/coverage_development_v1';P=ROOT/'COVERAGE_ALTERNATIVE_DEVELOPMENT_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    assert not P.exists(),'Existing coverage development snapshot retained'
    manifest=json.loads((DATA/'manifest.json').read_text());chosen=[]
    chosen.extend([c for c in manifest['cases'] if c['kind']=='null' and c['split']=='evaluation'][:96])
    for family,per_stratum in [('coherent',8),('singleton',8),('filament',4),('crossing_step',4)]:
        for level in [3.,6.,10.]:
            for separation in [0,1]:
                chosen.extend([c for c in manifest['cases'] if c['kind']=='positive' and c['support_family']==family and c['secondary']==level and c['separation_family']==separation][:per_stratum])
    assert len(chosen)==240
    for c in chosen:c['proposal_record_sha256']=sha(BASE/(c['id']+'.json'))
    names=['coverage_development.py','coverage_alternative.py','adaptive_evidence.py','envelope_evidence.py',
        'continuous_alternative.py','support_routing.py','guided_alternatives.py','likelihood_baseline.py']
    p=dict(status='DEVELOPMENT ONLY on a closed routing cohort. New mixture is not independently confirmed by these observations.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in names},
        source_manifest_sha256=sha(DATA/'manifest.json'),cases=chosen,
        settings=dict(alpha=.01,cycles=1000,fwhm_range=[4.5,5.5],initial_cells=32,max_boxes=256,
            global_top_components=16,secondary_amplitudes=[1.,2.,4.,8.,16.,32.],support_gate=3.5,
            spatial_mass_high='9/10',spatial_mass_low='1/10',numerical_screen_margin=1.),
        endpoints=['Matched256-box certified spatial-mixture versus spatial+full-delay coverage mixture.',
            'Report null declarations,correct secondary location and wrong-positive location separately in all4families.',
            'Both analytic acceptance tests use lower numerator minus the SAME full-cover null upper,thresholdlog100roundedup.',
            'A full numeric mixture score and posterior location are diagnostics/empirical candidates,not analytic e-values.'],
        notes=['Every rate vector,component grid and exact rational prior weight depends only on the saved donor proposal.',
            'Query-adaptive evaluation retains original weights. Its sub-sum is pointwise<=the fixed proper mixture; no renormalization.',
            'Full numeric score screen may discard candidates but never enables an acceptance. False-positive guarantee therefore does not rely on its numerical accuracy.',
            'Global location reports a within2bin posterior mean around the strongest global delay; spatial location remains donor-derived.',
            'This is an implementation of established mixture/universal-inference principles,not a new general statistical theorem.',
            'Original and current real outcomes remain closed; no retrospective independent claim for this later alternative.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen coverage development240cases',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DATA/'manifest.json')==p['source_manifest_sha256'];return p


def worker(c,digest,settings):
    path=OUT/(c['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return
    assert sha(DATA/(c['id']+'.npz'))==c['sha256'] and sha(BASE/(c['id']+'.json'))==c['proposal_record_sha256']
    h=np.load(DATA/(c['id']+'.npz'))['histogram'][8,8];proposal=json.loads((BASE/(c['id']+'.json')).read_text())['proposal'];assert proposal['raw_center_excluded']
    start=time.perf_counter();models=coverage_components(proposal,cycles=settings['cycles'],amplitudes=settings['secondary_amplitudes'])
    _,weighted,full_numeric=numeric_scores(h,models,settings['cycles']);report=location_report(models,weighted)
    old=mixture_interval(h,models['rates'][:4],settings['cycles']);witness=feasible_local_null(h,settings['cycles'],tuple(settings['fwhm_range']))
    threshold=outward_float(arb(100).log(),True);lower_threshold=outward_float(arb(100).log(),False)
    candidate=(old['upper']-witness['lower']>=lower_threshold or full_numeric-witness['lower']>=threshold-settings['numerical_screen_margin'])
    bound=None;sparse=None;old_declared=new_declared=False;old_log=new_log=None
    if candidate:
        sparse=sparse_numerator_lower(h,models,selected_indices(weighted,settings['global_top_components']),settings['cycles'])
        bound=continuous_null_bound(h,settings['cycles'],tuple(settings['fwhm_range']),initial_cells=settings['initial_cells'],
            max_boxes=settings['max_boxes'],normalized_envelope=False)
        old_log=outward_float(arb(old['lower'])-arb(bound['upper']),False)
        new_log=outward_float(arb(sparse['lower'])-arb(bound['upper']),False)
        old_declared=old_log>=threshold;new_declared=new_log>=threshold
    result=dict(id=c['id'],protocol_sha256=digest,proposal=proposal,settings=settings,
        full_components=len(models['weights']),spatial_mass=str(models['spatial_mass']),rate_matrix_sha256=hashlib.sha256(models['rates'].tobytes()).hexdigest(),
        old_mixture_interval=old,local_null_witness=witness,numeric_full_log_mixture=full_numeric,
        old_fast_score=(old['lower']+old['upper'])/2-witness['lower'],coverage_fast_score=full_numeric-witness['lower'],
        screened=not candidate,screen_is_not_acceptance_certificate=True,sparse_numerator=sparse,bound=bound,
        threshold_upper=threshold,old_log_e_lower=old_log,coverage_log_e_lower=new_log,
        old_report=dict(declared=bool(old_declared),delay=proposal['secondary'],primary=proposal['primary']),
        coverage_report=dict(declared=bool(new_declared),primary=proposal['primary'],**report),elapsed_seconds=time.perf_counter()-start)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,allow_nan=False));temp.replace(path)


def run(workers):
    p=verify();OUT.mkdir(exist_ok=True);t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,sha(P),p['settings']) for c in p['cases']]
        for i,job in enumerate(as_completed(jobs),1):
            job.result()
            if i%24==0:print('Coverage development',i,'/240 seconds',round(time.perf_counter()-t,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(p['cases']),elapsed_seconds=time.perf_counter()-t),indent=2))


def analyze():
    p=verify();assert json.loads((OUT/'completion.json').read_text())['cases']==240
    summary={method:{g:dict(n=0,declared=0,correct=0,wrong=0) for g in ['null','coherent','singleton','filament','crossing_step']} for method in ['old_report','coverage_report']}
    diagnostics=dict(screened=0,bounded=0,coverage_accepted=0,global_accepted=0)
    for c in p['cases']:
        r=json.loads((OUT/(c['id']+'.json')).read_text());assert r['protocol_sha256']==sha(P);g='null' if c['kind']=='null' else c['support_family']
        diagnostics['screened']+=r['screened'];diagnostics['bounded']+=r['bound'] is not None
        for method in summary:
            report=r[method];group=summary[method][g];group['n']+=1;group['declared']+=report['declared']
            if c['kind']=='positive' and report['declared']:
                correct=abs(report['delay']-c['metadata']['secondary_delay'])<=2.;group['correct']+=correct;group['wrong']+=not correct
        if r['coverage_report']['declared']:
            assert r['coverage_log_e_lower']>=r['threshold_upper'];assert r['sparse_numerator']['original_weights_retained']
            diagnostics['coverage_accepted']+=1;diagnostics['global_accepted']+=r['coverage_report']['head']=='global'
    result=dict(status='DEVELOPMENT ONLY; no independent efficacy claim.',protocol_sha256=sha(P),methods=summary,diagnostics=diagnostics)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    freeze() if args.action=='freeze' else run(args.workers) if args.action=='run' else analyze()
