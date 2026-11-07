"""Prospective confirmation of development-selected 100k-cycle candidate.

The failed 10k primary remains failed. This NEW protocol transparently selects
the secondary 100k candidate using development and evaluates new random seeds.
No claim of measured-sensor or universal SOTA superiority is made.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time
import numpy as np
from scipy.stats import beta,binomtest
from flint import arb
import shoulder_development as base
import shoulder_calibrated_development as development
from guided_alternatives import guided_pool,fit_pool
from coverage_alternative import coverage_components
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from calibration_confidence_v2 import lobe_confidence
from shoulder_envelope import shoulder_null_bound
from envelope_evidence import continuous_null_bound,outward_float

ROOT=base.ROOT;P=ROOT/'SHOULDER_CONFIRMATORY_PROTOCOL.json'
OUT=ROOT/'results/shoulder_confirmation_v1'
METHODS=['old_gaussian','new_uncalibrated','cal_100000']


def cases():
    templates=base.cases();rows=[]
    for g,group in enumerate(['null_plain','null_lobe','far_two','near_two']):
        group_templates=[c for c in templates if c['group']==group]
        for i in range(192):
            c=dict(group_templates[i%24]);c.update(id=f'{group}_{i:04}',seed=2026112800+100000*g+i,stratum=i%24)
            rows.append(c)
    return rows


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else
                     [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:pending.append(local)
    return {f'src/{name}':base.sha(ROOT/'src'/name) for name in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    d=development.check_protocol();a=json.loads((development.OUT/'analysis.json').read_text())
    assert not a['continue_to_fresh_confirmation']  # Preserve failed 10k primary.
    assert json.loads((development.OUT/'integrity.json').read_text())['status']=='PASS'
    p=dict(status='FROZEN_BEFORE_NEW_SCENE_GENERATION',sources=source_graph(),cases=cases(),
        development_protocol_sha256=base.sha(development.P),development_analysis_sha256=base.sha(development.OUT/'analysis.json'),
        candidate_selection='10k primary failed. 100k secondary on closed development gave37/48correct versus33/48oldGaussian,0wrong versus11,0/48null versus24/48. Selected only for this distinct prospective protocol; no rewrite of development primary.',
        settings=dict(calibration_cycles=100000,query_cycles=1000,total_error_budget='1/100',calibration_failure_budget='1/1000',
            query_error_budget='9/1000',optical_cap=.25,optical_shift=16,fwhm_range=[4.5,5.5],initial_cells=32,max_boxes=256,
            location_tolerance=2,bootstrap_replicates=10000,bootstrap_seed=2026112899),
        primary='Against oldGaussian, strictly more paired correct two-surface reports, fewer physical-null false objects, fewer wrong-location reports; report each endpoint separately.',
        success_rule=dict(all_required=True,paired_correct_gain='lower two-sided stratified percentile95%CI >0',
            false_object_reduction='strict count decrease',wrong_location_reduction='strict count decrease'),
        diagnostic='Uncalibrated optical-shape-aware null: calibration should restore near-return power. All positive/null families and photon/geometry strata remain visible.',
        design='192new independent acquisitions per designed family,8replicates of each frozen24strata;384physical nulls and384two-surface queries. New seeds, same stylized model/domain as development. No best-budget/threshold selection.',
        accounting='Independent single-surface calibration100k cycles per case; its measured detection count reported. Do not equate cycles to photons, or imply equal acquisition cost to oldGaussian.',
        limitations='Gaussian control intentionally misspecified for nonideal optical-lobe scenes; improvement over it is a response-model stress test, not general superiority over calibrated IRF-aware methods. No correlated afterpulsing, drift, real-data efficacy, FPGA timing, imagewise/FDR or Q1 claim.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.sha(P),len(p['sources']),'sources',len(p['cases']),'cases',flush=True)


def check_protocol():
    p=json.loads(P.read_text());development.check_protocol()
    assert p['cases']==cases()
    for path,digest in p['sources'].items():assert base.sha(ROOT/path)==digest,path
    assert base.sha(development.OUT/'analysis.json')==p['development_analysis_sha256']
    return p


def worker(c):
    start=time.perf_counter();h,cal,truth=base.simulate_case(c);query=h[4,4]
    cf=lobe_confidence(cal[2],100000);cap=cf['eta_upper']
    pool,meta=guided_pool(h,4,4);old=fit_pool(pool);old.update(meta)
    uncal=donor_proposal(h);calibrated=donor_proposal(h,cap=cap)
    om=coverage_components(old);om['spatial']=np.arange(len(om['rates']))<4
    numerators=dict(old_gaussian=numeric_and_sparse(query,om),new_uncalibrated=numeric_and_sparse(query,components(uncal)),
                    cal_100000=numeric_and_sparse(query,components(calibrated,cap=cap)))
    cut=outward_float(arb(100).log());ccut=outward_float((arb(1000)/9).log())
    bounds=dict(old_gaussian=continuous_null_bound(query,fwhm_range=(4.5,5.5),initial_cells=32,max_boxes=256,normalized_envelope=False))
    for method,cap_value,threshold in [('new_uncalibrated',.25,cut),('cal_100000',cap,ccut)]:
        level=outward_float(arb(numerators[method]['lower'])-arb(threshold),False)
        bounds[method]=shoulder_null_bound(query,cap=cap_value,decision_level=level)
    proposals=dict(old_gaussian=old,new_uncalibrated=uncal,cal_100000=calibrated);reports={}
    for method in METHODS:
        num=numerators[method];b=bounds[method];threshold=ccut if method=='cal_100000' else cut
        score=outward_float(arb(num['lower'])-arb(b['upper']),False)
        reports[method]=dict(declared=score>=threshold,log_e_lower=score,log_threshold_upper=threshold,
            primary=proposals[method]['primary'],secondary=num['location'],head=num['head'],
            numerator_lower=num['lower'],null_upper=b['upper'])
    return dict(id=c['id'],case=c,truth=truth,query=query.tolist(),
        raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),calibration_histogram=cal[2].tolist(),
        confidence=cf,proposals=proposals,numerators=numerators,bounds=bounds,reports=reports,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def run(workers):
    p=check_protocol();start=time.perf_counter();db=connect();ids={r[0] for r in db.execute('SELECT id FROM records')}
    pending=[c for c in p['cases'] if c['id'] not in ids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(worker,c) for c in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%48==0:print('Confirmed records',len(ids),'of',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),invocation_seconds=time.perf_counter()-start,
        protocol_sha256=base.sha(P),sqlite_sha256=base.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check_protocol();db=connect();rows=[json.loads(r[0]) for r in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert len(rows)==len(p['cases']);summary={}
    for method in METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            ep=[base.endpoint(r,method) for r in rows if r['case']['group']==group]
            groups[group]={k:sum(e[k] for e in ep) for k in ep[0]}
        summary[method]=groups
    positives=[r for r in rows if r['case']['secondary']>0]
    differences=np.array([base.endpoint(r,'cal_100000')['correct']-base.endpoint(r,'old_gaussian')['correct'] for r in positives])
    rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates']
    bootstrap=np.zeros(rep)
    for group in ['far_two','near_two']:
        for stratum in range(24):
            d=np.array([base.endpoint(r,'cal_100000')['correct']-base.endpoint(r,'old_gaussian')['correct'] for r in positives if r['case']['group']==group and r['case']['stratum']==stratum])
            assert len(d)==8
            bootstrap+=rng.choice(d,size=(rep,len(d)),replace=True).sum(axis=1)/len(positives)
    ci=np.quantile(bootstrap,[.025,.975]).tolist()
    gains=int(np.sum(differences>0));losses=int(np.sum(differences<0))
    correct=lambda m:sum(summary[m][g]['correct'] for g in ['near_two','far_two'])
    false=lambda m:sum(summary[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])
    wrong=lambda m:sum(summary[m][g]['wrong_positive_location'] for g in ['near_two','far_two'])
    success=dict(paired_correct_gain=ci[0]>0,false_object_reduction=false('cal_100000')<false('old_gaussian'),
                 wrong_location_reduction=wrong('cal_100000')<wrong('old_gaussian'))
    result=dict(status='CONFIRMATION_COMPLETE',methods=summary,success=success,all_prespecified_success_criteria=all(success.values()),
        paired_correct=dict(gains=gains,losses=losses,rate_difference=float(differences.mean()),stratified_percentile_95_ci=ci,
            exact_one_sided_mcnemar_p=float(binomtest(gains,gains+losses,.5,alternative='greater').pvalue) if gains+losses else 1.),
        totals={m:dict(correct=correct(m),wrong_location=wrong(m),false_objects=false(m)) for m in METHODS},
        calibration=dict(detected_photons_median=float(np.median([r['confidence']['detected'] for r in rows])),
            informative=sum(r['confidence']['informative'] for r in rows),realized_undercoverage=sum(r['confidence']['eta_upper']<r['case']['eta'] for r in rows)),
        null_empirical_95_one_sided_upper=float(beta.ppf(.95,false('cal_100000')+1,384-false('cal_100000'))) if false('cal_100000')<384 else 1.,
        limitations=p['limitations'],protocol_sha256=base.sha(P))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers)
    else:analyze()
