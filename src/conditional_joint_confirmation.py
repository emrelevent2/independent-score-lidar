"""Prospective comparison: coupled ratio, background-aware interval, point IRF."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time
import numpy as np
from flint import arb
import conditional_joint_diagnostic as development
import background_lobe_development as bgdev
from background_lobe_confidence import lobe_confidence
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from shoulder_envelope import shoulder_null_bound
from fixed_irf_control import estimated_eta,fixed_proposal,fixed_components,fixed_null_bound
from certified_ratio_screen import score_pair,sparse_numerator_upper,single_irf_witness
from joint_calibration_model import split_cycles,fit_training
from envelope_evidence import outward_float

ROOT=development.ROOT;P=ROOT/'CONDITIONAL_JOINT_CONFIRMATORY_PROTOCOL.json'
OUT=ROOT/'results/conditional_joint_confirmation_v1'
METHODS=['background_ci','point','joint']


def cases():
    templates=bgdev.original.cases();rows=[]
    for g,group in enumerate(['null_plain','null_lobe','far_two','near_two']):
        t=[c for c in templates if c['group']==group]
        for i in range(192):
            c=dict(t[i%24]);c.update(id=f'{group}_{i:04}',seed=3036120200+100000*g+i,stratum=i%24)
            rows.append(c)
    return rows


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:pending.append(local)
    return {f'src/{name}':bgdev.original.sha(ROOT/'src'/name) for name in sorted(seen)}


def single_candidate(query,q,model,threshold,cap,kind,compute_bound=True):
    num=numeric_and_sparse(query,model);upper=sparse_numerator_upper(query,model,num['indices'])
    eta=q['eta_donor'];witness=single_irf_witness(query,q,eta)
    ratio_upper=outward_float(arb(upper)-arb(witness['lower']))
    cut_low=outward_float(threshold,False);cut=outward_float(threshold)
    result=dict(proposal=q,numerator=num,numerator_upper=upper,null_witness=witness,
        certified_screened_out=ratio_upper<cut_low,cap_or_eta=cap,kind=kind)
    if ratio_upper<cut_low:
        result.update(bound=None,report=dict(declared=False,primary=None,secondary=None,head='certified_screen',
            log_ratio_lower=None,log_ratio_upper=ratio_upper,log_threshold_upper=cut))
    elif compute_bound:
        level=outward_float(arb(num['lower'])-arb(cut),False)
        bound=(shoulder_null_bound(query,cap=cap,decision_level=level) if kind=='confidence_interval'
               else fixed_null_bound(query,cap,decision_level=level))
        lower=outward_float(arb(num['lower'])-arb(bound['upper']),False)
        result.update(bound=bound,report=dict(declared=lower>=cut,primary=q['primary'],secondary=num['location'],head=num['head'],
            log_ratio_lower=lower,log_ratio_upper=ratio_upper,log_threshold_upper=cut))
    return result


def verify_controls():
    # Closed96 records only; prospective cases are not generated in this action.
    bgdev.check_protocol();rows=bgdev.read_records(bgdev.OUT);screened=0;checked=0
    for row in rows:
        h,cal,_=bgdev.original.simulate_case(row['case']);query=h[4,4]
        cf=lobe_confidence(cal[2],100000);q=donor_proposal(h,cap=cf['eta_upper'])
        eta=estimated_eta(cal[2]);pq=fixed_proposal(h,eta)
        for key,proposal,model,threshold,cap,kind in [
            ('background_ci_100000',q,components(q,cap=cf['eta_upper']),(arb(1000)/9).log(),cf['eta_upper'],'confidence_interval'),
            ('point_100000',pq,fixed_components(pq,eta),arb(100).log(),eta,'point')]:
            result=single_candidate(query,proposal,model,threshold,cap,kind,False)
            assert result['proposal']==row['proposals'][key] and result['numerator']==row['numerators'][key]
            assert result['null_witness']['lower']<=row['bounds'][key]['upper']
            assert result['numerator_upper']>=row['numerators'][key]['lower']
            assert result.get('report',{}).get('log_ratio_upper',float('inf'))>=row['reports'][key]['log_e_lower']
            if result['certified_screened_out']:assert not row['reports'][key]['declared'];screened+=1
            checked+=1
    marker=dict(status='PASS',closed_control_records_checked=checked,screened=screened,
        scope='Same query/null/sparse/screen kernels checked on closed background-CI and legacy-LS-point fixtures; prospective point uses full-calibration MLE,also used by joint training.',
        runner_source_sha256=bgdev.original.sha(Path(__file__)),screen_source_sha256=bgdev.original.sha(ROOT/'src/certified_ratio_screen.py'))
    (ROOT/'output/conditional_control_screen_validation.json').write_text(json.dumps(marker,indent=2));print(marker,flush=True)


def freeze():
    if P.exists():raise FileExistsError('Immutable prospective protocol')
    development.check_protocol()
    assert json.loads((development.OUT/'integrity.json').read_text())['status']=='PASS'
    screen=json.loads((development.OUT/'screen_validation.json').read_text());assert screen['status']=='PASS'
    verify=json.loads((ROOT/'output/conditional_control_screen_validation.json').read_text())
    assert verify['status']=='PASS' and verify['runner_source_sha256']==bgdev.original.sha(Path(__file__))
    assert verify['screen_source_sha256']==bgdev.original.sha(ROOT/'src/certified_ratio_screen.py')
    new=cases();old_seeds={c['seed'] for c in development.parent.cases()}|{c['seed'] for c in bgdev.original.cases()}
    assert not old_seeds&{c['seed'] for c in new}
    p=dict(status='FROZEN_BEFORE_NEW_DATA_GENERATION',sources=source_graph(),cases=new,
        pilot_protocol_sha256=bgdev.original.sha(development.method.P),diagnostic_protocol_sha256=bgdev.original.sha(development.P),
        selection_history='24casepilot failed strict-more-correct. Subsequent outcome-selected CLOSED768 diagnostic recovered24/49misses,kept24/24oldcorrect,0/24null alarms. That diagnostic is not confirmation; this is a distinct prospective comparison on new seeds.',
        settings=dict(calibration_cycles=100000,query_cycles=1000,split_seed_base=9136120200,
            joint_initial_eta=8,joint_initial_delay=16,joint_boxes_per_fold=1024,control_boxes=256,
            total_error_bound='1/100',joint_training_failure_each='1/2000',ratio_markov_budget='9/1000',
            background_interval_failure='1/1000',location_tolerance=2,bootstrap_replicates=10000,bootstrap_seed=9136120999),
        primary='Coupled versus background-aware five-endpoint interval: paired correct-report gain; empirical errors retained separately.',
        success_rule=dict(all_required=True,correct_gain='lower two-sided stratified-percentile95%paired CI>0',
            no_more_wrong='wrong positive locations <=background_ci',no_more_false='physical-null declarations <=background_ci'),
        controls='Same100k calibration for point-IRF control (assumes estimate exact) and confidence interval. Point and joint use the SAME first-event MLE and trusted calibration phase/width band; point fits all100k while each joint fold trains on50k. Legacy diagnostic point used LS. Query/donors identical. Computational budgets differ and will be reported. Point control is not assigned finite-calibration physical-null coverage.',
        screening='Certified rejection only on fixed selected sparse numerator upper and feasible null lower; joint confidence-box witness. All accepted decisions retain full-cover certification. No selected-weight renormalization.',
        probability_scope='Joint statistic has confidence-padded per-query bound.01 marginal over calibration,split,donor,query noise. It is not asserted an ordinary unconditional e-value nor conditional on every full realized calibration; no imagewise/FDR/localization guarantee.',
        design='New768acquisitions,192per four frozen optical-response families,8replicates per24strata. Same stylized physical domain,not sensor/domain transfer. Outcomes aggregated after all complete.',
        exclusions='No physical-sensor efficacy,calibration drift,correlated afterpulsing,unknown-shift robustness,universal dominance,FPGA timing or Q1 acceptance claim.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',bgdev.original.sha(P),len(p['sources']),'sources',len(new),'cases',flush=True)


def check_protocol():
    p=json.loads(P.read_text());development.check_protocol()
    assert p['cases']==cases()
    for path,digest in p['sources'].items():assert bgdev.original.sha(ROOT/path)==digest,path
    return p


def worker(c):
    start=time.perf_counter();h,cal,truth=bgdev.original.simulate_case(c);query=h[4,4]
    cf=lobe_confidence(cal[2],100000);q=donor_proposal(h,cap=cf['eta_upper'])
    interval=single_candidate(query,q,components(q,cap=cf['eta_upper']),(arb(1000)/9).log(),cf['eta_upper'],'confidence_interval')
    point_fit=fit_training(cal[2],100000);eta=point_fit['eta'];pq=fixed_proposal(h,eta)
    point=single_candidate(query,pq,fixed_components(pq,eta),arb(100).log(),eta,'point')
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=9136120200+100000*g+i
    a,b,na,nb=split_cycles(cal[2],100000,seed);joint=score_pair(h,a,b,na,nb)
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),query=query.tolist(),
        calibration_histogram=cal[2].tolist(),confidence=cf,point_eta=eta,point_calibration_fit=point_fit,calibration_split_seed=seed,
        first_calibration_half=a.tolist(),second_calibration_half=b.tolist(),
        methods=dict(background_ci=interval,point=point,joint=joint),
        reports=dict(background_ci=interval['report'],point=point['report'],joint=joint['report']),elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();return db


def read_records():
    db=connect();rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rows


def run():
    p=check_protocol();start=time.perf_counter();db=connect();ids={x[0] for x in db.execute('SELECT id FROM records')}
    pending=[c for c in p['cases'] if c['id'] not in ids]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,c) for c in pending]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%48==0:print('Prospective coupled',len(ids),'of',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),new_records_this_invocation=len(pending),invocation_seconds=time.perf_counter()-start,
        protocol_sha256=bgdev.original.sha(P),sqlite_sha256=bgdev.original.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check_protocol();rows=read_records();assert len(rows)==len(p['cases']);groups={};totals={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[bgdev.original.endpoint(r,m) for r in rows if r['case']['group']==g]
            groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    positive=[r for r in rows if r['case']['secondary']>0];rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates'];boot=np.zeros(rep)
    gain=loss=0
    for r in positive:
        a,b=bgdev.original.endpoint(r,'joint')['correct'],bgdev.original.endpoint(r,'background_ci')['correct'];gain+=a>b;loss+=b>a
    for g in ['far_two','near_two']:
        for s in range(24):
            d=np.array([bgdev.original.endpoint(r,'joint')['correct']-bgdev.original.endpoint(r,'background_ci')['correct'] for r in positive if r['case']['group']==g and r['case']['stratum']==s]);assert len(d)==8
            boot+=rng.choice(d,size=(rep,8),replace=True).sum(1)/len(positive)
    ci=np.quantile(boot,[.025,.975]).tolist();j,b=totals['joint'],totals['background_ci']
    rule=dict(correct_gain=ci[0]>0,no_more_wrong=j['wrong']<=b['wrong'],no_more_false=j['false']<=b['false'])
    computations={m:dict(screened=sum(r['methods'][m]['certified_screened_out'] for r in rows),
        boxes=sum(sum(f.get('bound',{}).get('boxes_evaluated',0) for f in r['methods'][m]['folds']) if m=='joint' else (r['methods'][m]['bound'] or {}).get('boxes_evaluated',0) for r in rows)) for m in METHODS}
    result=dict(status='PROSPECTIVE_CONFIRMATION_COMPLETE',totals=totals,groups=groups,
        paired=dict(gains=gain,losses=loss,rate_difference=(gain-loss)/len(positive),stratified_percentile95_ci=ci),
        success_rule=rule,all_prespecified_success_criteria=all(rule.values()),computation=computations,
        median_detected_calibration_photons=float(np.median([sum(r['calibration_histogram']) for r in rows])),
        protocol_sha256=bgdev.original.sha(P),scope=p['probability_scope'],limitations=p['exclusions'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify_controls','freeze','run','analyze']);args=parser.parse_args()
    if args.action=='verify_controls':verify_controls()
    elif args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
