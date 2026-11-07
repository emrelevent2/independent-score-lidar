"""Closed10k-calibration development of complete,causally scored lobe data."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import argparse,ast,json,sqlite3,time,hashlib
import numpy as np
from flint import arb
import background_lobe_development as base
from causal_calibration import score_lobe
from ordered_calibration import query_predictor
from physical_ordered_confirmation import strong_control
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from background_lobe_confidence import lobe_confidence
from conditional_calibration import nuisance_confidence,calibration_upper
from calibration_confidence import window_counts
from certified_ratio_screen import sparse_numerator_upper,calibration_box_witness
from refined_null_screen import witness
from joint_calibration_envelope import shared_response_cover
from envelope_evidence import outward_float

ROOT=base.ROOT;P=ROOT/'CAUSAL_CALIBRATION_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/causal_calibration_development_v1'
METHODS=['grid_ci','point_ci','causal','point'];NCAL=10000


def graph():
    todo=[Path(__file__).name];seen=set()
    while todo:
        n=todo.pop()
        if n in seen:continue
        seen.add(n)
        for node in ast.walk(ast.parse((ROOT/'src'/n).read_text())):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in names:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:todo.append(local)
    return {f'src/{n}':base.original.sha(ROOT/'src'/n) for n in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable development')
    old=base.check_protocol();p=dict(status='FROZEN_BEFORE_CAUSAL_LOBE_OUTCOMES',sources=graph(),cases=old['cases'],parent_protocol_sha256=base.original.sha(base.P),
        settings=dict(calibration_cycles=NCAL,query_cycles=1000,query_boxes=4096,markov_budget='9/1000',one_confidence_failure='1/1000'),
        question='Can replacing random half-calibration testing by ALL lobe counts improve finite-calibration sensitivity at SAME10kcycles? CalQ depends only on earlier0..42bins and raw-query-excluded donors. QueryQ may use ALLcalibration by ordered normalization.',
        predictor='33component proper rational mixture:eta0mass1/2,donoreta1/4,eightuniformmidpoints1/4;threeprefix-backgroundfactors.5/1/2withweights1/4,1/2,1/4. Mainhazardfeature fromprefix27..37;earlybackground0..19. ScoredK43..53conditionedonR=N-counts0..42. Futuredarkcounts usedONLY in denominatorCF andqueryQ,notCalQ.',
        primary='Causal full-lobe statistic vs BOTHgridCI andstrongpointQ+CI,onall96CLOSEDcases. All methods use10kcalibration and4096queryprofile ceiling,earlystopping/screens;runtime notassertedmatched.',
        rule='Strictly more correct than bothCIcontrols,andwrong/nullcounts no greater than either. Point-IRF control always reported,no finitecalibration guarantee assigned to it.',
        probability='Conditional on whole calibrationprefix,Z,R,the lobeK isbinomial;CalQ proper. IntegratequeryXfirstconditionalallCal,Z,thenKconditionalpast. OnefullCal nuisanceCFdelta.001 in denominator only,pointwise dominationoncoverageevent;Markov.009+delta=.01. Not ordinaryunconditionalevalue,notconditionalonallCal,notlocation/FDRcoverage.',
        attribution='Causal/predictive normalization and nuisanceconfidence correction are known principles;no newgeneraltheorem. Model-specific data reuse and useful empirical benefit must be established.',
        scope='CLOSEDdevelopmentonly;noindependentaccuracy,Q1,SOTA,real-sensor,FPGA orcalibrationdriftclaim;no additional cycles collected.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.original.sha(P),len(p['sources']),'sources',flush=True)


def check():
    p=json.loads(P.read_text());base.check_protocol()
    for n,d in p['sources'].items():assert base.original.sha(ROOT/n)==d,n
    return p


def causal_score(h,cal,pred,ncal):
    donor=donor_proposal(h,cap=.25);assert donor['raw_center_excluded']
    k,r=window_counts(cal,ncal,43,53);cn=score_lobe(cal[:43],ncal,donor['eta_donor'],k);assert cn['scored_risk']==r
    cf=nuisance_confidence(cal,ncal,delta=Fraction(1,1000));cf['scope']='ONE fullCal nuisancebox,denominator only'
    eta=pred['fit']['eta'];qw=witness(h[4,4],pred['proposal'],eta);cw=calibration_box_witness(k,r,eta,cf)
    lower=outward_float(arb(cn['lower'])+arb(pred['numerator']['lower']),False)
    upper=outward_float(arb(cn['upper'])+arb(pred['numerator_upper'])-arb(qw['lower'])-arb(cw['lower']))
    cv=(arb(1000)/9).log();cut=outward_float(cv);screen=upper<outward_float(cv,False)
    report=dict(declared=False,log_ratio_lower=None,log_ratio_upper=upper,log_threshold_upper=cut,primary=pred['proposal']['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head'])
    result=dict(calibration_predictor=cn,calibration_prior_donor=donor,nuisance_confidence=cf,query_witness=qw,calibration_witness=cw,joint_numerator_lower=lower,certified_screened_out=screen,bound=None,report=report)
    if screen:return result
    provider=lambda e0,e1:calibration_upper(k,r,e0,e1,cf)
    level=outward_float(arb(lower)-arb(cut),False);bound=shared_response_cover(h[4,4],provider,max_boxes=4096,decision_level=level)
    score=outward_float(arb(lower)-arb(bound['upper']),False);report.update(declared=score>=cut,log_ratio_lower=score);result['bound']=bound
    return result


def worker(c):
    start=time.perf_counter();h,cal,truth=base.original.simulate_case(c);cal=cal[1];pred=query_predictor(h,cal,NCAL);cf=lobe_confidence(cal,NCAL);cap=cf['eta_upper']
    q=donor_proposal(h,cap=cap);model=components(q,cap=cap);num=numeric_and_sparse(h[4,4],model)
    gp=dict(fit=dict(eta=q['eta_donor']),proposal=q,numerator=num,numerator_upper=sparse_numerator_upper(h[4,4],model,num['indices']))
    # strong_control's fixed4096? Existing helper uses8192;use its refined
    # feasible screen locally,then the desired4096cover ceiling explicitly.
    from ordered_calibration import interval_score
    def control(p,point=False):
        eta=p['fit']['eta'];w=witness(h[4,4],p['proposal'],eta if point else min(eta,cap),None if point else cap)
        cv=arb(100).log() if point else (arb(1000)/9).log();upper=outward_float(arb(p['numerator_upper'])-arb(w['lower']))
        if upper<outward_float(cv,False):return dict(bound=None,certified_screened_out=True,refined_null_witness=w,report=dict(declared=False,log_ratio_lower=None,log_ratio_upper=upper,log_threshold_upper=outward_float(cv),primary=p['proposal']['primary'],secondary=p['numerator']['location'],head=p['numerator']['head']))
        result=interval_score(h[4,4],p,cap,4096,point);result['refined_null_witness']=w;return result
    methods=dict(grid_ci=control(gp),point_ci=control(pred),point=control(pred,True),causal=causal_score(h,cal,pred,NCAL))
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),calibration_histogram=cal.tolist(),interval_confidence=cf,full_calibration_predictor=pred,methods=methods,reports={m:methods[m]['report'] for m in METHODS},elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={r[0] for r in db.execute('SELECT id FROM records')};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,c) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Causal calibration',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=base.original.sha(P),sqlite_sha256=base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);totals={};groups={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    x=totals['causal'];rule={b:dict(more_correct=x['correct']>totals[b]['correct'],no_more_wrong=x['wrong']<=totals[b]['wrong'],no_more_false=x['false']<=totals[b]['false']) for b in ['grid_ci','point_ci']}
    v=dict(status='CLOSED_CAUSAL_DEVELOPMENT_COMPLETE',totals=totals,groups=groups,rule=rule,continue_to_confirmation=all(all(v.values()) for v in rule.values()),protocol_sha256=base.original.sha(P),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run','analyze']);args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
