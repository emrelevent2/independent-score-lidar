"""Closed two-window calibration likelihood development,with one-window ablation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,sqlite3
import numpy as np
from flint import arb
import causal_calibration_development as parent
from causal_calibration_v2 import score_lobe,score_dark,joint_numerator,two_window_upper
from conditional_calibration import calibration_upper
from calibration_confidence import window_counts
from joint_calibration_model import calibration_basis
from joint_calibration_envelope import shared_response_cover
from refined_null_screen import witness
from certified_ratio_screen import calibration_box_witness
from envelope_evidence import outward_float

ROOT=parent.ROOT;P=ROOT/'CAUSAL_TWO_WINDOW_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/causal_two_window_development_v1'
METHODS=['grid_ci','point_ci','causal_old','one_window','two_windows','point'];NEW=['one_window','two_windows']


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    old=parent.check();assert json.loads((parent.OUT/'analysis.json').read_text())['continue_to_confirmation'] is False
    sources=dict(old['sources'])
    for n in ['causal_two_window_development.py','causal_calibration_v2.py']:
        sources['src/'+n]=parent.base.original.sha(ROOT/'src'/n)
    p=dict(status='FROZEN_BEFORE_TWO_WINDOW_OUTCOMES',sources=sources,cases=old['cases'],parent_protocol_sha256=parent.base.original.sha(parent.P),parent_sqlite_sha256=parent.base.original.sha(parent.OUT/'records.sqlite3'),
        settings=dict(calibration_cycles=10000,query_cycles=1000,max_query_boxes_each=4096,query_budget='9/1000',one_fullCal_confidence_failure='1/1000'),
        primary='two_windows vs BOTHstrongpointCI andgridCI;strictly more correct andneither error count greater. One-window quantile predictor is ablation,not alternateprimary.',
        hypothesis='Scoring dark counts with lobe counts and SHARINGthe background nuisance prevents null explanation that lowersbackground only to absorb a weak echo into IRF. Existinginterval-only treatment loses thatlikelihoodconstraint.',
        construction='Qc=QK(K|0..42,Z)*QD(D|0..127);D128..255,K43..53,randomriskR_D includesunscoredmiddle. QDtrainsbackground on64..127only. Nineposteriorquantilebackgroundpoints are explicitproper equal-weightmixtures,not exactBayesianintegration. Calnull profiles2binomialcountswithDIFFERENTconditionalrisksets andsharedsigma/beta,eta. QueryQfullCal;oneCFdelta=.001 denominatoronly.',
        probability='IntegrateXfirstconditionalallCal,Z;thenDconditionalwholepast;thenunscoredmiddle;thenKconditionalprefix. Proper sequentialdensity hasmean1attrueparameters. PointwisedominationonONEfullCalCFcoverageevent gives.009+.001=.01 marginal bound. Not unconditionalevalue,conditionalonallCal,location/FDRcoverage.',
        controls='Cached SAME10kcalibration/1000query/4096queryceilings from parent: grid28,pointCI30,causalold30,point42 correct,all0errors. ParentprimaryFAILED,tieagainststrongCI;preserved. No extra cycles.',
        scope='All96CLOSEDcases,not selected errors;notindependentefficacy,Q1/SOTA,physicalsensor,adaptive-stoppingorFPGAproof.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.original.sha(P),flush=True)


def check():
    p=json.loads(P.read_text());parent.check();sha=parent.base.original.sha
    for n,d in p['sources'].items():assert sha(ROOT/n)==d,n
    assert sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256'];return p


def cal_two_witness(k,r,d,rd,eta,cf):
    lo,hi=calibration_basis(eta,eta)
    sigma=(arb(cf['signal_main_lower'])+arb(cf['signal_main_upper']))/2 if cf['informative'] else arb(.05)
    beta=(arb(cf['background_lower'])+arb(cf['background_upper']))/2 if cf['informative'] else arb(.00002)
    F=(arb(cf['forward_leakage_lower'])+arb(cf['forward_leakage_upper']))/2 if cf['informative'] else arb(.000001)
    leakage=(arb(float(lo[5,0]))+arb(float(hi[5,0])))/2
    H=sigma*(arb(eta)+F)+11*beta;HD=sigma*leakage+128*beta
    value=-(r-k)*H-(rd-d)*HD
    if k:value+=k*(-(-H).expm1()).log()
    if d:value+=d*(-(-HD).expm1()).log()
    return dict(lower=outward_float(value,False),kind='Feasible sharedsigma/beta point in same calibration box/shape-rate relaxation',eta=eta)


def worker(old):
    start=time.perf_counter();c=old['case'];h,cal,truth=parent.base.original.simulate_case(c);cal=cal[1];pred=old['full_calibration_predictor'];cf=old['methods']['causal']['nuisance_confidence'];donor=old['methods']['causal']['calibration_prior_donor'];eta=pred['fit']['eta']
    k,r=window_counts(cal,10000,43,53);d,rd=window_counts(cal,10000,128,255)
    qk=score_lobe(cal[:43],10000,donor['eta_donor'],k);qd=score_dark(cal[:128],10000,d);joint=joint_numerator(qk,qd)
    assert qk['risk']==r and qd['risk']==rd
    qw=witness(h[4,4],pred['proposal'],eta);cw1=calibration_box_witness(k,r,eta,cf);cw2=cal_two_witness(k,r,d,rd,eta,cf)
    methods={};cv=(arb(1000)/9).log();cut=outward_float(cv)
    for m,cn,cw in [('one_window',qk,cw1),('two_windows',joint,cw2)]:
        lower=outward_float(arb(cn['lower'])+arb(pred['numerator']['lower']),False);upper=outward_float(arb(cn['upper'])+arb(pred['numerator_upper'])-arb(qw['lower'])-arb(cw['lower']))
        screen=upper<outward_float(cv,False)
        rep=dict(declared=False,log_ratio_lower=None,log_ratio_upper=upper,log_threshold_upper=cut,primary=pred['proposal']['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head'])
        bound=None
        if not screen:
            provider=(lambda e0,e1:calibration_upper(k,r,e0,e1,cf)) if m=='one_window' else (lambda e0,e1:two_window_upper(k,r,d,rd,e0,e1,cf))
            level=outward_float(arb(lower)-arb(cut),False);bound=shared_response_cover(h[4,4],provider,max_boxes=4096,decision_level=level,calibration_observed_bins=1 if m=='one_window' else 2)
            score=outward_float(arb(lower)-arb(bound['upper']),False);rep.update(declared=score>=cut,log_ratio_lower=score)
        methods[m]=dict(joint_numerator_lower=lower,query_witness=qw,calibration_witness=cw,certified_screened_out=screen,bound=bound,report=rep)
    reports={m:old['reports'][m] for m in ['grid_ci','point_ci','point']};reports['causal_old']=old['reports']['causal'];reports.update({m:v['report'] for m,v in methods.items()})
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=old['raw_histogram_sha256'],calibration_predictors=dict(lobe=qk,dark=qd,joint=joint),nuisance_confidence=cf,methods=methods,reports=reports,elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={r[0] for r in db.execute('SELECT id FROM records')};old={r['id']:r for r in parent.read()};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,old[c['id']]) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Causal two-window',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=parent.base.original.sha(P),sqlite_sha256=parent.base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);totals={};groups={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[parent.base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    x=totals['two_windows'];rule={b:dict(more_correct=x['correct']>totals[b]['correct'],no_more_wrong=x['wrong']<=totals[b]['wrong'],no_more_false=x['false']<=totals[b]['false']) for b in ['grid_ci','point_ci']}
    v=dict(status='CLOSED_CAUSAL_TWO_WINDOW_COMPLETE',totals=totals,groups=groups,rule=rule,continue_to_confirmation=all(all(v.values()) for v in rule.values()),protocol_sha256=parent.base.original.sha(P),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run','analyze']);args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
