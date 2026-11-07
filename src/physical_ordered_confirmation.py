"""New-seed prospective check of the selected optical calibration predictor."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import argparse,ast,json,sqlite3,time,hashlib
import numpy as np
from flint import arb
import physical_ordered_development as development
from ordered_calibration import query_predictor,interval_score
from physical_calibration_predictive import predictors
from conditional_calibration import nuisance_confidence
from calibration_confidence import window_counts
from joint_calibration_model import split_cycles
from background_lobe_confidence import lobe_confidence
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from certified_ratio_screen import sparse_numerator_upper,calibration_box_witness,average_log
from refined_null_screen import witness
from envelope_evidence import outward_float

base=development.parent.base;ROOT=development.ROOT;P=ROOT/'PHYSICAL_ORDERED_CONFIRMATORY_PROTOCOL.json';OUT=ROOT/'results/physical_ordered_confirmation_v1'
METHODS=['grid_ci','point_ci','physical','point']


def cases():
    rows=[]
    for g,group in enumerate(['null_plain','null_lobe','far_two','near_two']):
        templates=[c for c in base.original.cases() if c['group']==group]
        for i in range(192):
            c=dict(templates[i%24]);c.update(id=f'{group}_{i:04}',seed=4036120400+100000*g+i,stratum=i%24);rows.append(c)
    return rows


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


def strong_control(h,pred,cap,point=False):
    eta=pred['fit']['eta'];qw=witness(h[4,4],pred['proposal'],eta if point else min(eta,cap),None if point else cap)
    upper=outward_float(arb(pred['numerator_upper'])-arb(qw['lower']))
    cv=arb(100).log() if point else (arb(1000)/9).log();cut=outward_float(cv);cl=outward_float(cv,False)
    if upper<cl:
        return dict(certified_screened_out=True,refined_null_witness=qw,bound=None,report=dict(declared=False,log_ratio_upper=upper,log_ratio_lower=None,log_threshold_upper=cut,primary=pred['proposal']['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head']))
    result=interval_score(h[4,4],pred,cap,8192,point);result['refined_null_witness']=qw;return result


def score_physical(h,cal,a,b,na,nb,pred):
    cf=nuisance_confidence(cal,100000,delta=Fraction(1,1000));eta=pred['fit']['eta']
    qw=witness(h[4,4],pred['proposal'],eta);folds=[]
    for train,test,nt,nv in [(a,b,na,nb),(b,a,nb,na)]:
        k,r=window_counts(test,nv,43,53);pv=predictors(train,nt,k,r);cw=calibration_box_witness(k,r,eta,cf)
        num=outward_float(arb(pv['physical']['upper'])+arb(pred['numerator_upper']))
        lo=outward_float(arb(cw['lower'])+arb(qw['lower']),False)
        folds.append(dict(test_lobe_count=k,test_lobe_risk=r,predictors=pv,calibration_witness=cw,query_witness=qw,log_ratio_upper=outward_float(arb(num)-arb(lo))))
    upper=average_log([f['log_ratio_upper'] for f in folds],True);cv=(arb(1000)/9).log();cut=outward_float(cv)
    if upper<outward_float(cv,False):
        cf['scope']='ONE full-calibration confidence box,delta=.001; denominator only'
        return dict(certified_screened_out=True,nuisance_confidence=cf,folds=folds,refined_screen_log_ratio_upper=upper,
            report=dict(declared=False,log_ratio_lower=None,log_ratio_upper=upper,log_threshold_upper=cut,primary=pred['proposal']['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head']))
    calc=development.calculate(h,cal,100000,a,b,na,nb,pred,4096)
    calc['nuisance_confidence']['scope']='ONE full-calibration confidence box,delta=.001; denominator only'
    return dict(certified_screened_out=calc['screened']['physical'],nuisance_confidence=calc['nuisance_confidence'],folds=calc['folds'],
        refined_screen_log_ratio_upper=upper,report=calc['reports']['physical'])


def verify_screen():
    development.check();old={r['id']:r for r in development.parent.read()};rows=development.read();n=0
    for row in rows:
        o=old[row['id']];h,cal,_=base.original.simulate_case(row['case']);g=['null_plain','null_lobe','far_two','near_two'].index(row['case']['group']);i=int(row['id'].rsplit('_',1)[1])
        a,b,na,nb=split_cycles(cal[2],100000,2026113010+100*g+i);pred=o['full_calibration_predictor']
        # Refined screen upper only: any subsequently accepted old score must
        # stay below it. Full null-cover replay is unnecessary for this check.
        cf=nuisance_confidence(cal[2],100000,delta=Fraction(1,1000));eta=pred['fit']['eta'];qw=witness(h[4,4],pred['proposal'],eta);uppers=[]
        for train,test,nt,nv in [(a,b,na,nb),(b,a,nb,na)]:
            k,r=window_counts(test,nv,43,53);pv=predictors(train,nt,k,r);cw=calibration_box_witness(k,r,eta,cf)
            uppers.append(outward_float(arb(pv['physical']['upper'])+arb(pred['numerator_upper'])-arb(cw['lower'])-arb(qw['lower'])))
        upper=average_log(uppers,True);rep=row['reports']['physical']
        if rep['log_ratio_lower'] is not None:assert upper>=rep['log_ratio_lower']
        if upper<outward_float((arb(1000)/9).log(),False):assert not rep['declared']
        n+=1
    marker=dict(status='PASS',closed_physical_records_checked=n,source_sha256=base.original.sha(Path(__file__)),screen_source_sha256=base.original.sha(ROOT/'src/refined_null_screen.py'))
    (ROOT/'output/physical_refined_screen_validation.json').write_text(json.dumps(marker,indent=2));print(marker,flush=True)


def freeze():
    if P.exists():raise FileExistsError('Immutable prospective protocol')
    development.check();a=json.loads((development.OUT/'analysis.json').read_text());assert a['selected']=='physical'
    assert json.loads((development.OUT/'integrity.json').read_text())['status']=='PASS'
    v=json.loads((ROOT/'output/physical_refined_screen_validation.json').read_text());assert v['status']=='PASS' and v['source_sha256']==base.original.sha(Path(__file__))
    ctrl=json.loads((ROOT/'output/refined_null_screen_validation.json').read_text());assert ctrl['status']=='PASS' and ctrl['source_sha256']==base.original.sha(ROOT/'src/refined_null_screen.py')
    seeds=set()
    for path in ROOT.glob('*PROTOCOL*.json'):
        try:seeds|={c['seed'] for c in json.loads(path.read_text()).get('cases',[]) if 'seed' in c}
        except (ValueError,TypeError):pass
    assert not seeds&{c['seed'] for c in cases()}
    p=dict(status='FROZEN_BEFORE_NEW_DATA_GENERATION',sources=graph(),cases=cases(),selected_method='physical',development_protocol_sha256=base.original.sha(development.P),development_sqlite_sha256=base.original.sha(development.OUT/'records.sqlite3'),
        history='Prior independent coupled study FAILED282vsCI286correct. Later96closed development: gridCI40,pointCI40,orderedBeta41,physical41,mixture41,point43,0errors each. Physical selected by separately frozen stage rule; not a claim of improvement over point control.',
        settings=dict(calibration_cycles=100000,query_cycles=1000,control_boxes=8192,coupled_boxes_per_fold=4096,split_seed_base=9236120400,bootstrap_seed=9236120999,bootstrap_replicates=10000),
        primary='Selected physical coupled versus gridCI: paired correct-report gain. Strong crossed pointQ+CI and full-calibration point-IRF controls always reported.',
        success_rule=dict(correct_gain='stratified two-sided percentile95%paired CI lower>0',no_more_wrong='wrong positive locations <=gridCI',no_more_false='physical-null declarations <=gridCI'),
        model='Same frozen four optical families,192each,384positives384nulls,8replicates/24strata; NEWraw/auxiliaryseeds;notnewphysicaldomain.',
        numerical='8192total query-profile ceiling/case for each method,earlystopping and certifiedrejectscreen retained. Refined feasible null lower screen only excludes impossible fixed-sparse accepts; does not authorizeaccepts. Physicalandmixture share cover as frozendevelopment; onlyselectedphysical is tested. Ceiling matching is not runtime matching.',
        probability_scope='One full-Cal nuisanceconfidence boxdelta.001;proper opticalcalpredictor trains oppositehalf;Qquery usesallCal+excluded-center donors;mean twofold ratios withMarkov.009 gives marginalperquery.01 underfixedindependent-cyclemodel. NOT ordinary unconditionalevalue,conditionaloneveryCal,FDRorlocationcoverage.',
        exclusions='Simulationsonly;notactualsensorvalidity,calibrationdrift,correlatedafterpulsing,unknownshift,FPGAacceleration,SOTAorQ1proof.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.original.sha(P),len(p['sources']),'sources',len(p['cases']),'cases',flush=True)


def check():
    p=json.loads(P.read_text());development.check();assert p['cases']==cases()
    for n,d in p['sources'].items():assert base.original.sha(ROOT/n)==d,n
    return p


def worker(c):
    start=time.perf_counter();h,cal,truth=base.original.simulate_case(c);pred=query_predictor(h,cal[2],100000);cf=lobe_confidence(cal[2],100000);cap=cf['eta_upper']
    q=donor_proposal(h,cap=cap);model=components(q,cap=cap);num=numeric_and_sparse(h[4,4],model)
    gp=dict(fit=dict(eta=q['eta_donor']),proposal=q,numerator=num,numerator_upper=sparse_numerator_upper(h[4,4],model,num['indices']))
    methods=dict(grid_ci=strong_control(h,gp,cap),point_ci=strong_control(h,pred,cap),point=strong_control(h,pred,cap,True))
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=9236120400+100000*g+i
    a,b,na,nb=split_cycles(cal[2],100000,seed);methods['physical']=score_physical(h,cal[2],a,b,na,nb,pred)
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),calibration_histogram=cal[2].tolist(),interval_confidence=cf,full_calibration_predictor=pred,grid_predictor=gp,
        calibration_split_seed=seed,methods=methods,reports={m:methods[m]['report'] for m in METHODS},elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={x[0] for x in db.execute('SELECT id FROM records')};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,c) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%48==0:print('Prospective physical',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=base.original.sha(P),sqlite_sha256=base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);groups={};totals={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    pos=[r for r in rs if r['case']['secondary']>0];rng=np.random.default_rng(p['settings']['bootstrap_seed']);rep=p['settings']['bootstrap_replicates'];boot=np.zeros(rep)
    def correct(r,m):return base.original.endpoint(r,m)['correct']
    gains=sum(correct(r,'physical')>correct(r,'grid_ci') for r in pos);losses=sum(correct(r,'grid_ci')>correct(r,'physical') for r in pos)
    for g in ['far_two','near_two']:
        for st in range(24):
            d=np.array([correct(r,'physical')-correct(r,'grid_ci') for r in pos if r['case']['group']==g and r['case']['stratum']==st]);assert len(d)==8
            boot+=rng.choice(d,size=(rep,8),replace=True).sum(1)/len(pos)
    ci=np.quantile(boot,[.025,.975]).tolist();x,y=totals['physical'],totals['grid_ci'];rule=dict(correct_gain=ci[0]>0,no_more_wrong=x['wrong']<=y['wrong'],no_more_false=x['false']<=y['false'])
    v=dict(status='PROSPECTIVE_PHYSICAL_CONFIRMATION_COMPLETE',totals=totals,groups=groups,paired=dict(gains=gains,losses=losses,rate_difference=(gains-losses)/len(pos),stratified_percentile95_ci=ci),success_rule=rule,all_prespecified_success_criteria=all(rule.values()),protocol_sha256=base.original.sha(P),scope=p['probability_scope'],limitations=p['exclusions'])
    (OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['verify_screen','freeze','run','analyze']);a=ap.parse_args()
    if a.action=='verify_screen':verify_screen()
    elif a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:analyze()
