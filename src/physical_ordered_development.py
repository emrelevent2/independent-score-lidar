"""Closed development: optical-model calibration predictors, fixed comparisons."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,json,sqlite3,time,hashlib
from flint import arb
import ordered_development as parent
from ordered_calibration import query_predictor
from joint_calibration_model import split_cycles
from calibration_confidence import window_counts
from physical_calibration_predictive import predictors
from conditional_calibration import nuisance_confidence,calibration_upper
from certified_ratio_screen import calibration_box_witness,single_irf_witness,average_log
from joint_calibration_envelope import shared_response_cover
from envelope_evidence import outward_float
from fractions import Fraction

ROOT=parent.ROOT;P=ROOT/'PHYSICAL_ORDERED_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/physical_ordered_development_v1'
METHODS=['grid_ci','point_ci','ordered_beta','physical','mixture','point'];NEW=['physical','mixture']


def graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for n in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for module in names:
                path=module.split('.')[0]+'.py'
                if (ROOT/'src'/path).exists() and path not in seen:pending.append(path)
    return {f'src/{name}':parent.base.original.sha(ROOT/'src'/name) for name in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    old=parent.check();a=json.loads((parent.OUT/'analysis.json').read_text());assert a['selected'] is None
    assert json.loads((parent.OUT/'integrity.json').read_text())['status']=='PASS'
    p=dict(status='FROZEN_BEFORE_NEW_PREDICTOR_OUTCOMES',sources=graph(),cases=old['cases'],parent_protocol_sha256=parent.base.original.sha(parent.P),parent_sqlite_sha256=parent.base.original.sha(parent.OUT/'records.sqlite3'),
        history='Previous development selection rule FAILED: grid40,pointCI40,orderedBeta41,point43 correct;0errors each. Strict pointCI-improvement gate failed. This is a separate closed model-development stage,not reinterpretation of that rule.',
        settings=dict(calibration_cycles=100000,query_cycles=1000,boxes_per_fold=4096,ratio_budget='9/1000',one_shared_confidence_failure='1/1000',split_seed_base=2026113010),
        candidates=dict(physical='Pure binomial predictive probability from opposite-half optical-response first-event MLE; conditional on scored R',mixture='Fixed1/2optical-model predictor+1/2Jeffreys beta-binomial predictor; never max'),
        controls='Identical cached closed96gridCI,pointCI,orderedBeta,point controls. Same query Q,one full-Cal nuisanceCF and4096/fold cover. New candidates share ONE cover with early-stop level satisfying both; not minimum-combined with old cover and no extra search budget.',
        selection='Eligible if strictly more correct than BOTH gridCI and pointCI,with wrong/null counts no greater than either. Select largest correct among eligible; ties prefer purephysical. Point estimate control is always reported. Any efficacy claim requires NEW independently frozen data.',
        scope='Confidence-padded marginalperquery.01,as ordered reuse proof;NOT unconditional e-values,notlocation/FDRcoverage;closedstylizedsimulationsonly;noQ1/SOTA/physical-sensorclaim.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.original.sha(P),len(p['sources']),'sources',flush=True)


def check():
    p=json.loads(P.read_text());parent.check()
    assert parent.base.original.sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    for n,d in p['sources'].items():assert parent.base.original.sha(ROOT/n)==d,n
    return p


def calculate(h,cal,ncal,a,b,na,nb,pred,max_boxes=4096):
    cf=nuisance_confidence(cal,ncal,delta=Fraction(1,1000));folds=[];eta=pred['fit']['eta'];qw=single_irf_witness(h[4,4],pred['proposal'],eta)
    for train,test,nt,nv in [(a,b,na,nb),(b,a,nb,na)]:
        k,r=window_counts(test,nv,43,53);pv=predictors(train,nt,k,r);cw=calibration_box_witness(k,r,eta,cf)
        witness=outward_float(arb(cw['lower'])+arb(qw['lower']),False);nums={}
        for m in NEW:
            cn=pv[m];lo=outward_float(arb(cn['lower'])+arb(pred['numerator']['lower']),False)
            hi=outward_float(arb(cn['upper'])+arb(pred['numerator_upper']))
            nums[m]=dict(joint_numerator_lower=lo,log_ratio_upper=outward_float(arb(hi)-arb(witness)))
        folds.append(dict(test_lobe_count=k,test_lobe_risk=r,predictors=pv,calibration_witness=cw,query_witness=qw,variants=nums))
    cut=outward_float((arb(1000)/9).log());cutlow=outward_float((arb(1000)/9).log(),False)
    uppers={m:average_log([f['variants'][m]['log_ratio_upper'] for f in folds],True) for m in NEW}
    active=[m for m in NEW if uppers[m]>=cutlow]
    if active:
        for fold in folds:
            k,r=fold['test_lobe_count'],fold['test_lobe_risk'];provider=lambda e0,e1:calibration_upper(k,r,e0,e1,cf)
            level=min(outward_float(arb(fold['variants'][m]['joint_numerator_lower'])-(arb(2000)/9).log(),False) for m in active)
            bound=shared_response_cover(h[4,4],provider,max_boxes=max_boxes,decision_level=level);fold['bound']=bound
            for m in active:fold['variants'][m]['log_ratio_lower']=outward_float(arb(fold['variants'][m]['joint_numerator_lower'])-arb(bound['upper']),False)
    reports={}
    for m in NEW:
        lower=average_log([f['variants'][m]['log_ratio_lower'] for f in folds]) if m in active else None
        reports[m]=dict(declared=lower is not None and lower>=cut,log_ratio_lower=lower,log_ratio_upper=uppers[m],log_threshold_upper=cut,
            primary=pred['proposal']['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head'])
    return dict(nuisance_confidence=cf,folds=folds,reports=reports,screened={m:m not in active for m in NEW})


def worker(old):
    start=time.perf_counter();c=old['case'];h,cal,truth=parent.base.original.simulate_case(c);pred=query_predictor(h,cal[2],100000)
    assert pred==old['full_calibration_predictor']
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=2026113010+g*100+i
    a,b,na,nb=split_cycles(cal[2],100000,seed);new=calculate(h,cal[2],100000,a,b,na,nb,pred)
    reports={m:old['reports'][m] for m in ['grid_ci','point_ci','point']};reports['ordered_beta']=old['reports']['ordered'];reports.update(new['reports'])
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),new=new,reports=reports,elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit()
    ids={r[0] for r in db.execute('SELECT id FROM records')};old={r['id']:r for r in parent.read()};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,old[c['id']]) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Physical predictive',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=parent.base.original.sha(P),sqlite_sha256=parent.base.original.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);groups={};totals={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[parent.base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    eligible=[m for m in NEW if all(totals[m]['correct']>totals[b]['correct'] and all(totals[m][k]<=totals[b][k] for k in ['wrong','false']) for b in ['grid_ci','point_ci'])]
    selected=max(eligible,key=lambda m:totals[m]['correct']) if eligible else None
    v=dict(status='CLOSED_PHYSICAL_PREDICTOR_DEVELOPMENT_COMPLETE',totals=totals,groups=groups,eligible=eligible,selected=selected,protocol_sha256=parent.base.original.sha(P),scope=p['scope'])
    (OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:analyze()
