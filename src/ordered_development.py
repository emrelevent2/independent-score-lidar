"""Closed 96-case development; full-calibration reuse and crossed control."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import argparse,ast,json,sqlite3,time,hashlib
import numpy as np
from flint import arb
import background_lobe_development as base
from background_lobe_confidence import lobe_confidence
from conditional_calibration import nuisance_confidence
from joint_calibration_model import split_cycles
from ordered_calibration import query_predictor,interval_score,score_ordered
from shoulder_alternative import donor_proposal,components,numeric_and_sparse
from certified_ratio_screen import sparse_numerator_upper

ROOT=base.ROOT;P=ROOT/'ORDERED_CALIBRATION_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/ordered_calibration_development_v1'
METHODS=['grid_ci','point_ci','ordered','point']


def graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in names:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:pending.append(local)
    return {f'src/{name}':base.original.sha(ROOT/'src'/name) for name in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    parent=base.check_protocol()
    p=dict(status='FROZEN_BEFORE_ORDERED_CLOSED_DEVELOPMENT',sources=graph(),cases=parent['cases'],
        parent_protocol_sha256=base.original.sha(base.P),
        settings=dict(calibration_cycles=100000,query_cycles=1000,control_max_boxes=8192,joint_max_boxes_per_fold=4096,
            split_seed_base=2026113010,ratio_alpha='9/1000',confidence_delta='1/1000'),
        methods=dict(grid_ci='Background-aware five-endpoint eta interval + original three-eta-grid query predictor',
            point_ci='SAME eta interval + proper full-calibration-MLE point-IRF query predictor',
            ordered='Full-calibration-MLE query predictor shared by folds; calibration Q trained ONLY on opposite half; ONE full-calibration nuisance box delta=.001; original rectangular continuous joint cover,4096 boxes/fold',
            point='Full-calibration-MLE query predictor and fixed-eta null; alpha=.01; no finite-calibration physical-null guarantee'),
        primary='Development only: point_ci and ordered compared with grid_ci AND strong point control. Numerical ceilings matched at8192 query profiles per case,early stopping/screens retained; not a runtime-match assertion.',
        selection='Prefer point_ci if it strictly increases correct vs grid_ci without increasing either error count. Prefer ordered over point_ci only if strictly more correct and neither error count greater. No independent-efficacy claim; any chosen method requires separately frozen NEW data.',
        rationale='Ordered conditional normalization integrates independent query first,so query Q may use all calibration. Confidence-set reuse is denominator-only with pointwise domination on ONE shared coverage event. Known predictive-chain/confidence-set principles,not a novel general theorem.',
        exclusions='Closed simulations only; no physical validation,Q1/SOTA claim,ordinary unconditional e-value,conditional-on-every-calibration or localization guarantee.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',base.original.sha(P),len(p['sources']),'sources',flush=True)


def check():
    p=json.loads(P.read_text());base.check_protocol()
    for name,digest in p['sources'].items():assert base.original.sha(ROOT/name)==digest,name
    return p


def worker(c):
    start=time.perf_counter();h,cal,truth=base.original.simulate_case(c);query=h[4,4]
    cf=lobe_confidence(cal[2],100000);cap=cf['eta_upper'];pred=query_predictor(h,cal[2],100000)
    q=donor_proposal(h,cap=cap);model=components(q,cap=cap);num=numeric_and_sparse(query,model)
    gp=dict(fit=dict(eta=q['eta_donor']),proposal=q,numerator=num,numerator_upper=sparse_numerator_upper(query,model,num['indices']))
    methods=dict(grid_ci=interval_score(query,gp,cap,8192),point_ci=interval_score(query,pred,cap,8192),point=interval_score(query,pred,cap,8192,True))
    g=['null_plain','null_lobe','far_two','near_two'].index(c['group']);i=int(c['id'].rsplit('_',1)[1]);seed=2026113010+g*100+i
    a,b,na,nb=split_cycles(cal[2],100000,seed)
    methods['ordered']=score_ordered(h,cal[2],100000,a,b,na,nb,pred,4096)
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
        calibration_histogram=cal[2].tolist(),interval_confidence=cf,full_calibration_predictor=pred,grid_predictor=gp,
        calibration_split_seed=seed,methods=methods,reports={m:methods[m]['report'] for m in METHODS},elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rows


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit()
    ids={r[0] for r in db.execute('SELECT id FROM records')};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,c) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES (?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Ordered closed development',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    result=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=base.original.sha(P),sqlite_sha256=base.original.sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check();rows=read();assert len(rows)==len(p['cases']);groups={};totals={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[base.original.endpoint(r,m) for r in rows if r['case']['group']==g]
            groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['far_two','near_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['far_two','near_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    def beats(a,b):return totals[a]['correct']>totals[b]['correct'] and all(totals[a][k]<=totals[b][k] for k in ['wrong','false'])
    selected='point_ci' if beats('point_ci','grid_ci') else None
    if selected and beats('ordered','point_ci'):selected='ordered'
    result=dict(status='CLOSED_DEVELOPMENT_COMPLETE',totals=totals,groups=groups,selected=selected,
        scope=p['exclusions'],protocol_sha256=base.original.sha(P),
        computation={m:dict(boxes=sum(sum(f.get('bound',{}).get('boxes_evaluated',0) for f in r['methods'][m]['folds']) if m=='ordered' else (r['methods'][m]['bound'] or {}).get('boxes_evaluated',0) for r in rows),screened=sum(r['methods'][m]['certified_screened_out'] for r in rows)) for m in METHODS})
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:analyze()
