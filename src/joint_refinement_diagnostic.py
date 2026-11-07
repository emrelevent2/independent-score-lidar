"""Selected numerical diagnosis: tighter cone vs more rectangular boxes."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
from flint import arb
import conditional_joint_confirmation as parent
import joint_bound_diagnostic as witness
from conditional_calibration import calibration_upper
from joint_calibration_envelope import shared_response_cover
from joint_cone_envelope import cone_shared_cover
from certified_ratio_screen import average_log
from envelope_evidence import outward_float

ROOT=parent.ROOT;P=ROOT/'JOINT_REFINEMENT_DIAGNOSTIC_PROTOCOL.json'
OUT=ROOT/'results/joint_refinement_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    p=parent.check_protocol();sources=dict(p['sources'])
    for name in ['joint_refinement_diagnostic.py','joint_cone_envelope.py','joint_bound_diagnostic.py']:
        sources['src/'+name]=parent.bgdev.original.sha(ROOT/'src'/name)
    w=json.loads((witness.OUT/'analysis.json').read_text())
    record=dict(status='FROZEN_BEFORE_REFINED_COVERS',sources=sources,
        parent_protocol_sha256=parent.bgdev.original.sha(parent.P),parent_sqlite_sha256=parent.bgdev.original.sha(parent.OUT/'records.sqlite3'),
        numerical_witness_analysis_sha256=parent.bgdev.original.sha(witness.OUT/'analysis.json'),selected_ids=[r['id'] for r in w['records']],
        settings=dict(rectangular_boxes=4096,cone_boxes=1024,initial_eta=8,initial_delay=16),
        purpose='Same8selectedCI-correct/joint-missed cases,including2whosefeasiblepointprovesnullrefinementalonecannotaccept. Compare increased rectanglebudget with exacteta-endpointcone queryprofile. Samepredictors,calintervals,sparseweights,ratio threshold.',
        inference='Numerical diagnosis on closedoutcome-selectedcases,notnewcohortaccuracy. MincombiningVALIDboundsforsamefamilyrequiresnoadditionalalpha;doesnotmeancombiningdifferentstatisticaltests.')
    P.write_text(json.dumps(record,indent=2));print('Frozen',parent.bgdev.original.sha(P),len(record['selected_ids']),'cases',flush=True)


def worker(row):
    query=np.array(row['query']);folds=[]
    for old in row['methods']['joint']['folds']:
        cf=old['nuisance_confidence'];k,r=old['test_lobe_count'],old['test_lobe_risk']
        provider=lambda e0,e1:calibration_upper(k,r,e0,e1,cf)
        level=outward_float(arb(old['joint_numerator_lower'])-(arb(2000)/9).log(),False)
        cone=cone_shared_cover(query,provider,max_boxes=1024,decision_level=level)
        rect=shared_response_cover(query,provider,max_boxes=4096,decision_level=level)
        assert rect['upper']<=old['bound']['upper']
        item=dict(cone=cone,rectangular_refined=rect,old_upper=old['bound']['upper'])
        for name,upper in [('cone',cone['upper']),('rectangle4096',rect['upper']),('combined',min(cone['upper'],rect['upper'],old['bound']['upper']))]:
            item[name+'_log_ratio_lower']=outward_float(arb(old['joint_numerator_lower'])-arb(upper),False)
        folds.append(item)
    cut=outward_float((arb(1000)/9).log());reports={}
    for name in ['cone','rectangle4096','combined']:
        vals=[f[name+'_log_ratio_lower'] for f in folds];lower=average_log(vals);i=int(np.argmax(vals));old=row['methods']['joint']['folds'][i]
        declared=lower>=cut;primary=old['proposal']['primary'];secondary=old['query_numerator']['location']
        correct=bool(declared and abs(primary-row['truth']['primary'])<=2 and abs(secondary-row['truth']['secondary'])<=2)
        reports[name]=dict(declared=declared,correct=correct,wrong=bool(declared and not correct),log_ratio_lower=lower,reported_fold=i)
    return dict(id=row['id'],folds=folds,reports=reports)


def run():
    p=json.loads(P.read_text());parent.check_protocol();sha=parent.bgdev.original.sha
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
    assert sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    rows={r['id']:r for r in parent.read_records()};w={r['id']:r for r in json.loads((witness.OUT/'analysis.json').read_text())['records']}
    result=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,rows[id]) for id in p['selected_ids']]):
            r=f.result();proof=w[r['id']]
            if proof.get('further_null_refinement_ruled_out'):
                assert not any(v['declared'] for v in r['reports'].values())
            for a,b in zip(r['folds'],proof['folds']):
                assert a['cone']['upper']>=b['feasible_joint_loglik_lower'] and a['rectangular_refined']['upper']>=b['feasible_joint_loglik_lower']
            result.append(r);print('Refined diagnostic',len(result),'of',len(p['selected_ids']),flush=True)
    OUT.mkdir(parents=True,exist_ok=True);result.sort(key=lambda r:r['id'])
    a=dict(status='SELECTED_REFINEMENT_DIAGNOSTIC_COMPLETE',records=result,
        totals={m:dict(accepted=sum(r['reports'][m]['declared'] for r in result),correct=sum(r['reports'][m]['correct'] for r in result),wrong=sum(r['reports'][m]['wrong'] for r in result)) for m in ['cone','rectangle4096','combined']},
        seconds=time.perf_counter()-start,protocol_sha256=sha(P),scope=p['inference'])
    (OUT/'analysis.json').write_text(json.dumps(a,indent=2));print(json.dumps({k:v for k,v in a.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);a=parser.parse_args()
    if a.action=='freeze':freeze()
    else:run()
