"""Denser quadrature on fixed and near-threshold saved acquisitions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,sqlite3,sys,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import calibrated_recovery_study as study
import calibrated_recovery as model
from null_information_bank import NullInformationBank


def worker(task):
    record,threshold=task;p=study.check();name=record['name'];cfg=p['settings'][name];sensor=p['sensor_calibration'][name]
    observed,_,_,_=study.acquisition((name,'evaluate',record['index']),p)
    assert study.hashlib.sha256(observed.tobytes()).hexdigest()==record['data_sha256']
    pulse=study.template(cfg);theta=record['null_fit']['theta']
    bank=NullInformationBank.load(study.OUT/('bank_'+name+'.npz'));h,_=bank.lookup(theta,0)
    geo=model.prepare(observed,64,cfg['td'],sensor['mu'],sensor['sd'],sensor['jitter'],
        recovery_order=16,jitter_order=5,endpoint_order=4)
    refined=model.scan(pulse,theta,geo,h);original=record['methods']['calibrated']
    return dict(id=record['id'],original_score=original['score'],refined_score=refined['score'],
        absolute_score_difference=abs(original['score']-refined['score']),
        relative_score_difference=abs(original['score']-refined['score'])/max(1.,abs(original['score'])),
        mean_logdensity_difference=abs(original['loglik']-refined['loglik'])/(len(observed)-1),
        fixed_threshold_decision_unchanged=(original['score']>threshold)==(refined['score']>threshold))


def main():
    p=study.check();analysis=json.loads((study.OUT/'analysis.json').read_text())
    with sqlite3.connect(study.OUT/'records.sqlite3') as db:
        records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
    tasks=[]
    for name in p['settings']:
        cohort=[r for r in records if r['name']==name and r['kind']=='evaluate']
        threshold=analysis['thresholds'][name]['calibrated']
        fixed=next(r for r in cohort if r['index']==0)
        positive=min((r for r in cohort if r['positive']),key=lambda r:abs(r['methods']['calibrated']['score']-threshold))
        null=min((r for r in cohort if not r['positive']),key=lambda r:abs(r['methods']['calibrated']['score']-threshold))
        selected={r['id']:r for r in [fixed,positive,null]}
        tasks.extend((r,threshold) for r in selected.values())
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            row=future.result();rows.append(row);print(row,flush=True)
    result=dict(status='COMPLETE',records=len(rows),selection='First positive and nearest-to-threshold positive and null per condition',
        maximum_relative_score_difference=max(r['relative_score_difference'] for r in rows),
        maximum_mean_logdensity_difference=max(r['mean_logdensity_difference'] for r in rows),
        fixed_threshold_decision_changes=sum(not r['fixed_threshold_decision_unchanged'] for r in rows),
        seconds=time.perf_counter()-start,rows=rows)
    (study.OUT/'quadrature_refinement.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
