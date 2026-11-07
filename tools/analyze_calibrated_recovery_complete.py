"""Recount the full cohort, including explicitly unavailable exposure scores.

All499 null draws remain in calibration. An unavailable censored statistic is
the non-declaration outcome (-infinity for ordering), not a deleted draw.
Raw inference outputs and all frozen scientific source files remain unchanged.
"""
from pathlib import Path
import sys,json,sqlite3,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import calibrated_recovery_study as study


def main():
    p=study.check();out=study.OUT
    with sqlite3.connect(out/'records.sqlite3') as db:
        records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY id')]
    assert len(records)==6302
    thresholds={};rows=[];unavailable_training={}
    for name in p['settings']:
        train=[r for r in records if r['name']==name and r['kind']=='threshold'];assert len(train)==499
        thresholds[name]={};unavailable_training[name]={}
        for method in ['calibrated','censored']:
            values=[r['methods'][method] for r in train]
            scores=np.array([v['score'] if v['applicable'] else -np.inf for v in values],float)
            assert not np.any(np.isnan(scores)) and np.count_nonzero(np.isfinite(scores))>=5
            thresholds[name][method]=float(np.sort(scores)[494])
            unavailable_training[name][method]=sum(not v['applicable'] for v in values)
        for kind in ['evaluate','public']:
            subset=[r for r in records if r['name']==name and r['kind']==kind]
            if not subset:continue
            assert len(subset)==(128 if kind=='public' else (320 if name=='measured_high' else 96))
            groups=[None] if name!='measured_high' or kind=='public' else [.0625,.125,.25]
            for ratio in groups:
                cohort=[r for r in subset if not r['positive'] or ratio is None or r['truth']['ratio']==ratio]
                for method in ['calibrated','censored']:
                    npos=sum(r['positive'] for r in cohort);nnull=len(cohort)-npos
                    declare=correct=wrong=false=unavailable=0;errors=[]
                    for r in cohort:
                        v=r['methods'][method]
                        if not v['applicable']:unavailable+=1;continue
                        decision=v['score']>thresholds[name][method]
                        if not r['positive']:false+=int(decision);continue
                        declare+=int(decision)
                        if decision:
                            ep=abs((v['primary']-r['truth']['primary']+32)%64-32)
                            ed=abs((v['delay']-r['truth']['secondary']+32)%64-32)
                            correct+=int(ep<=2 and ed<=2);wrong+=int(ep>2 or ed>2);errors.append(ed)
                    rows.append(dict(name=name,kind=kind,ratio=ratio,method=method,
                        positives=npos,nulls=nnull,declared=declare,correct=correct,wrong=wrong,
                        false_alarms=false,unavailable=unavailable,
                        false_alarm_interval=study.interval(false,nnull) if nnull else None,
                        conditional_secondary_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                        threshold=thresholds[name][method]))
    result=dict(status='COMPLETE',protocol_sha256=study.sha(study.P),records=len(records),thresholds=thresholds,rows=rows,
        sensor_calibration=p['sensor_calibration'],sensor_output_hashes={name:study.sha(out/('sensor_'+name+'.json')) for name in p['settings']},
        unavailable_calibration_draws=unavailable_training,
        unavailable_rule='All499 draws retained; unavailable exposure statistic is a non-declaration, ordered as -infinity. Query availability is reported separately.',
        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sensor_calibration']},indent=2))
    for row in rows:print(row)


if __name__=='__main__':main()
