"""Recount every saved decision and check the M2/M3 extension inputs."""
from pathlib import Path
from collections import Counter
import json, sys, sqlite3, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import calibrated_recovery_study as study
import calibrated_recovery as model


def main():
    protocol=json.loads(study.P.read_text());out=study.OUT
    for path,digest in protocol['source_hashes'].items():assert study.sha(ROOT/path.replace('\\','/'))==digest
    for name,value in protocol['sensor_calibration'].items():
        if value is None:protocol['sensor_calibration'][name]=json.loads((out/('sensor_'+name+'.json')).read_text())
    with sqlite3.connect(f'file:{(out/"records.sqlite3").as_posix()}?mode=ro',uri=True) as db:
        records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY id')]
    assert len(records)==6302,len(records)
    assert len({r['id'] for r in records})==6302
    counts=Counter((r['name'],r['kind']) for r in records)
    seen=set();duplicates=[]
    for record in records:
        if record['data_sha256'] in seen:duplicates.append(record['id'])
        seen.add(record['data_sha256'])
        for method in record['methods'].values():
            if method['applicable']:
                assert np.isfinite(method['score']) and method['score']>=0
    assert not duplicates,duplicates
    bank_checks=[]
    for name in protocol['settings']:
        assert counts[(name,'threshold')]==499
        assert counts[(name,'evaluate')]==(320 if name=='measured_high' else 96)
        sensor=protocol['sensor_calibration'][name];assert sensor is not None
        assert sensor.get('calibration_data_sha256') not in seen
        path=out/('bank_'+name+'.npz')
        with np.load(path) as z:matrix=z['information']
        assert np.all(np.isfinite(matrix))
        minimum=float(np.linalg.eigvalsh(matrix).min())
        symmetry=float(np.max(abs(matrix-matrix.swapaxes(-1,-2))))
        assert minimum>=-1e-7 and symmetry<1e-10
        bank_checks.append(dict(name=name,sha256=study.sha(path),minimum_eigenvalue=minimum,max_asymmetry=symmetry))
    assert counts[('measured_high','public')]==128
    # Match against the original frozen acquisitions without requiring libm-
    # identical floating-point simulation on every replay platform.
    with sqlite3.connect(f'file:{(ROOT/"results/revision_sensitivity_v1/robust.sqlite3").as_posix()}?mode=ro',uri=True) as db:
        old=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
    original={}
    for name in protocol['settings']:
        group=sorted((r for r in old if r['case']['condition']['name']==name),key=lambda r:r['case']['index'])
        for i,r in enumerate(group):original[(name,i)]=r
    reconstructed=matched=0
    for record in records:
        if record['kind']=='evaluate' and record['name']!='measured_high':
            parent=original[(record['name'],record['index'])]
            assert parent['data_sha256']==record['data_sha256']
            assert parent['case']['positive']==record['positive'] and parent['truth']==record['truth']
            matched+=1
            if '--regenerate-paths' in sys.argv:
                observed,positive,truth,theta=study.acquisition((record['name'],'evaluate',record['index']),protocol)
                assert hashlib.sha256(observed.tobytes()).hexdigest()==record['data_sha256']
                reconstructed+=1
    assert matched==864
    analysis=json.loads((out/'analysis.json').read_text())
    if 'postprocessor_sha256' in analysis:
        assert study.sha(ROOT/'tools/analyze_calibrated_recovery_complete.py')==analysis['postprocessor_sha256']
    for row in analysis['rows']:
        cohort=[r for r in records if r['name']==row['name'] and r['kind']==row['kind'] and (row['ratio'] is None or not r['positive'] or r['truth']['ratio']==row['ratio'])]
        train=[r['methods'][row['method']]['score'] if r['methods'][row['method']]['applicable'] else -np.inf for r in records if r['name']==row['name'] and r['kind']=='threshold']
        declare=correct=wrong=false=unavailable=0;errors=[]
        for record in cohort:
            value=record['methods'][row['method']]
            if not value['applicable']:unavailable+=1;continue
            p=(1+sum(x>=value['score'] for x in train))/500
            decision=p<=.01
            assert decision==(value['score']>row['threshold'])
            if not record['positive']:false+=int(decision);continue
            declare+=int(decision)
            if decision:
                dist=lambda a,b:abs((a-b+32)%64-32)
                a=dist(value['primary'],record['truth']['primary']);b=dist(value['delay'],record['truth']['secondary'])
                correct+=int(a<=2 and b<=2);wrong+=int(a>2 or b>2);errors.append(b)
        assert (declare,correct,wrong,false,unavailable)==tuple(row[k] for k in ['declared','correct','wrong','false_alarms','unavailable'])
        assert declare==correct+wrong
        if errors:assert abs(np.sqrt(np.mean(np.square(errors)))-row['conditional_secondary_rmse'])<1e-12
    warnings=Counter((r['name'],m) for r in records for m,v in r['methods'].items() if v.get('optimizer_success') is False)
    result=dict(status='PASS',records=len(records),all_saved_decisions_recounted=True,
        independent_rank_decisions_recounted=True,distinct_query_paths=len(seen),
        original_mismatch_hashes_matched=matched,physical_acquisitions_reconstructed=reconstructed,bank_checks=bank_checks,
        kernel_checks=json.loads((out/'kernel_checks.json').read_text())['status'],
        optimizer_warning_counts={f'{a}/{b}':n for (a,b),n in warnings.items()},
        protocol_sha256=study.sha(study.P),analysis_sha256=study.sha(out/'analysis.json'),
        source_hashes=protocol['source_hashes'],external_journal_submission=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='bank_checks'},indent=2))


if __name__=='__main__':main()
