"""Saved-output recount of the conditional measured-path injection study."""
from pathlib import Path
import hashlib,json,sqlite3,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/semisynthetic_recovery_v2';parent=ROOT/'results/calibrated_recovery_v1'
    p=json.loads((ROOT/'SEMISYNTHETIC_TERMINAL_PROTOCOL_20261107.json').read_text())
    if not (out/'analysis.json').exists():
        subprocess.run([sys.executable,str(ROOT/'tools/semisynthetic_recovery_tail.py'),'analyze'],check=True)
    assert hashlib.sha256((ROOT/'tools/semisynthetic_recovery_study.py').read_bytes()).hexdigest()==p['source_sha256']
    assert hashlib.sha256((ROOT/'tools/semisynthetic_recovery_tail.py').read_bytes()).hexdigest()==p['terminal_source_sha256']
    with sqlite3.connect(f'file:{(out/"records.sqlite3").as_posix()}?mode=ro',uri=True) as db:records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
    with sqlite3.connect(f'file:{(parent/"records.sqlite3").as_posix()}?mode=ro',uri=True) as db:original={json.loads(r[0])['index']:json.loads(r[0]) for r in db.execute('SELECT payload FROM records') if json.loads(r[0])['kind']=='public'}
    analysis=json.loads((out/'analysis.json').read_text());base=json.loads((parent/'analysis.json').read_text())
    assert len(records)==192 and len({r['index'] for r in records})==64
    assert len({r['counterfactual_data_sha256'] for r in records})==192
    for r in records:
        assert r['source_data_sha256']==original[r['index']]['data_sha256']
        assert r['source_completion']['zero_injection_roundtrip']
        assert r['source_completion']['terminal_completion']
    for row in analysis['rows']:
        threshold=base['thresholds']['measured_high'][row['method']];declared=correct=wrong=0;errors=[]
        for r in records:
            if r['ratio']!=row['ratio']:continue
            v=r['methods'][row['method']]
            if v['score']>threshold:
                declared+=1;error=abs((v['delay']-r['target_delay']+32)%64-32)
                correct+=int(error<=2);wrong+=int(error>2);errors.append(error)
        assert (declared,correct,wrong)==(row['declared'],row['correct_secondary'],row['wrong_secondary'])
        if errors:assert abs(np.sqrt(np.mean(np.square(errors)))-row['conditional_secondary_rmse'])<1e-12
    report=dict(status='PASS',records=192,source_blocks=64,zero_injection_roundtrips=64,
        decisions_recounted=True,source_block_hashes_matched=True,
        maximum_quadrature_difference=analysis['quadrature_max_absolute_difference'],
        analysis_sha256=hashlib.sha256((out/'analysis.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
