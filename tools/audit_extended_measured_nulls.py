"""Recount the extended measured false-alarm cohort from saved outputs."""
from pathlib import Path
import hashlib,json,sqlite3,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/extended_measured_nulls_v1';parent=ROOT/'results/calibrated_recovery_v1'
    p=json.loads((ROOT/'EXTENDED_MEASURED_NULL_PROTOCOL_20261107.json').read_text())
    assert hashlib.sha256((ROOT/'tools/extended_measured_nulls.py').read_bytes()).hexdigest()==p['source_sha256']
    if not (out/'analysis.json').exists():
        subprocess.run([sys.executable,str(ROOT/'tools/extended_measured_nulls.py'),'analyze'],check=True)
    result=json.loads((out/'analysis.json').read_text());base=json.loads((parent/'analysis.json').read_text())
    records=[]
    for path in [parent/'records.sqlite3',out/'records.sqlite3']:
        with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True) as db:
            records.extend(json.loads(r[0]) for r in db.execute('SELECT payload FROM records') if json.loads(r[0])['kind']=='public')
    assert len(records)==1024 and len({r['data_sha256'] for r in records})==1024
    assert sorted(r['index'] for r in records)==list(range(1024))
    for row in result['rows']:
        method=row['method'];threshold=base['thresholds']['measured_high'][method]
        assert sum(r['methods'][method]['applicable'] for r in records)==row['applicable']
        assert sum(r['methods'][method]['score']>threshold for r in records if r['methods'][method]['applicable'])==row['false_alarms']
    audit=dict(status='PASS',records=1024,additional_records=896,unique_source_hashes=1024,decisions_recounted=True,
        analysis_sha256=hashlib.sha256((out/'analysis.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
