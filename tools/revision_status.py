"""Read-only committed counts; never computes interim efficacy summaries."""
from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3
ROOT=Path(__file__).resolve().parents[1]
items={
 'full_likelihood_reference':('localization_postfailure_reference_v1/equal_information_joint_reference_v2.sqlite3',640),
 'power_sweep':('revision_sensitivity_v1/power.sqlite3',480),
 'model_mismatch':('revision_sensitivity_v1/robust.sqlite3',864),
 'censored_information_bank_original':('revision_censored_bank_v1/records.sqlite3',640),
 'censored_information_bank_common':('revision_censored_bank_common_v1/records.sqlite3',640),
 'censored_exposure_bank_both':('revision_exposure_bank_v1/records.sqlite3',1280),
 'original_censored_model_mismatch':('revision_safe_robustness_v1/records.sqlite3',864)}
out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='ACTIVE_REVISION',studies={},audits={})
for name,(file,n) in items.items():
    path=ROOT/'results'/file
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True) as db:count=db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    out['studies'][name]=dict(committed=count,expected=n,complete=count==n)
for file in ['localization_postfailure_reference_v1/postfailure_full_joint_integrity.json','localization_postfailure_reference_v1/postfailure_full_joint_failure_audit.json','revision_sensitivity_v1/power_audit.json','revision_sensitivity_v1/robust_audit.json','revision_censored_bank_audit.json','revision_exposure_bank_v1/audit.json','revision_simulation_tests.json','revision_safe_robustness_v1/audit.json']:
    path=ROOT/'results'/file;out['audits'][file]=json.loads(path.read_text())['status'] if path.exists() else 'RUNNING'
(ROOT/'output/revision_progress_20261107.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
