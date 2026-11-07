"""Read-only integrity audit of the frozen sources, inputs and final analysis."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=ROOT/'CONFIRMATORY_FREEZE.json';freeze=json.loads(f.read_text())
 checks=[]
 assert sha(ROOT/'CONFIRMATORY_PROTOCOL.md')==freeze['protocol_sha256'];checks.append('Protocol hash')
 for name,digest in freeze['source_sha256'].items():assert sha(ROOT/'src'/name)==digest,name
 for name,digest in freeze['matlab_source_sha256'].items():assert sha(ROOT/name)==digest,name
 checks.append('14 frozen Python and two MATLAB source hashes')
 data=ROOT/'data'/'confirmation_v1';r=ROOT/'results'/'confirmation_v1'
 manifest=json.loads((data/'manifest.json').read_text());assert manifest['freeze_sha256']==sha(f)
 assert len(manifest['cases'])==1449
 for c in manifest['cases']:
  p=data/(c['id']+'.npz');assert sha(p)==c['sha256'],c['id']
  with np.load(p) as z:h=z['histogram']
  assert h.shape==(16,16,256) and np.all(h>=0) and np.all(h.sum(-1)<=1000)
 checks.append('All 1449 input hashes and first-event count invariants')
 a=json.loads((r/'analysis.json').read_text());assert a['caspi_cases_available']==579
 assert len(json.loads((r/'caspi_reports.json').read_text())['reports'])==579
 assert 'rt3d_100' not in a['methods'] and 'rt3d_authors_default' not in a['methods']
 assert len(a['excluded_runs'])==8
 checks.append('All 579 CASPI reports; eight failed RT3D runs excluded')
 for method,recs in a['per_case'].items():
  ids=[c['id'] for c in recs];assert len(ids)==len(set(ids))
  null=[x for x in recs if not x['positive']];pos=[x for x in recs if x['positive']]
  ss=a['methods'][method]['summaries']
  assert len(null)==ss['null']['n'] and len(pos)==ss['positive']['n']
  assert sum(x['declared'] for x in null)==ss['null']['declared']['count']
  assert sum(x['localized'] for x in pos)==ss['positive']['localized']['count']
  assert all(not x['localized'] or x['declared'] for x in pos)
 checks.append('Analysis denominators, duplicate IDs and count consistency')
 evidence=[json.loads((r/(x['id']+'.json')).read_text())['reports']['proposed'] for x in a['per_case']['proposed']]
 assert len(evidence)==1050
 assert all(e['evidence']['upper']>=e['evidence']['feasible_lower'] for e in evidence)
 assert all(e['evidence']['boxes_evaluated']==256 for e in evidence)
 assert all(not e['declared'] or e['score']>=e['evidence']['log_threshold_upper'] for e in evidence)
 checks.append('Evidence gaps, budget and certified decision consistency')
 out=dict(status='PASS',checks=checks,freeze_sha256=sha(f),analysis_sha256=sha(r/'analysis.json'),
  source_sha256=sha(Path(__file__)),scope='Implementation and records audit; not external peer review or physical sensor validation.')
 (r/'integrity_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
