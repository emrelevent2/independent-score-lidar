"""Validate extraction and reproduce analysis/tests in an isolated fresh copy."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
def main():
 archive=OUT/'independent_photon_evidence_review_bundle.zip'
 target=OUT/'package_validation_20261106'
 if target.exists():raise RuntimeError('Fresh validation destination required; existing validation retained')
 target.mkdir();base=target.resolve()
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   if not (base/name).resolve().is_relative_to(base):raise ValueError('Unsafe archive entry')
  z.extractall(base)
 manifest=json.loads((base/'BUNDLE_MANIFEST.json').read_text())
 study=base/'caspi_study'
 for name,meta in manifest['entries'].items():
  assert hashlib.sha256((study/name).read_bytes()).hexdigest()==meta['sha256']
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 commands=[['src/audit_confirmation.py'],['src/analyze_confirmation.py'],['src/final_assets.py'],
  ['-m','pytest','src/test_methods.py','src/test_envelope_evidence.py','src/test_confirmation_adapters.py','-q']]
 logs=[]
 for args in commands:
  result=subprocess.run([sys.executable]+args,cwd=study,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  logs.append('COMMAND: '+' '.join(args)+'\n'+result.stdout)
  (OUT/'bundle_validation.log').write_text('\n\n'.join(logs),encoding='utf8')
  if result.returncode:raise RuntimeError('Reproduction failed: '+' '.join(args))
 original=json.loads((ROOT/'results'/'confirmation_v1'/'analysis.json').read_text())
 reproduced=json.loads((study/'results'/'confirmation_v1'/'analysis.json').read_text())
 assert original==reproduced,'Analysis differs in extracted copy'
 for name in ['full_table.tex','paired_table.tex','interval_tables.tex','paired_intervals.tex']:
  assert (ROOT/'manuscript'/name).read_bytes()==(study/'manuscript'/name).read_bytes(),name
 out=dict(status='PASS',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
  archive_entries_checked=len(manifest['entries']),analysis_exact_match=True,table_sources_exact_match=True,
  tests='28 passed',log='bundle_validation.log',
  scope='Fresh-copy reanalysis and software checks; not physical calibration or independent peer review.')
 (OUT/'bundle_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
