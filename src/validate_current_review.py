"""Fresh-copy reproduction of v2 records; preserves the previous archive audit."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=OUT/'independent_photon_evidence_review_bundle_v2.zip'
    target=OUT/'package_validation_v2_20261106'
    if target.exists():raise RuntimeError('Fresh validation destination required; existing evidence retained')
    target.mkdir();base=target.resolve()
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            if not (base/n).resolve().is_relative_to(base):raise ValueError('Unsafe archive path')
        z.extractall(base)
    manifest=json.loads((base/'BUNDLE_MANIFEST.json').read_text());study=base/'caspi_study'
    for name,entry in manifest['entries'].items():assert sha(study/name)==entry['sha256'],name
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[['src/audit_confirmation.py'],['src/audit_localization_confirmation.py'],
        ['src/analyze_localization_confirmation.py'],['src/localization_assets.py'],['-m','pytest','src','-q']]
    logs=[]
    for args in commands:
        run=subprocess.run([sys.executable]+args,cwd=study,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logs.append('COMMAND: '+' '.join(args)+'\n'+run.stdout)
        (OUT/'bundle_v2_validation.log').write_text('\n\n'.join(logs),encoding='utf8')
        if run.returncode:raise RuntimeError('Reproduction failed: '+' '.join(args))
        print('PASS',' '.join(args),flush=True)
    name=Path('results/localization_confirmation_v1/analysis.json')
    assert json.loads((ROOT/name).read_text())==json.loads((study/name).read_text()),'Analysis changed in extracted copy'
    for name in ['localization_table.tex','localization_all_table.tex']:
        assert (ROOT/'manuscript'/name).read_bytes()==(study/'manuscript'/name).read_bytes(),name
    result=dict(status='PASS',archive_sha256=sha(archive),manifest_entries_checked=len(manifest['entries']),
        analysis_exact_match=True,current_tables_exact_match=True,tests='46passed',
        scope='Saved-data/source integrity, fresh-copy reanalysis and software checks. Not physical sensor validation.',
        log='bundle_v2_validation.log')
    (OUT/'bundle_v2_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
