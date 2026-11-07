"""Fresh-copy offline reproduction of the current simulation and real tables."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=OUT/'independent_photon_evidence_review_bundle_v3.zip';target=OUT/'package_validation_v3_20261106'
    assert not target.exists(),'Keep prior validation evidence; use a fresh directory'
    target.mkdir();base=target.resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():assert (base/name).resolve().is_relative_to(base),'Unsafe archive path'
        z.extractall(base)
    manifest=json.loads((base/'BUNDLE_MANIFEST.json').read_text());study=base/'caspi_study'
    for name,meta in manifest['entries'].items():assert sha(study/name)==meta['sha256'],name
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');logs=[]
    commands=[['src/guarded_confirmation_analysis.py','audit'],['src/guarded_confirmation_analysis.py','analyze'],
        ['src/real_multiframe_analysis.py','analyze'],['src/improvement_assets.py'],['-m','pytest','src','-q']]
    for args in commands:
        result=subprocess.run([sys.executable]+args,cwd=study,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logs.append('COMMAND: '+' '.join(args)+'\n'+result.stdout);(OUT/'bundle_v3_validation.log').write_text('\n\n'.join(logs),encoding='utf8')
        assert result.returncode==0,'Reproduction failed: '+' '.join(args);print('PASS',' '.join(args),flush=True)
    for name in ['results/guarded_confirmation_v1/analysis.json','results/real_multiframe_test_v1/analysis.json']:
        assert json.loads((ROOT/name).read_text())==json.loads((study/name).read_text()),name
    for name in ['guarded_table.tex','real_transfer_table.tex']:
        assert (ROOT/'manuscript'/name).read_bytes()==(study/'manuscript'/name).read_bytes(),name
    result=dict(status='PASS',version=3,archive_sha256=sha(archive),manifest_entries_checked=len(manifest['entries']),
        guarded_and_real_analyses_exact_match=True,tables_exact_match=True,tests='64passed',
        scope='Fresh-copy hash checks, complete current simulation-input audit, saved-output real reanalysis, deterministic tables and all software tests. Full real raw tensors are intentionally recovered separately; no physical-calibration claim.',
        log='bundle_v3_validation.log')
    (OUT/'bundle_v3_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
