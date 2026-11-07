"""Package the final article and saved-result replay, checking every ZIP member."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'manuscript/ded_style_article_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    qa=json.loads((ROOT/'output/revised_article_qa.json').read_text());assert qa['status']=='PASS'
    assert qa['pdf_sha256']==sha(ROOT/'output/pdf/weak_return_two_recovery_revised.pdf')
    data=json.loads((D/'article_data.json').read_text());assert data['status']=='COMPLETE'
    files=set(D.glob('*.tex'))|set(D.glob('*.json'))|set((D/'figures').glob('*.pdf'))|set((D/'figures').glob('*.png'))
    files.update(ROOT/name for name in data['input_hashes'])
    files.update((ROOT/'src').glob('*.py'))
    files.update((ROOT/'tools').glob('revision_*.py'))
    files.update((ROOT/'tools').glob('audit_revision*.py'))
    files.update(ROOT/'tools'/name for name in ['build_revision_article.py','build_ded_style_article_assets.py','qa_revision_article.py','test_revision_simulation.py','prepare_revision_fonts.py','package_revision_article.py','validate_revision_source.py','build_measurement_assets.py','measurement_recovery_model.py','measurement_recovery_study.py','measurement_recovery_predictive.py','audit_measurement_recovery.py','audit_measured_results.py','measured_cluster_extension.py','audit_measured_cluster.py'])
    for pattern in ['*PROTOCOL*.json','*ADDENDUM*.json','*DESIGN*.json']:files.update(ROOT.glob(pattern))
    for folder in ['null_information_bank_v1','revision_censored_bank_v1','revision_censored_bank_common_v1','revision_exposure_bank_v1','revision_sensitivity_v1','localization_postfailure_reference_v1','revision_safe_robustness_v1']:
        directory=ROOT/'results'/folder
        files.update(directory.glob('*.json'));files.update(directory.glob('*.npz'))
    files.update(ROOT/'results'/name for name in ['revision_censored_bank_audit.json','revision_simulation_tests.json','localization_test_runtime_v1.json'])
    files.update(ROOT/'output'/name for name in ['revised_article_qa.json'])
    # Keep the frozen first-bank training summaries used to reconstruct exposure.
    files.update((ROOT/'results/revision_censored_bank_v1/cells').glob('*.json'))
    files.add(ROOT/'manuscript/author_metadata.json')
    files.add(ROOT/'REPRODUCIBILITY_NOTES_20261107.txt')
    assert all(p.exists() for p in files)
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    readme='''REVISED TWO-RECOVERY ARTICLE

Main source: manuscript/ded_style_article_v2/weak_return_two_recovery_revised.tex
The article has no appendices. All displayed comparisons use completed results.

REBUILD FIGURES AND TABLES FROM SAVED RESULTS
Install numpy 1.26.4, scipy 1.13.1 and matplotlib 3.8.4. Times New Roman must be
installed on the build machine. Run:
  python tools/build_revision_article.py --require-complete
  python tools/prepare_revision_fonts.py
Then compile the master TeX with Tectonic (XeTeX). Local font files are prepared
from the machine's installed fonts and are not part of this archive.

The results databases, protocols and source hashes support saved-output replay.
REPRODUCIBILITY_NOTES_20261107.txt contains numerical audit details removed from
the journal text. The measured recovery study uses original public MAT inputs,
not included here. Obtain them through the HighFluxSPL link in the article;
MEASURED_RECOVERY_PROTOCOL_20261107.json records their exact hashes. Saved
measured fits and predictions are included for rebuilding figures and tables.
The experiment runners and scientific sources are included for inspection.
Original simulation execution additionally uses the frozen development
environment and its upstream validation artifacts. The article does not use
measured two-surface accuracy results. Public timestamp sources are cited in
the manuscript.

SOURCE MAP
tools/build_revision_article.py: figures, tables and result text from databases.
tools/revision_censored_bank.py and *_common.py: matched information-bank controls.
tools/revision_exposure_bank.py: exposure interpolation with exact rate division.
tools/revision_sensitivity.py: fixed power and sensor-model sensitivity study.
src/localization_postfailure_full_joint_diagnosis.py: complete full-LR reference.
tools/qa_revision_article.py: counts, provenance, fonts and PDF consistency.
output/revised_article_qa.json: final visual and numerical verification.

MANIFEST.json contains SHA-256 digests for all supplied file members.
'''
    path=ROOT/'output/weak_return_two_recovery_revised_sources.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('README_REBUILD.txt',readme)
        z.writestr('requirements-figures.txt','numpy==1.26.4\nscipy==1.13.1\nmatplotlib==3.8.4\n')
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
        assert not any('/fonts/' in name for name in z.namelist())
    result=dict(status='PASS',archive=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,members=len(manifest)+3,archive_sha256=sha(path),all_member_hashes_checked=True,all_CRCs_checked=True,pdf_sha256=qa['pdf_sha256'],package_source_sha256=sha(__file__))
    (ROOT/'output/revised_article_package_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
