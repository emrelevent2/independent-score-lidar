"""Versioned current local package; no third-party source, data or weights."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';PDF=OUT/'pdf'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    PDF.mkdir(exist_ok=True);files=set()
    def add(p):
        if p.is_file():files.add(p)
    def tree(folder,suffixes):
        for p in folder.rglob('*'):
            if p.suffix in suffixes and '__pycache__' not in p.parts:add(p)
    for pattern in ['*.md','*.json','*.m','requirements-lock.txt']:
        for p in ROOT.glob(pattern):add(p)
    tree(ROOT/'src',{'.py'})
    for p in (ROOT/'manuscript').glob('*'):
        if p.suffix in ['.tex','.pdf','.json'] and p.name not in ['methods_working.tex','methods_working.pdf','evidence_results.tex']:add(p)
    tree(ROOT/'manuscript'/'figures',{'.pdf','.png'})
    add(ROOT/'manuscript'/'qa'/'render_manifest.json')
    for folder in ['confirmation_v1','guided_confirmation_v1','localization_confirmation_v1','improvement_development_v1']:
        tree(ROOT/'data'/folder,{'.json','.npz','.mat'})
    for name in ['format_fixture.mat','format_fixture.rbin','download_manifest.json','LICENSE']:
        add(ROOT/'data'/'rt3d_measured'/name)
    for folder in ['confirmation_v1','guided_confirmation_v1','localization_confirmation_v1',
                   'improvement_development_v1','adaptive_development_v1','guided_likelihood_development_v1',
                   'profile_development_v1','measured_response_v1']:
        tree(ROOT/'results'/folder,{'.json','.csv','.npz','.mat'})
    for p in (ROOT/'results').glob('*'):
        if p.suffix in ['.json','.csv']:add(p)
    for folder in (ROOT/'tools').glob('rt3d_*v1_*'):
        for name in ['run.log','run_manifest.json']:add(folder/name)
    for p in (ROOT/'tools'/'rt3d_adapter_check').rglob('*'):
        if p.name in ['frame0_w0.ply','frame1_w0.ply','run_manifest.json','run.log']:add(p)
    for name in ['confirmation_caspi.log','guided_confirmation_caspi.log','current_checks_20261106.log']:
        add(ROOT/'logs'/name)
    for source,target in [('main.pdf','independent_photon_evidence.pdf'),('supplement.pdf','supplementary_methods.pdf'),('cover_letter.pdf','cover_letter_draft.pdf')]:
        dst=PDF/target;shutil.copyfile(ROOT/'manuscript'/source,dst);add(dst)
    render=json.loads((ROOT/'manuscript'/'qa'/'render_manifest.json').read_text())
    qa=dict(status='Research author-review draft; physical reference validation and isolated-target robustness remain outstanding.',
        version=2,pages={k:len(v['pages']) for k,v in render.items() if isinstance(v,dict)},
        abstract_words=render['abstract_word_count'],visual_review='All9main,9supplement and1cover pages inspected at full render resolution.',
        checks='46tests; current_checks_20261106.log',
        integrity_audit='results/localization_confirmation_v1/integrity_audit.json',
        current_analysis_sha256=sha(ROOT/'results'/'localization_confirmation_v1'/'analysis.json'),
        no_external_submission=True)
    qa_path=OUT/'CURRENT_QA_v2.json';qa_path.write_text(json.dumps(qa,indent=2));add(qa_path)
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    assert not any(x.endswith(('.exe','.dll','.pth','.pt','.whl')) or '.venv/' in x for x in manifest)
    archive=OUT/'independent_photon_evidence_review_bundle_v2.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,'caspi_study/'+p.relative_to(ROOT).as_posix())
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(dict(version=2,entries=manifest,
            note='Manifest excludes its own bytes. Third-party source, executable, weights and large real data excluded.'),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,v in manifest.items():assert hashlib.sha256(z.read('caspi_study/'+n)).hexdigest()==v['sha256']
    digest=sha(archive);(OUT/'bundle_v2_sha256.txt').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(path=str(archive),entries=len(manifest),bytes=archive.stat().st_size,sha256=digest),indent=2))


if __name__=='__main__':main()
