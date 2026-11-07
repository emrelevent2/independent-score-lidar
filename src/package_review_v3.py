"""Version3 local evidence package, excluding unlicensed upstream materials."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';PDF=OUT/'pdf';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=OUT/'independent_photon_evidence_review_bundle_v3.zip'
    assert not archive.exists(),'Preserve an existing versioned package; use a new version'
    files=set();PDF.mkdir(exist_ok=True)
    def add(path):
        if path.is_file():files.add(path)
    def tree(folder,suffixes):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:files.add(path)
    for pattern in ['*.md','*.json','*.m','requirements-lock.txt','STB_DATA_NOTICE.txt']:
        for path in ROOT.glob(pattern):add(path)
    tree(ROOT/'src',{'.py'})
    for path in (ROOT/'manuscript').glob('*'):
        if path.suffix in ['.tex','.pdf','.json'] and path.name not in ['methods_working.tex','methods_working.pdf','evidence_results.tex']:add(path)
    tree(ROOT/'manuscript/figures',{'.png','.pdf','.json'});tree(ROOT/'manuscript/archive_v2',{'.tex'});add(ROOT/'manuscript/qa/render_manifest.json')
    for folder in ['confirmation_v1','guided_confirmation_v1','localization_confirmation_v1','improvement_development_v1','routing_confirmation_v1','guarded_confirmation_v1']:
        tree(ROOT/'data'/folder,{'.npz','.json','.mat'})
    for folder in ['confirmation_v1','guided_confirmation_v1','localization_confirmation_v1','improvement_development_v1',
        'adaptive_development_v1','guided_likelihood_development_v1','profile_development_v1','measured_response_v1',
        'support_development_v1','disagreement_development_v1','routing_confirmation_v1','guarded_development_v1','guarded_confirmation_v1',
        'real_poisson_development_v1','real_poisson_development_v2','agreement_development_v1','anchored_development_v1','real_multiframe_test_v1']:
        tree(ROOT/'results'/folder,{'.json','.csv','.npz','.mat'})
    for path in (ROOT/'results').glob('*'):
        if path.suffix in ['.json','.csv']:add(path)
    # Real raw tensors are recoverable by a pinned member-fetch script. Only
    # small required labels/profile and exact metadata are included offline.
    for folder in ['real_poisson_development_v1','real_multiframe_test_v1','stb_real_test_selection','stb_real_test_selection_more']:
        tree(ROOT/'data'/folder,{'.json'})
    add(ROOT/'data/real_poisson_development_v1/empirical_profile.npz')
    selection=json.loads((ROOT/'data/stb_real_test_selection_more/selection.json').read_text())
    for item in selection['selected']:add(ROOT/item['label_file_relative'])
    add(ROOT/'data/stb_metadata/README.md')
    for name in ['format_fixture.mat','format_fixture.rbin','download_manifest.json','LICENSE']:add(ROOT/'data/rt3d_measured'/name)
    for folder in (ROOT/'tools').glob('rt3d_*v1_*'):
        for name in ['run.log','run_manifest.json']:add(folder/name)
    for path in (ROOT/'tools/rt3d_adapter_check').rglob('*'):
        if path.name in ['frame0_w0.ply','frame1_w0.ply','run_manifest.json','run.log']:add(path)
    for name in ['confirmation_caspi.log','guided_confirmation_caspi.log','current_checks_20261106.log']:add(ROOT/'logs'/name)
    for source,target in [('main.pdf','independent_photon_evidence_v3.pdf'),('supplement.pdf','supplementary_methods_v3.pdf'),('cover_letter.pdf','cover_letter_v3.pdf')]:
        destination=PDF/target;shutil.copyfile(ROOT/'manuscript'/source,destination);add(destination)
    for folder in ['guarded_confirmation_v1','real_multiframe_test_v1']:assert json.loads((ROOT/'results'/folder/'integrity.json').read_text())['status']=='PASS'
    render=json.loads((ROOT/'manuscript/qa/render_manifest.json').read_text())
    qa=dict(version=3,status='Author-review research manuscript; scoped computational/simulation claim and explicit negative real transfer.',
        pages={k:len(v['pages']) for k,v in render.items() if isinstance(v,dict)},abstract_words=render['abstract_word_count'],
        visual_review='All9main,13supplement and1cover pages inspected at full render resolution; changed pages rechecked.',tests='64passed',
        guarded_analysis_sha256=sha(ROOT/'results/guarded_confirmation_v1/analysis.json'),real_analysis_sha256=sha(ROOT/'results/real_multiframe_test_v1/analysis.json'),
        no_external_submission=True,no_physical_improvement_claim=True)
    path=OUT/'CURRENT_QA_v3.json';path.write_text(json.dumps(qa,indent=2));add(path)
    manifest={path.relative_to(ROOT).as_posix():dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(files)}
    assert not any(name.endswith(('.exe','.dll','.pth','.pt','.whl')) or '.venv/' in name for name in manifest)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):z.write(path,'caspi_study/'+path.relative_to(ROOT).as_posix())
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(dict(version=3,entries=manifest,notes=[
            'Manifest excludes itself. No upstream source/executable/weights or large real tensors.',
            'Small STB-derived numeric labels/profile/coordinates/endpoints retain dataset CC BY-NC-SA4.0 attribution; see STB_DATA_NOTICE.txt.',
            'Recorded-output real analysis is offline; full raw-count audit uses pinned member recovery.']),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,meta in manifest.items():assert hashlib.sha256(z.read('caspi_study/'+name)).hexdigest()==meta['sha256']
    digest=sha(archive);(OUT/'bundle_v3_sha256.txt').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(path=str(archive),entries=len(manifest),bytes=archive.stat().st_size,sha256=digest),indent=2),flush=True)


if __name__=='__main__':main()
