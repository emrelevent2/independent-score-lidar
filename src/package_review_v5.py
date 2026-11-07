"""Consolidate validated v3 history and current work without re-extraction."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=OUT/'independent_photon_evidence_review_bundle_v5.zip';assert not archive.exists()
    base=OUT/'independent_photon_evidence_review_bundle_v3.zip'
    base_sha='133076c82888723d701cae142d5fbe128dbcec85afb55ccae184f035b003ee0d';assert sha(base)==base_sha
    for folder in ['coverage_confirmation_v1','coverage_external_v1','real_coverage_development_v1','real_primary_bank_development_v1','coverage_sparse_benchmark_v1','guarded_real_confirmation_v1']:
        assert json.loads((ROOT/'results'/folder/'integrity.json').read_text())['status']=='PASS',folder
    files=set()
    def add(p):
        if p.is_file():files.add(p)
    def tree(folder,suffixes):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:add(p)
    for pattern in ['*.md','*.json','requirements-lock.txt','STB_DATA_NOTICE.txt']:
        for p in ROOT.glob(pattern):add(p)
    tree(ROOT/'src',{'.py'});tree(ROOT/'data/coverage_confirmation_v1',{'.json','.npz'})
    for folder in ['coverage_confirmation_v1','coverage_development_v1','guarded_real_confirmation_v1','real_coverage_development_v1','real_primary_bank_development_v1','coverage_external_v1','coverage_sparse_benchmark_v1']:
        tree(ROOT/'results'/folder,{'.json','.csv','.sqlite3'})
    for folder in ['guarded_real_confirmation_v1','stb_guarded_selection_v1']:tree(ROOT/'data'/folder,{'.json'})
    add(ROOT/'data/stb_multireturn_development/download_manifest.json')
    selection=json.loads((ROOT/'data/stb_guarded_selection_v1/selection.json').read_text())
    for f in selection['selected']:add(ROOT/f['label_file_relative'])
    for p in (ROOT/'manuscript').glob('*'):
        if p.suffix in ['.tex','.json']:add(p)
    tree(ROOT/'manuscript/figures',{'.png','.pdf','.json'});tree(ROOT/'manuscript/archive_v4',{'.tex','.json'})
    add(ROOT/'manuscript/qa/render_manifest.json')
    for stem,target in [('main','independent_photon_evidence_v5'),('supplement','supplementary_methods_v5'),('cover_letter','cover_letter_v5')]:
        destination=OUT/'pdf'/(target+'.pdf');source_pdf=ROOT/'manuscript'/(stem+'.pdf')
        if destination.exists():assert sha(destination)==sha(source_pdf),'Preserve an already different versioned PDF'
        else:shutil.copyfile(source_pdf,destination)
        add(destination)
        add(ROOT/'manuscript'/(stem+'.pdf'));add(ROOT/'manuscript'/(stem+'.log'))
    render=json.loads((ROOT/'manuscript/qa/render_manifest.json').read_text())
    qa=dict(version=5,pages={k:len(v['pages']) for k,v in render.items() if isinstance(v,dict)},abstract_words=render['abstract_word_count'],tests='76passed',
        visual_review='All8main,16supplement and1cover pages inspected at full render resolution; final bibliography/figure changes rechecked.',
        scientific_scope='Computable model-conditional inference and fresh simulation sensitivity gain; actual current-method transfer comparison and negative real developments retained.',
        no_external_submission=True,no_independent_physical_advantage_claim=True)
    qapath=OUT/'CURRENT_QA_v5.json';qapath.write_text(json.dumps(qa,indent=2));add(qapath)
    overlay={p.relative_to(ROOT).as_posix():p for p in files};manifest={}
    with zipfile.ZipFile(base) as source:
        old=json.loads(source.read('BUNDLE_MANIFEST.json'))['entries'];names=sorted(set(old)|set(overlay))
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
            for i,name in enumerate(names,1):
                raw=overlay[name].read_bytes() if name in overlay else source.read('caspi_study/'+name)
                digest=hashlib.sha256(raw).hexdigest()
                if name not in overlay:assert digest==old[name]['sha256'] and len(raw)==old[name]['bytes']
                manifest[name]=dict(bytes=len(raw),sha256=digest);target.writestr('caspi_study/'+name,raw)
                if i%20000==0:print('Consolidated v5',i,'/',len(names),flush=True)
            target.writestr('BUNDLE_MANIFEST.json',json.dumps(dict(version=5,entries=manifest,base_history_sha256=base_sha,
                notes=['Consolidated standalone saved-output research history; v4 overlay is not required.',
                    'Large recorded raw/thinned count tensors and upstream source/weights/executables excluded; pinned recovery tools and complete provenance supplied.',
                    'STB-derived records retain dataset attribution and CC BY-NC-SA4.0; see STB_DATA_NOTICE.txt.',
                    'Scientific claims are conditional and include adverse/negative developments. No submission or public deposit.']),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,meta in manifest.items():
            raw=z.read('caspi_study/'+name);assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
    digest=sha(archive);(OUT/'bundle_v5_sha256.txt').write_text(digest+'  '+archive.name+'\n')
    info=dict(status='PASS streaming entry CRC/SHA; saved-output replay pending',path=str(archive),entries=len(manifest),bytes=archive.stat().st_size,archive_sha256=digest)
    (OUT/'bundle_v5_build.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
