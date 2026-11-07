"""Rebuild saved-result figures and PDF from an isolated archive extraction."""
from pathlib import Path
import os,json,zipfile,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    qa=json.loads((ROOT/'output/revised_article_qa.json').read_text());assert qa['status']=='PASS'
    archive=ROOT/'output/weak_return_two_recovery_revised_sources.zip'
    destination=ROOT/'output/revised_article_source_validation_20261107';suffix=1
    while destination.exists():
        suffix+=1;destination=ROOT/'output'/f'revised_article_source_validation_20261107_{suffix}'
    assert destination.resolve().is_relative_to((ROOT/'output').resolve())
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in z.namelist():assert (destination/member).resolve().is_relative_to(destination.resolve())
        z.extractall(destination)
    manifest=json.loads((destination/'MANIFEST.json').read_text())
    for name,digest in manifest.items():assert sha(destination/name)==digest
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=ROOT/'.venv_ded/Scripts/python.exe'
    def run(arguments):
        r=subprocess.run([str(a) for a in arguments],cwd=destination,env=env,capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
        return r.stdout+r.stderr
    run([python,'tools/build_revision_article.py','--require-complete'])
    source=ROOT/'manuscript/ded_style_article_v2';rebuilt=destination/'manuscript/ded_style_article_v2'
    generated=[*source.glob('table_*.tex'),*[source/n for n in ['normalization_results.tex','glrt_results.tex','power_results.tex','robustness_results.tex','article_data.json','power_data.json','robustness_data.json','measured_results.tex','measured_data.json']],*(source/'figures').glob('*.pdf'),*(source/'figures').glob('*.png')]
    for original in generated:assert sha(original)==sha(rebuilt/original.relative_to(source)),original
    run([python,'tools/prepare_revision_fonts.py'])
    out=destination/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    run([ROOT/'tools/tectonic.exe','--keep-logs','--outdir',out,rebuilt/'weak_return_two_recovery_revised.tex'])
    images=rebuilt/'qa';images.mkdir()
    poppler=Path('C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
    run([poppler,'-r','110','-png',out/'weak_return_two_recovery_revised.pdf',images/'page'])
    expected=sorted((source/'qa').glob('page-*.png'));actual=sorted(images.glob('page-*.png'))
    assert len(expected)==len(actual)==qa['pages']
    for a,b in zip(expected,actual):assert a.name==b.name and sha(a)==sha(b),(a,b)
    result=dict(status='PASS',archive_sha256=sha(archive),source_members_checked=len(manifest),generated_assets_identical=len(generated),all_final_pages_pixel_identical=len(expected),validation_directory=destination.relative_to(ROOT).as_posix(),isolated_pdf_sha256=sha(out/'weak_return_two_recovery_revised.pdf'),primary_pdf_sha256=qa['pdf_sha256'],validator_source_sha256=sha(__file__))
    (ROOT/'output/revised_article_isolated_replay_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
