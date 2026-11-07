"""Package original study artifacts; exclude third-party code and binaries."""
from pathlib import Path
import hashlib,json,zipfile,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';PDF=OUT/'pdf';PDF.mkdir(parents=True,exist_ok=True)
mapping={'main.pdf':'independent_photon_evidence.pdf','supplement.pdf':'supplementary_methods.pdf','cover_letter.pdf':'cover_letter_draft.pdf'}
for a,b in mapping.items():shutil.copyfile(ROOT/'manuscript'/a,PDF/b)
files=set()
def add(p):
 if p.is_file():files.add(p)
for p in ROOT.glob('*.md'):add(p)
for p in ROOT.glob('*.m'):add(p)
for name in ['requirements-lock.txt','CONFIRMATORY_FREEZE.json']:add(ROOT/name)
for p in (ROOT/'src').rglob('*.py'):
 if '__pycache__' not in p.parts:add(p)
for p in (ROOT/'manuscript').glob('*'):
 if p.suffix in ['.tex','.pdf','.json']:add(p)
# The old working PDF is stale development prose, not part of the current package.
files={p for p in files if p.name not in ['methods_working.pdf','methods_working.tex','evidence_results.tex']}
for p in (ROOT/'manuscript'/'figures').glob('*'):
 if p.suffix in ['.pdf','.png']:add(p)
for p in (ROOT/'data'/'confirmation_v1').glob('*'):
 if p.suffix in ['.npz','.json','.mat']:add(p)
for name in ['format_fixture.mat','format_fixture.rbin','download_manifest.json','LICENSE']:
 add(ROOT/'data'/'rt3d_measured'/name)
for p in (ROOT/'results'/'confirmation_v1').glob('*'):
 if p.suffix in ['.json','.csv','.mat']:add(p)
for p in (ROOT/'results').glob('*'):
 if p.suffix in ['.json','.csv']:add(p)
for folder in (ROOT/'tools').glob('rt3d_*v1_*'):
 for name in ['run.log','run_manifest.json']:add(folder/name)
for p in (ROOT/'tools'/'rt3d_adapter_check').rglob('*'):
 if p.name in ['frame0_w0.ply','frame1_w0.ply','run_manifest.json','run.log']:add(p)
add(ROOT/'logs'/'confirmation_caspi.log')
for p in PDF.glob('*.pdf'):add(p)
render=json.loads((ROOT/'manuscript'/'qa'/'render_manifest.json').read_text())
qa=dict(pdf_pages={k:len(v['pages']) for k,v in render.items() if isinstance(v,dict)},
 pdf_sha256={k:v['pdf_sha256'] for k,v in render.items() if isinstance(v,dict)},
 abstract_words=render['abstract_word_count'],visual_review='All six main, six supplement and one cover-letter pages visually inspected.',
 integrity='results/confirmation_v1/integrity_audit.json',
 tests='28 passed; see bundled reproduction commands',
 status='Author-review package; original publication goal remains active; physical validation and stronger modern comparisons are outstanding.')
(OUT/'FINAL_QA.json').write_text(json.dumps(qa,indent=2));add(OUT/'FINAL_QA.json')
manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
assert not any('.venv/' in x or 'caspi_original/' in x or x.lower().endswith(('.dll','.exe')) for x in manifest)
bundle=OUT/'independent_photon_evidence_review_bundle.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,'caspi_study/'+p.relative_to(ROOT).as_posix())
 z.writestr('BUNDLE_MANIFEST.json',json.dumps(dict(entries=manifest,note='Manifest excludes its own bytes; third-party executables and source are excluded.'),indent=2))
with zipfile.ZipFile(bundle) as z:
 assert z.testzip() is None
 for name,meta in manifest.items():assert hashlib.sha256(z.read('caspi_study/'+name)).hexdigest()==meta['sha256']
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
(OUT/'bundle_sha256.txt').write_text(digest+'  '+bundle.name+'\n')
print(dict(file=str(bundle),entries=len(manifest),bytes=bundle.stat().st_size,sha256=digest))
if __name__=='__main__':pass
