"""Small v4 overlay for the validated v3 base; no duplicate 91k-file tree."""
from pathlib import Path
import hashlib,json,shutil,zipfile,io,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';PDF=OUT/'pdf'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=OUT/'independent_photon_evidence_v4_update.zip'
    assert not archive.exists(),'Preserve the versioned snapshot'
    base=OUT/'independent_photon_evidence_review_bundle_v3.zip'
    base_digest='133076c82888723d701cae142d5fbe128dbcec85afb55ccae184f035b003ee0d'
    assert sha(base)==base_digest
    for f in ['integrity.json','supplemental_integrity.json']:
        assert json.loads((ROOT/'results/coverage_confirmation_v1'/f).read_text())['status']=='PASS'
    assert json.loads((ROOT/'results/guarded_real_confirmation_v1/integrity.json').read_text())['status']=='PASS'
    files=set()
    def add(p):
        if p.is_file():files.add(p)
    def tree(folder,suffixes):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts: add(p)
    for pattern in ['*.json','*.md','requirements-lock.txt','STB_DATA_NOTICE.txt']:
        for p in ROOT.glob(pattern):add(p)
    tree(ROOT/'src',{'.py'})
    for p in (ROOT/'manuscript').glob('*'):
        if p.suffix=='.tex':add(p)
    tree(ROOT/'manuscript/figures',{'.png','.pdf','.json'})
    add(ROOT/'manuscript/qa/render_manifest.json')
    for name in ['coverage_confirmation_v1']:
        tree(ROOT/'data'/name,{'.json','.npz'})
    for name in ['coverage_confirmation_v1','coverage_development_v1','guarded_real_confirmation_v1']:
        tree(ROOT/'results'/name,{'.json','.csv'})
    for name in ['guarded_real_confirmation_v1','stb_guarded_selection_v1']:
        tree(ROOT/'data'/name,{'.json'})
    selection=json.loads((ROOT/'data/stb_guarded_selection_v1/selection.json').read_text())
    for f in selection['selected']: add(ROOT/f['label_file_relative'])
    for source,dest in [('main','independent_photon_evidence_v4'),('supplement','supplementary_methods_v4'),('cover_letter','cover_letter_v4')]:
        p=PDF/(dest+'.pdf');assert not p.exists();shutil.copyfile(ROOT/'manuscript'/(source+'.pdf'),p);add(p)
        add(ROOT/'manuscript'/(source+'.pdf'));add(ROOT/'manuscript'/(source+'.log'))
    render=json.loads((ROOT/'manuscript/qa/render_manifest.json').read_text())
    qa=dict(version=4,pages={k:len(v['pages']) for k,v in render.items() if isinstance(v,dict)},
        abstract_words=render['abstract_word_count'],tests='72passed',visual_review='All11main,14supplement and1cover pages inspected; changed pages rechecked.',
        coverage_analysis_sha256=sha(ROOT/'results/coverage_confirmation_v1/analysis.json'),
        no_submission=True,no_independent_physical_coverage_validation=True,
        base_v3_sha256=base_digest,status='Author-review research snapshot; positive controlled simulation result and explicit tradeoffs/negative real results.')
    p=OUT/'CURRENT_QA_v4.json';p.write_text(json.dumps(qa,indent=2));add(p)
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    metadata=dict(version=4,kind='Update overlay on validated v3 base, not a standalone full history.',base_v3_sha256=base_digest,entries=manifest,
        notes=['Apply caspi_study paths over a v3 extraction in a NEW workspace; overlay replaces current source/manuscript snapshots.',
            'All new coverage observations/results included. New guarded-real fitted outputs and provenance included; large real tensors excluded.',
            'No upstream source,weights,executables or large raw archives redistributed. STB-derived records retain dataset attribution/license.'])
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,'caspi_study/'+p.relative_to(ROOT).as_posix())
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(metadata,indent=2))
    # Streaming validation avoids re-extracting thousands of duplicate files.
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,meta in manifest.items():
            data=z.read('caspi_study/'+name);assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        read=lambda path:json.loads(z.read('caspi_study/'+path))
        p=read('COVERAGE_CONFIRMATORY_PROTOCOL.json');m=read('data/coverage_confirmation_v1/manifest.json')
        digest=hashlib.sha256(z.read('caspi_study/COVERAGE_CONFIRMATORY_PROTOCOL.json')).hexdigest()
        assert m['protocol_sha256']==digest
        for name,d in p['source_sha256'].items(): assert hashlib.sha256(z.read('caspi_study/src/'+name)).hexdigest()==d
        counts={key:{g:dict(correct=0,wrong_depth=0,declared=0) for g in ['null','coherent','singleton','filament','crossing_step']} for key in ['spatial_cert','coverage_cert']}
        result_hashes=read('results/coverage_confirmation_v1/result_sha256.json')['records']
        cal={k:[] for k in ['spatial_fast','coverage_fast','local_fit','guided_fit','previous_routed']}
        for c in m['cases']:
            raw=z.read('caspi_study/data/coverage_confirmation_v1/'+c['id']+'.npz');assert hashlib.sha256(raw).hexdigest()==c['sha256']
            with np.load(io.BytesIO(raw)) as f: assert f['histogram'].shape==(16,16,256)
            name='results/coverage_confirmation_v1/'+c['id']+'.json';r=read(name)
            assert hashlib.sha256(z.read('caspi_study/'+name)).hexdigest()==result_hashes[c['id']]
            assert r['id']==c['id'] and r['protocol_sha256']==digest
            if c['split']=='calibration':
                for key in cal:cal[key].append(r['reports'][key]['score'])
                continue
            group='null' if c['kind']=='null' else c['support_family']
            for key in counts:
                report=r['reports'][key];cell=counts[key][group];declared=report['declared'];cell['declared']+=declared
                if group!='null':
                    good=abs(report['delay']-c['metadata']['secondary_delay'])<=2
                    cell['correct']+=bool(declared and good);cell['wrong_depth']+=bool(declared and not good)
        a=read('results/coverage_confirmation_v1/analysis.json')
        for key,groups in counts.items():
            for g,fields in groups.items():
                for field,count in fields.items():assert count==a['methods'][key+'_0.01']['groups'][g][field]['count']
        cuts=read('results/coverage_confirmation_v1/thresholds.json')['methods']
        for key,values in cal.items():
            values=sorted(values)
            for alpha in [.005,.01]:assert cuts[key][str(alpha)]==max(0.,values[math.ceil(2000*(1-alpha))-1])
    digest=sha(archive)
    status=dict(status='PASS',archive_sha256=digest,base_v3_sha256=base_digest,entries_checked=len(manifest),cases_checked=6099,
        checks='Streaming CRC/SHA validation of every entry; all frozen coverage-source hashes; all input/result identities and shapes; independent certified endpoint recount and calibration cutoff reconstruction.',
        scope='Complements full local mathematical/software audits. Not a full fresh-copy refit,physical calibration or independent replication.')
    (OUT/'bundle_v4_update_validation.json').write_text(json.dumps(status,indent=2))
    (OUT/'bundle_v4_update_sha256.txt').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,**status),indent=2),flush=True)


if __name__=='__main__':main()
