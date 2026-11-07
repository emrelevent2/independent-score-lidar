"""Archive-backed saved-output replay and small fresh-source test extraction.

No duplicate 100k-file tree: only Python sources/small test fixtures and one
SQLite database are materialized. Statistical source remains unmodified;
the path adapter reads immutable archived records and captures new summaries.
"""
from pathlib import Path,PurePosixPath
import contextlib,hashlib,importlib,io,json,os,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';ARCHIVE=OUT/'independent_photon_evidence_review_bundle_v5.zip'
TEMP=ROOT/'tmp/release_v5_validation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not TEMP.exists(),'Retain a prior validation workspace for inspection'
    TEMP.mkdir(parents=True);extracted=TEMP/'caspi_study';log=io.StringIO()
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))['entries']
        for name,meta in manifest.items():
            raw=z.read('caspi_study/'+name);assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
        fixtures=['data/rt3d_measured/format_fixture.rbin','data/rt3d_measured/format_fixture.mat',
            'tools/rt3d_adapter_check/output_batch/frame0_w0.ply','tools/rt3d_adapter_check/output_batch/frame1_w0.ply']
        for name in manifest:
            if (name.startswith('src/') and name.endswith('.py')) or name in fixtures:
                destination=extracted/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(z.read('caspi_study/'+name))
        env=dict(os.environ,PYTHONPATH=str(extracted/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        result=subprocess.run([sys.executable,'-m','pytest','src','-q'],cwd=extracted,env=env,capture_output=True,text=True)
        log.write('FRESH ARCHIVED SOURCE TESTS\n'+result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
        assert '76 passed' in result.stdout
        writes={};materialized={}
        class CaptureText(io.StringIO):
            def __init__(self,name):super().__init__();self.target=name
            def close(self):
                if not self.closed:writes[self.target]=self.getvalue()
                super().close()
        class ArchivePath:
            def __init__(self,name=''):self.name=str(PurePosixPath(name)) if name else ''
            def __truediv__(self,value):return ArchivePath(str(PurePosixPath(self.name)/str(value)))
            def read_bytes(self):return writes[self.name].encode() if self.name in writes else z.read('caspi_study/'+self.name)
            def read_text(self,*args,**kwargs):return self.read_bytes().decode('utf-8')
            def write_text(self,value,*args,**kwargs):writes[self.name]=value;return len(value)
            def open(self,mode='r',*args,**kwargs):
                if mode.startswith('w'):
                    assert 'b' not in mode
                    return CaptureText(self.name)
                return io.BytesIO(self.read_bytes()) if 'b' in mode else io.StringIO(self.read_text())
            def exists(self):return self.name in writes or self.name in manifest
            def is_file(self):return self.exists()
            def __str__(self):return 'ARCHIVE/'+self.name
            def __fspath__(self):
                # SQLite needs a real local file; all JSON reads remain archived.
                assert self.name.endswith('.sqlite3'),self.name
                if self.name not in materialized:
                    path=TEMP/'sqlite'/Path(self.name).name;path.parent.mkdir(exist_ok=True);path.write_bytes(self.read_bytes());materialized[self.name]=path
                return str(materialized[self.name])
        sys.path.insert(0,str(extracted/'src'))
        modules=[importlib.import_module(n) for n in ['coverage_confirmation_analysis','real_coverage_development','real_primary_bank_development','coverage_external_comparison']]
        # Imported path constants are redirected, not statistical operations.
        for module in list(sys.modules.values()):
            if not getattr(module,'__file__',None):continue
            try:Path(module.__file__).relative_to(extracted/'src')
            except ValueError:continue
            for key,value in list(vars(module).items()):
                if isinstance(value,Path):
                    try:relative=value.relative_to(extracted)
                    except ValueError:continue
                    setattr(module,key,ArchivePath('' if str(relative)=='.' else relative.as_posix()))
        comparisons={}
        outputs=['results/coverage_confirmation_v1/analysis.json','results/real_coverage_development_v1/analysis.json',
            'results/real_primary_bank_development_v1/analysis.json','results/coverage_external_v1/analysis.json']
        for module,name in zip(modules,outputs):
            with contextlib.redirect_stdout(log):module.analyze()
            assert json.loads(writes[name])==json.loads(z.read('caspi_study/'+name)),name
            comparisons[name]='exact match';print('Archived saved-output replay PASS',name,flush=True)
    (OUT/'bundle_v5_validation.log').write_text(log.getvalue())
    status=dict(status='PASS',archive_sha256=sha(ARCHIVE),manifest_entries_checked=len(manifest),tests='76passed on fresh archived sources and small fixtures',
        reanalyses=comparisons,materialized_workspace=str(TEMP),
        scope='All archive entry hashes; fresh-source tests; exact archived saved-output statistical replay through a read-only archive path adapter. Not a full independent refit or physical-sensor replication.')
    (OUT/'bundle_v5_validation.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
