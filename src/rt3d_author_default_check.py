"""Additional authors-default reference, outside the primary frozen analysis.

Declared before query-level outcomes were inspected. The planned native-grid
controls remain unchanged. Aggregate the super-resolved point cloud into its
source pixel, then apply the SAME fitted-flux secondary report as other flux
controls. No ground-truth delays enter aggregation or reporting.
"""
from pathlib import Path
import hashlib,json,re,struct,subprocess,time
import numpy as np
from rt3d_io import write_rbin,read_pcl_ply
from likelihood_baseline import fitted_flux_report
from complete_rt3d_confirmation import repair_empty_outputs

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'confirmation_v1';RESULTS=ROOT/'results'/'confirmation_v1'
EXE=ROOT/'tools'/'rt3d'/'RT3D.exe'


def aggregate(points,upsampling,scale_ratio=.25):
    row=(16-points['z']/scale_ratio)*upsampling
    col=(16-points['y']/scale_ratio)*upsampling
    if not np.allclose(row,np.rint(row),atol=.002) or not np.allclose(col,np.rint(col),atol=.002):
        raise ValueError('Unexpected transverse output grid')
    row=np.rint(row).astype(int)//upsampling;col=np.rint(col).astype(int)//upsampling
    selected=(row==8)&(col==8)
    depth=points['x'][selected];strength=points['intensity'][selected]
    if not len(depth):return np.zeros(256)
    t=np.arange(256);g=np.exp(-4*np.log(2)*((t[None,:]-depth[:,None])/5.)**2)
    g/=g.sum(-1,keepdims=True)
    return np.sum(strength[:,None]*g,axis=0)/1000


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());groups=[[] for _ in range(4)]
    for item in manifest['cases']:
        with np.load(DATA/(item['id']+'.npz')) as d:h=d['histogram']
        groups[int(np.searchsorted([40.,80.,200.],float(h.sum(-1).mean()),side='right'))].append(item)
    references=[20.,60.,140.,400.];reports={};runs=[]
    for bucket,items in enumerate(groups):
        if not items:continue
        work=ROOT/'tools'/f'rt3d_authors_default_v1_{bucket}'
        (work/'datasets').mkdir(parents=True,exist_ok=True);(work/'hyperparams').mkdir(exist_ok=True)
        if (work/'run_manifest.json').exists():raise RuntimeError('Existing run; inspect before restarting')
        t=np.arange(256);g=np.exp(-4*np.log(2)*((t-80)/5.)**2);g/=g.sum()
        pilot=np.broadcast_to(np.round(references[bucket]*g).astype(int),(16,16,256))
        frames=np.stack([pilot]+[np.load(DATA/(r['id']+'.npz'))['histogram'] for r in items])
        irf=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
        path=work/'datasets'/'batch.rbin';write_rbin(path,frames,irf,.25)
        process=subprocess.Popen([str(EXE)],cwd=work,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        start=time.perf_counter()
        try:stdout,_=process.communicate(b'1\n1\n1\nn\n\n',timeout=600)
        except subprocess.TimeoutExpired:
            process.kill();stdout,_=process.communicate();(work/'run.log').write_bytes(stdout);raise
        (work/'run.log').write_bytes(stdout)
        hyp=(work/'hyperparams'/'batch.hyp').read_bytes()
        parameters=struct.unpack('<ifififfifffffif',hyp)
        upsampling=int(parameters[2])
        if upsampling!=6:raise RuntimeError('Unexpected authors default upsampling')
        info=dict(executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  frames=len(frames),exit_code=process.returncode,elapsed_seconds=time.perf_counter()-start,
                  upsampling=upsampling,default_parameters=list(parameters),
                  additional_reference=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (work/'run_manifest.json').write_text(json.dumps(info,indent=2))
        output,_,note=repair_empty_outputs(work)
        for index,item in enumerate(items,1):
            flux=aggregate(read_pcl_ply(output/f'frame{index}_w0.ply'),upsampling)
            reports[item['id']]=fitted_flux_report(flux)
        runs.append(dict(bucket=bucket,empty_note=note,**info))
        print('Completed authors defaults bucket',bucket,'cases',len(items),flush=True)
    for report in reports.values():
        if not np.isfinite(report['score']):report['score']=None
    (RESULTS/'rt3d_author_default_reports.json').write_text(json.dumps(dict(reports=reports,runs=runs,
        status='additional non-primary reference; original defaults; declared before accuracy inspection'),indent=2,allow_nan=False))


if __name__=='__main__':main()
