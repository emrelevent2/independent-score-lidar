"""Technical post-freeze adapter repair: preserve zero-point author outputs.

No frozen method/parameter file is changed. The official PCL writer omits a
file for an empty cloud. A valid zero-vertex PLY is materialized ONLY when the
completed executable log explicitly reports zero points for that frame.
Completed batches are verified and reused, not recomputed.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
from run_rt3d import run_batch,EXE
from run_rt3d_confirmation import point_report
from rt3d_io import read_pcl_ply

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'confirmation_v1';RESULTS=ROOT/'results'/'confirmation_v1'


def repair_empty_outputs(work):
    manifest=json.loads((work/'run_manifest.json').read_text())
    if manifest['exit_code']!=0:raise RuntimeError('Executable failed; cannot infer empty clouds')
    log=(work/'run.log').read_text(encoding='utf-8',errors='replace')
    pairs=re.findall(r'Number of points:\s*(\d+).*?frame(\d+)_w0\.ply was saved\.',log,re.S)
    mapping={int(frame):int(n) for n,frame in pairs}
    if len(pairs)!=manifest['frames'] or set(mapping)!=set(range(manifest['frames'])):
        raise RuntimeError('Incomplete per-frame author log')
    output=work/'output_batch';repaired=[]
    header=('ply\nformat binary_little_endian 1.0\ncomment Verified zero-point RT3D log\n'
            'element vertex 0\n'+''.join('property float '+n+'\n' for n in ['x','y','z','intensity','nx','ny','nz','curvature'])+'end_header\n').encode('ascii')
    for frame,n in mapping.items():
        path=output/f'frame{frame}_w0.ply'
        if not path.exists() or path.stat().st_size==0:
            if n!=0:raise RuntimeError('Missing NONEMPTY cloud: '+str(path))
            path.write_bytes(header);repaired.append(frame)
        points=read_pcl_ply(path)
        if len(points)!=n:raise RuntimeError('Point count mismatch in '+str(path))
    note=dict(reason='Official PCL writer omits empty point-cloud files',
              zero_frame_indices=repaired,all_frame_counts_verified=True,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (work/'empty_output_adapter_note.json').write_text(json.dumps(note,indent=2))
    return output,manifest,note


def main():
    freeze=json.loads((ROOT/'CONFIRMATORY_FREEZE.json').read_text())
    for name in ['run_rt3d_confirmation.py','run_rt3d.py','rt3d_io.py']:
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=freeze['source_sha256'][name]:raise RuntimeError('Frozen source changed')
    if hashlib.sha256(EXE.read_bytes()).hexdigest()!=freeze['rt3d_executable_sha256']:raise RuntimeError('Executable changed')
    manifest=json.loads((DATA/'manifest.json').read_text());groups=[[] for _ in range(4)]
    for item in manifest['cases']:
        with np.load(DATA/(item['id']+'.npz')) as d:h=d['histogram']
        bucket=int(np.searchsorted([40.,80.,200.],float(h.sum(-1).mean()),side='right'));groups[bucket].append(item)
    references=[20.,60.,140.,400.];irf=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    reports={r['id']:{} for r in manifest['cases']};runs=[]
    for iterations in [25,100]:
        for bucket,items in enumerate(groups):
            if not items:continue
            name=f'rt3d_confirmation_v1_{iterations}_{bucket}';work=ROOT/'tools'/name
            existing=work/'run_manifest.json'
            if existing.exists():
                info=json.loads(existing.read_text())
                data=work/'datasets'/'batch.rbin'
                if hashlib.sha256(data.read_bytes()).hexdigest()!=info['dataset_sha256']:raise RuntimeError('Retained data changed')
                if info['frames']!=len(items)+1 or info['iterations']!=iterations:raise RuntimeError('Retained batch differs')
            else:
                t=np.arange(256);g=np.exp(-4*np.log(2)*((t-80)/5.)**2);g/=g.sum()
                pilot=np.broadcast_to(np.round(references[bucket]*g).astype(np.int32),(16,16,256))
                frames=np.stack([pilot]+[np.load(DATA/(i['id']+'.npz'))['histogram'] for i in items])
                try:run_batch(name,frames,irf,iterations,.25,references[bucket])
                except RuntimeError as error:
                    if 'Incomplete RT3D outputs' not in str(error):raise
            output,info,note=repair_empty_outputs(work)
            for frame,item in enumerate(items,1):
                reports[item['id']][f'rt3d_{iterations}']=point_report(read_pcl_ply(output/f'frame{frame}_w0.ply'))
            runs.append(dict(bucket=bucket,run=name,adapter_note=note,**info))
            (RESULTS/'rt3d_reports_partial.json').write_text(json.dumps(dict(reports=reports,runs=runs),indent=2,allow_nan=False))
            print('Completed RT3D',iterations,'bucket',bucket,'cases',len(items),'verified empty outputs',len(note['zero_frame_indices']),flush=True)
    if any(len(r)!=2 for r in reports.values()):raise RuntimeError('Incomplete query reports')
    (RESULTS/'rt3d_reports.json').write_text(json.dumps(dict(reports=reports,runs=runs,
        technical_adapter_note='Post-freeze empty-cloud representation repair only; frozen executable/parameters unchanged.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,allow_nan=False))


if __name__=='__main__':main()
