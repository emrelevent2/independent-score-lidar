"""Paired official RT3D controls at native resolution,25 and100 iterations."""
from pathlib import Path
import json,hashlib
import numpy as np
from run_rt3d import run_batch,EXE
from rt3d_io import read_pcl_ply,native_pixel_points

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'confirmation_v1'
RESULTS=ROOT/'results'/'confirmation_v1'


def point_report(points):
    row,col,delay,strength=native_pixel_points(points,16,16,.25)
    ids=(row==8)&(col==8);d=delay[ids];s=strength[ids]
    if not len(s):return dict(score=None,delay=None,reported_points=0)
    primary=int(np.argmax(s));allowed=np.abs(d-d[primary])>=8.
    if not allowed.any():return dict(score=None,delay=None,primary=float(d[primary]),reported_points=len(s))
    second=int(np.argmax(np.where(allowed,s,-np.inf)))
    return dict(score=float(s[second]),delay=float(d[second]),primary=float(d[primary]),reported_points=len(s))


def main():
    freeze=json.loads((ROOT/'CONFIRMATORY_FREEZE.json').read_text())
    for name in ['run_rt3d_confirmation.py','run_rt3d.py','rt3d_io.py']:
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=freeze['source_sha256'][name]:raise RuntimeError('Frozen source changed')
    if hashlib.sha256(EXE.read_bytes()).hexdigest()!=freeze['rt3d_executable_sha256']:raise RuntimeError('Executable changed')
    manifest=json.loads((DATA/'manifest.json').read_text())
    groups=[[] for _ in range(4)];edges=np.array([40.,80.,200.]);references=[20.,60.,140.,400.]
    for item in manifest['cases']:
        with np.load(DATA/(item['id']+'.npz')) as d:h=d['histogram']
        mean=float(h.sum(-1).mean());groups[int(np.searchsorted(edges,mean,side='right'))].append(item)
    irf=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    reports={item['id']:{} for item in manifest['cases']};runs=[]
    for iterations in [25,100]:
        for bucket,items in enumerate(groups):
            if not items:continue
            t=np.arange(256);g=np.exp(-4*np.log(2)*((t-80)/5.)**2);g/=g.sum()
            pilot=np.broadcast_to(np.round(references[bucket]*g).astype(np.int32),(16,16,256))
            frames=np.stack([pilot]+[np.load(DATA/(i['id']+'.npz'))['histogram'] for i in items])
            name=f'rt3d_confirmation_v1_{iterations}_{bucket}'
            output,info=run_batch(name,frames,irf,iterations,.25,references[bucket])
            for frame,item in enumerate(items,1):
                reports[item['id']][f'rt3d_{iterations}']=point_report(read_pcl_ply(output/f'frame{frame}_w0.ply'))
            runs.append(dict(bucket=bucket,reference_mean=references[bucket],run=name,**info))
            (RESULTS/'rt3d_reports_partial.json').write_text(json.dumps(dict(reports=reports,runs=runs),indent=2,allow_nan=False))
            print('Completed RT3D',iterations,'bucket',bucket,'observations',len(items),flush=True)
    if any(len(v)!=2 for v in reports.values()):raise RuntimeError('Incomplete RT3D cases')
    (RESULTS/'rt3d_reports.json').write_text(json.dumps(dict(reports=reports,runs=runs,
        notes=['Unmodified official executable; native grid; nominal FWHM5.',
               'One deterministic pilot frame per photon-count bucket is excluded from statistics.',
               'Buckets use only observed photon count and fixed reference values20,60,140,400.',
               'Each frame is independently initialized by the executable.']),indent=2,allow_nan=False))


if __name__=='__main__':main()
