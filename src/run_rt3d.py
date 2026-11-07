"""Run the unmodified authors' executable without an interactive window.

The archived executable's actual prompt order was checked on an alignment
fixture. Two or more frames suppress its single-frame viewer. Keep provenance
and report every chosen parameter; this is native-resolution RT3D, not its
default super-resolution configuration.
"""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
from rt3d_io import write_rbin

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'tools'/'rt3d'/'RT3D.exe'


def run_batch(name,h,irf,iterations=25,scale_ratio=.25,reference_mean=None):
    if len(h)<2:raise ValueError('At least two frames are required to suppress the viewer')
    work=ROOT/'tools'/name
    (work/'datasets').mkdir(parents=True,exist_ok=True);(work/'hyperparams').mkdir(exist_ok=True)
    data=work/'datasets'/'batch.rbin';hyp=work/'hyperparams'/'batch.hyp'
    if hyp.exists():raise ValueError('Existing hyperparameters: choose a fresh run name')
    write_rbin(data,h,irf,scale_ratio)
    mean=float(np.mean(np.sum(h,axis=-1))) if reference_mean is None else float(reference_mean)
    max_reflectivity=5*mean
    # Restore the authors' 10*mean_signal rule for a NATIVE grid at SBR=1,
    # rather than retaining the subpixel cap from default upsampling.
    values=['1','1','1','y',str(scale_ratio),'2','1','1','0','0.1',str(iterations),
            '1','6','10',str(max_reflectivity),str(1/max_reflectivity),'','']
    begin=time.perf_counter()
    process=subprocess.Popen([str(EXE)],cwd=work,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:stdout,_=process.communicate(('\n'.join(values)+'\n').encode('ascii'),timeout=600)
    except subprocess.TimeoutExpired:
        process.kill();stdout,_=process.communicate()
        (work/'run.log').write_bytes(stdout)
        raise RuntimeError('RT3D timed out; inspect saved log')
    (work/'run.log').write_bytes(stdout)
    info=dict(executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
              dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
              iterations=iterations,scale_ratio=scale_ratio,upsampling=1,
              initial_returns=2,internal_intensity_threshold=0.,SBR=1.,
              proportion=.1,neighborhood_radius=1,APSS_depth_kernel=6.,
              background_regularization=10.,max_reflectivity=max_reflectivity,
              parameter_reference_mean=mean,
              reflectivity_step=1/max_reflectivity,depth_step='authors default',
              elapsed_seconds=time.perf_counter()-begin,exit_code=process.returncode,
              frames=len(h),prompt_inputs=values)
    (work/'run_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    output=work/'output_batch'
    if process.returncode!=0 or len(list(output.glob('frame*_w0.ply')))!=len(h):
        raise RuntimeError('Incomplete RT3D outputs; inspect '+str(work/'run.log'))
    return output,info


if __name__=='__main__':
    from rt3d_io import read_pcl_ply,native_pixel_points
    t=np.arange(256);yy,xx=np.mgrid[:16,:16]
    truth=np.stack([np.full((16,16),80.),60.+.25*yy+.5*xx])
    flux=np.exp(-4*np.log(2)*((t-truth[...,None])/5.)**2)
    flux/=flux.sum(-1,keepdims=True)
    h=np.round(500*flux).astype(int)
    irf=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    output,info=run_batch('rt3d_adapter_check',h,irf)
    errors=[]
    for frame in range(2):
        points=read_pcl_ply(output/f'frame{frame}_w0.ply')
        r,c,d,s=native_pixel_points(points,16,16,.25)
        error=np.abs(d-truth[frame,r,c]);errors.extend(error)
        print('Frame',frame,'points',len(d),'maximum alignment error',float(error.max()),flush=True)
    info['maximum_alignment_error_bins']=float(np.max(errors))
    (ROOT/'results'/'rt3d_adapter_check.json').write_text(json.dumps(info,indent=2))
