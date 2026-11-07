"""Development-only calibration/geometry audit on STB scene1.

No unpickling is performed. The pinned object NPY is statically disassembled
and its numeric byte buffers are decoded after an exact source-hash check.
Livox reference geometry is not a full hidden-secondary-return truth map.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,pickletools,struct
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];META=ROOT/'data'/'stb_metadata';DEV=ROOT/'data'/'stb_development'
OUT=ROOT/'results'/'stb_alignment_development_v1';P=ROOT/'STB_ALIGNMENT_DEVELOPMENT_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
CALIBRATION_SHA='6f45a6b4c7cc80310744d530654d3fe84a75b60982fc54b25935bdfa20152aa2'
BIN_M=299792458*750e-12/2


def static_calibration(path):
    if sha(path)!=CALIBRATION_SHA:raise ValueError('Only the pinned inspected calibration artifact is supported')
    with path.open('rb') as f:
        assert np.lib.format.read_magic(f)==(1,0)
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        assert shape==() and not fortran and dtype==np.dtype('O')
        payload=f.read()
    ops=list(pickletools.genops(payload))
    blobs=[a for o,a,pos in ops if isinstance(a,bytes) and len(a)>100]
    assert [len(b) for b in blobs]==[589824,196850]
    # These are data bytes, not calls to NumPy/Torch pickle reconstruction.
    rays=np.frombuffer(blobs[0],dtype='<f4').reshape(192,256,3).copy()
    storage=blobs[1];pos=0;streams=[]
    for i in range(5):
        stream=list(pickletools.genops(storage[pos:]));streams.append(stream);pos+=stream[-1][2]+1
    assert streams[0][1][1]==119547037146038801333356
    assert streams[1][1][1]==1001
    assert any(o.name=='GLOBAL' and a=='torch FloatStorage' for o,a,p in streams[3])
    assert struct.unpack('<Q',storage[pos:pos+8])[0]==49152
    assert len(storage)-pos-8==49152*4
    shift=np.frombuffer(storage,dtype='<f4',offset=pos+8).reshape(192,256).copy()
    assert np.isfinite(rays).all() and np.isfinite(shift).all()
    assert np.max(abs(np.linalg.norm(rays,axis=-1)-1))<1e-6
    return rays,shift


def freeze():
    if P.exists():raise RuntimeError('Existing development snapshot retained')
    files=[META/'config'/'intrinsics.npy',META/'config'/'offset.txt',META/'config'/'final_extrinsic_sp_to_livox.txt',
        DEV/'DepthEstimate'/'Histogram'/'1.txt',DEV/'DepthEstimate'/'gt'/'1.csv',DEV/'scene1_histograms.npz']
    p=dict(status='DEVELOPMENT geometry audit on already downloaded scene1; no new method accuracy claim.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        input_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        settings=dict(bin_ps=750,range_min_m=.25,range_max_m=30,reference_assignment_pixel_radius=.75,
            reference_flat_spread_bins=2.,reference_min_points=2),
        notes=['Primary geometry uses supplied numeric rays/shift, statically decoded without executing any pickle globals.',
            'Source raw-to-raw SP-to-Livox transform is applied as supplied. No ICP, fitted transform, depth bias, or truth-driven camera choice.',
            'Official shift relation is checked against (peakbin-17)*bin_width-offset on calibration-valid pixels.',
            'Livox is converted to the SP frame with the full inverse of the supplied4x4 matrix, not a transpose.',
            'Reference pixels use nearest calibrated angular ray within0.75localpixel pitch, minimum range per pixel; flat mask depends only on Livox.',
            'Raw argmax is a calibration baseline. Its reference consistency is not a proposed-method result.',
            'No N1000 first-event model is used; no cycle count is invented. Other nine depth scenes remain untouched.',
            'Livox geometry does not certify absence of hidden surfaces. No secondary false-positive or true-positive labels are inferred.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen STB development alignment settings',flush=True)


def run():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['source_sha256']
    for n,d in p['input_sha256'].items():assert sha(ROOT/n)==d,n
    OUT.mkdir(exist_ok=True);rays,shift=static_calibration(META/'config'/'intrinsics.npy')
    offset=np.loadtxt(META/'config'/'offset.txt').reshape(192,256);valid_cal=offset<10
    shift_error=float(np.max(abs(shift[valid_cal]+17*BIN_M+offset[valid_cal])))
    assert shift_error<5e-7
    np.savez_compressed(OUT/'calibration_numeric.npz',rays=rays,shift_m=shift,valid_calibration=valid_cal)
    transform=np.loadtxt(META/'config'/'final_extrinsic_sp_to_livox.txt')
    assert transform.shape==(4,4) and np.allclose(transform[3],[0,0,0,1])
    gt=np.loadtxt(DEV/'DepthEstimate'/'gt'/'1.csv',delimiter=',',skiprows=1,usecols=(8,9,10))
    gt=gt[np.isfinite(gt).all(-1)&(np.linalg.norm(gt,axis=-1)>.01)]
    inv=np.linalg.inv(transform);camera=gt@inv[:3,:3].T+inv[:3,3]
    radius=np.linalg.norm(camera,axis=-1);settings=p['settings']
    valid=(camera[:,2]>0)&(radius>settings['range_min_m'])&(radius<settings['range_max_m'])
    gt,camera,radius=gt[valid],camera[valid],radius[valid]
    directions=camera/radius[:,None];rf=rays.reshape(-1,3)
    dx=np.linalg.norm(np.diff(rays,axis=1),axis=-1);dy=np.linalg.norm(np.diff(rays,axis=0),axis=-1)
    pitch=np.minimum(np.pad(dx,((0,0),(0,1)),mode='edge'),np.pad(dy,((0,1),(0,0)),mode='edge')).ravel()
    chord,pixel=cKDTree(rf).query(directions,workers=1)
    accepted=(chord<=settings['reference_assignment_pixel_radius']*pitch[pixel])&valid_cal.ravel()[pixel]
    nearest=np.full(49152,np.inf);farthest=np.full(49152,-np.inf)
    np.minimum.at(nearest,pixel[accepted],radius[accepted]);np.maximum.at(farthest,pixel[accepted],radius[accepted])
    counts=np.bincount(pixel[accepted],minlength=49152)
    reference_mask=np.isfinite(nearest)&valid_cal.ravel()
    flat=reference_mask&(counts>=settings['reference_min_points'])&((farthest-nearest)<=settings['reference_flat_spread_bins']*BIN_M)
    with np.load(DEV/'scene1_histograms.npz') as z:h=z['histograms']
    peak=h.argmax(-1);pred_range=peak*BIN_M+shift
    prediction_valid=valid_cal&(h.sum(-1)>0)&(pred_range>settings['range_min_m'])&(pred_range<settings['range_max_m'])
    def metrics(mask):
        ok=mask&prediction_valid.ravel();err=pred_range.ravel()[ok]-nearest[ok]
        return dict(reference_pixels=int(mask.sum()),prediction_pixels=int(ok.sum()),
            prediction_coverage=float(ok.sum()/mask.sum()),median_signed_range_error_m=float(np.median(err)),
            median_absolute_range_error_m=float(np.median(abs(err))),
            reference_consistency={str(b):float(np.sum(ok&(abs(pred_range.ravel()-nearest)<=b*BIN_M))/mask.sum()) for b in [1,3,5]})
    pred_cam=rf[prediction_valid.ravel()]*pred_range.ravel()[prediction_valid.ravel(),None]
    pred_livox=pred_cam@transform[:3,:3].T+transform[:3,3]
    d_forward=cKDTree(gt).query(pred_livox,workers=1)[0]
    d_backward=cKDTree(pred_livox).query(gt[accepted],workers=1)[0]
    summary=dict(status=p['status'],protocol_sha256=sha(P),bin_m=BIN_M,
        static_numeric_decode=True,maximum_shift_formula_error_m=shift_error,calibrated_pixels=int(valid_cal.sum()),
        livox_points=int(len(gt)),angularly_assigned_livox_points=int(accepted.sum()),
        raw_argmax=dict(all_reference=metrics(reference_mask),flat_reference=metrics(flat)),
        nearest_point_consistency=dict(forward_mean_m=float(d_forward.mean()),backward_mean_m=float(d_backward.mean()),
            chamfer_sum_mean_distances_m=float(d_forward.mean()+d_backward.mean()),
            note='Own stated field-of-view/count filters. Not an exact reproduction of the published benchmark score.'),
        no_secondary_truth_claim=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    reference=np.where(reference_mask,nearest,np.nan).reshape(192,256)
    np.savez_compressed(OUT/'geometry_maps.npz',raw_range_m=pred_range,reference_range_m=reference,
        reference_mask=reference_mask.reshape(192,256),flat_mask=flat.reshape(192,256),valid_prediction=prediction_valid,
        reference_point_counts=counts.reshape(192,256))
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    axes[0].imshow(np.where(prediction_valid,pred_range,np.nan),vmin=0,vmax=10,cmap='viridis');axes[0].set_title('Raw argmax range (m)')
    axes[1].imshow(reference,vmin=0,vmax=10,cmap='viridis');axes[1].set_title('Livox range reference (m)')
    im=axes[2].imshow(np.where(flat.reshape(192,256)&prediction_valid,pred_range-reference,np.nan),vmin=-.5,vmax=.5,cmap='coolwarm');axes[2].set_title('Range residual, flat reference')
    for ax in axes:ax.set_axis_off()
    fig.colorbar(im,ax=axes[2],shrink=.75,label='Range error (m)')
    fig.savefig(OUT/'geometry_audit.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
