from pathlib import Path
import json
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from pilot import coates

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--full',action='store_true'); args=parser.parse_args()
mode='full' if args.full else 'crop'
out=root/'results'; figdir=out/'figures'; figdir.mkdir(exist_ok=True)
h=loadmat(root.parent/'caspi_original/Data/LiDAR/art.mat')['hst_map_set']
if not args.full: h=h[64:96,80:112]
with h5py.File(out/f'caspi_art_{mode}.mat') as f:
    flux=np.asarray(f['result/flux']).transpose(2,1,0)
    stage=np.asarray(f['result/stage_seconds']).ravel()
    seconds=float(np.asarray(f['result/runtime_seconds']).squeeze())
raw=coates(h)
sigma=5/(2*np.sqrt(2*np.log(2)))
time_bin=80e-12; c=3e8
raw_depth=np.argmax(gaussian_filter1d(raw,sigma,axis=-1),axis=-1)*time_bin*c/2
caspi_depth=np.argmax(gaussian_filter1d(flux,sigma,axis=-1),axis=-1)*time_bin*c/2
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
im=axes[0].imshow(raw_depth,vmin=1.4,vmax=2.17,cmap='viridis'); axes[0].set_title('Raw + matched filter')
axes[1].imshow(caspi_depth,vmin=1.4,vmax=2.17,cmap='viridis'); axes[1].set_title('Original CASPI + matched filter')
for ax in axes[:2]: ax.set_xlabel('Column'); ax.set_ylabel('Row')
fig.colorbar(im,ax=axes[:2],label='Estimated depth (m)',shrink=.8)
t=np.arange(h.shape[-1])*time_bin*1e9
row,col=(80,96) if args.full else (16,16)
axes[2].plot(t,raw[row,col],color='.65',lw=.8,label='Coates corrected')
axes[2].plot(t,flux[row,col],color='#176087',label='CASPI')
axes[2].set_xlim(5,22); axes[2].set_xlabel('Time (ns)'); axes[2].set_ylabel('Estimated photons per cycle per bin')
axes[2].legend(fontsize=8); axes[2].set_title('One observed transient')
fig.suptitle(f'Execution check on supplied art {mode} scene — no supplied depth ground truth')
for ext in ['png','pdf']: fig.savefig(figdir/f'caspi_{mode}_reproduction.{ext}',dpi=200)
summary={'status':'execution reproduction, not accuracy validation','shape':list(flux.shape),
 'runtime_seconds':seconds,'stage_seconds':stage.tolist(),
 'all_finite':bool(np.isfinite(flux).all()),'nonnegative':bool((flux>=0).all()),
 'depth_plot_estimator':'Gaussian matched-filter argmax, zero-based bin centers; qualitative only'}
(out/f'{mode}_reproduction_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
