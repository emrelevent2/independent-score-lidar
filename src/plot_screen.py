from pathlib import Path
import csv
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from methods import all_methods,block_wiener_candidate

root=Path(__file__).resolve().parents[1]
dest=root/'results'; figdir=dest/'figures'; figdir.mkdir(exist_ok=True)
rows=list(csv.DictReader((dest/'development_screen.csv').open()))
methods=['raw','box3','haar_mad','anscombe_haar','candidate','candidate_constant_variance']
conditions=list(dict.fromkeys(r['condition'] for r in rows))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for method in methods:
    entries=[next(r for r in rows if r['condition']==c and r['method']==method) for c in conditions]
    axes[0].plot(conditions,[float(r['weak_recall']) for r in entries],marker='o',label=method)
    axes[1].plot(conditions,[float(r['null_fpr']) for r in entries],marker='o',label=method)
axes[0].set_ylabel('Weak return recall within 2 timing bins')
axes[1].set_ylabel('Observed null false positives per pixel')
axes[1].axhline(.01,color='k',ls='--',lw=1,label='Calibration target')
for ax in axes:
    ax.tick_params(axis='x',rotation=30); ax.grid(alpha=.2)
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
fig.suptitle('Development screening: fixed detector, independent calibration seeds, no CASPI batch yet')
fig.savefig(figdir/'development_screen.png',dpi=180)
fig.savefig(figdir/'development_screen.pdf')

data=loadmat(root/'data/pilot_two_returns.mat')
h=data['hst_map_set']; truth=data['truth']; methods=all_methods(h)
with h5py.File(dest/'caspi_pilot_two_returns.mat') as f:
    methods['original_CASPI']=np.asarray(f['result/flux']).transpose(2,1,0)
methods['block_wiener_v2']=block_wiener_candidate(h)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
t=np.arange(256)
for name in ['box3','candidate','candidate_constant_variance','original_CASPI','block_wiener_v2']:
    f=methods[name]
    axes[0].plot(t,f[5:11,5:11].mean((0,1)),label=name)
    # Spatial profile of the late-window flux, shown as leakage diagnostic.
    axes[1].plot(np.arange(16),f[6:10,:,89:96].sum(-1).mean(0),label=name)
axes[0].plot(t,truth[5:11,5:11].mean((0,1)),color='k',ls='--',label='Known incident flux')
axes[1].plot(np.arange(16),truth[6:10,:,89:96].sum(-1).mean(0),color='k',ls='--',label='Known incident flux')
axes[0].set_xlim(70,108); axes[0].set_xlabel('Timing bin'); axes[0].set_ylabel('Incident photons per cycle per bin')
axes[1].set_xlabel('Column'); axes[1].set_ylabel('Late-window incident photons per cycle')
for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
fig.suptitle('Single development observation: waveform preservation and spatial leakage')
fig.savefig(figdir/'pilot_waveforms.png',dpi=180); fig.savefig(figdir/'pilot_waveforms.pdf')
print('Saved screening and waveform figures')
