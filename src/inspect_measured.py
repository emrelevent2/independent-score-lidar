"""Measured-data provenance and suitability audit, not an accuracy benchmark."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from read_rt3d import read_rbin

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data'/'rt3d_measured'
    items={}
    for name in ['1_mannequin_head','2_mannequin_head_no_backplane',
                 '4_camouflage_3_2ms','5_camouflage_0_32ms']:
        d=read_rbin(folder/f'{name}.rbin');h=d['histograms'];irf=d['irf']
        items[name]=dict(shape=list(h.shape),irf_length=len(irf),
                         irf_fwhm_native_bins=int((irf>=irf.max()/2).sum()),
                         count_quantiles=np.quantile(h.sum(-1),[0,.25,.5,.75,1]).tolist(),
                         sha256=hashlib.sha256((folder/f'{name}.rbin').read_bytes()).hexdigest())
    result=dict(status='measured data import and suitability audit only',datasets=items,
                source='Tachella et al., Nature Communications 10,4984 (2019)',
                doi='10.1038/s41467-019-12943-7',
                limitations=['No ground-truth return map supplied in these containers.',
                             'Acquisition cycle count and physical bin duration absent from container.',
                             'Camouflage measured IRF FWHM 11 native bins, not the nominal 5-bin Gaussian.',
                             'Do not assume independence or registration of the two acquisition files.',
                             'No method accuracy or weak-return advantage is established by import.'],
                reader_validation='Exact counts/axes match official MATLAB writer fixture; pytest test.')
    (ROOT/'results'/'measured_data_audit.json').write_text(json.dumps(result,indent=2))
    hi=read_rbin(folder/'4_camouflage_3_2ms.rbin');lo=read_rbin(folder/'5_camouflage_0_32ms.rbin')
    h=hi['histograms'][0];l=lo['histograms'][0]
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for ax,x,title in zip(axes[0],[h,l],['Supplied 3.2 ms file','Supplied 0.32 ms file']):
        im=ax.imshow(x.sum(-1),cmap='magma',aspect='auto')
        ax.set(title=title,xlabel='Column',ylabel='Row');fig.colorbar(im,ax=ax,label='Recorded photons / pixel')
    irf=hi['irf']/hi['irf'].sum()
    axes[1,0].plot(np.arange(len(irf))-np.argmax(irf),irf,color='#2563eb')
    axes[1,0].set(title='Measured instrument response',xlabel='Native bins from response peak',ylabel='Normalized response')
    pixel=(79,39)
    axes[1,1].step(np.arange(550),h[pixel],where='mid',lw=.8,label='Fixed pixel (79,39)')
    pooled=uniform_filter(h.astype(float),(9,9,1),mode='reflect')[pixel]
    axes[1,1].plot(pooled,color='#b45309',lw=1.1,label='9x9 average, same units')
    axes[1,1].set(title='Measured count profile (illustration)',xlabel='Native time bin',ylabel='Photons / bin')
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Public measured LiDAR data: import audit, not reconstruction accuracy',fontsize=11)
    output=ROOT/'results'/'figures'
    for ext in ['png','pdf']:fig.savefig(output/f'measured_data_audit.{ext}',dpi=180)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
