"""Known-interval inventory of the selected real development frame."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'stb_multireturn_development';OUT=ROOT/'results'/'stb_multireturn_inventory_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=json.loads((DATA/'download_manifest.json').read_text());OUT.mkdir(exist_ok=True)
    for key in ['annotation','histogram']:assert sha(DATA/manifest[key]['local'])==manifest[key]['sha256']
    labels=np.load(DATA/manifest['annotation']['local'],allow_pickle=False,mmap_mode='r')
    assert labels.shape==(49152,672) and labels.dtype==np.uint8
    cache=DATA/'frame_histograms.npz'
    if cache.exists():h=np.load(cache)['histograms'].reshape(49152,672)
    else:
        h=np.loadtxt(DATA/manifest['histogram']['local'],dtype=np.int32);assert h.shape==(49152,672) and np.all(h>=0)
        np.savez_compressed(cache,histograms=h.reshape(192,256,672))
    known=(labels>0)&(labels<14);starts=known&~np.pad(known[:,:-1],((0,0),(1,0)))
    ends=known&~np.pad(known[:,1:],((0,0),(0,1)));n=starts.sum(-1)
    pix,begin=np.nonzero(starts);p2,end=np.nonzero(ends);assert np.array_equal(pix,p2)
    cumulative=np.pad(np.cumsum(h,axis=1,dtype=np.int64),((0,0),(1,0)))
    counts=cumulative[pix,end+1]-cumulative[pix,begin]
    widths=end-begin+1;peak=np.array([a+int(np.argmax(h[p,a:b+1])) for p,a,b in zip(pix,begin,end)],dtype=np.int32)
    twopix=np.flatnonzero(n==2);first=np.searchsorted(pix,twopix);second=first+1
    eligible=(begin[second]-end[first]>=8)&(twopix//256>=4)&(twopix//256<188)&(twopix%256>=4)&(twopix%256<252)
    ids=twopix[eligible];i=first[eligible];j=second[eligible]
    weak=np.minimum(counts[i],counts[j]);strong=np.maximum(counts[i],counts[j])
    summary=dict(status='DEVELOPMENT real annotation inventory; no method result or complete hidden-surface truth claim.',
        protocol_sha256=manifest['protocol_sha256'],pixels_with_known_windows={str(k):int((n==k).sum()) for k in range(int(n.max())+1)},
        eligible_two_window_interior_pixels=int(len(ids)),known_class_ids=sorted(int(v) for v in np.unique(labels) if 0<v<14),
        unknown14pixels=int((labels==14).any(-1).sum()),
        raw_photon_quantiles=np.quantile(h.sum(-1),[0,.1,.5,.9,1]).tolist(),
        known_window_width_quantiles=np.quantile(widths,[0,.1,.5,.9,1]).tolist(),
        weak_window_counts_quantiles=np.quantile(weak,[0,.1,.5,.9,1]).tolist(),
        strong_window_counts_quantiles=np.quantile(strong,[0,.1,.5,.9,1]).tolist(),
        first_window_weaker_fraction=float(np.mean(counts[i]<counts[j])),
        label_peak_separation_quantiles=np.quantile(peak[j]-peak[i],[0,.1,.5,.9,1]).tolist(),
        derived_histogram_sha256=sha(cache),warnings=[
            'Reference intervals come from the released histogram-peeling labels, not independent range measurements.',
            'Unknown14andUnlabeled0are excluded from known references, never treated as physically empty bins.',
            'A supplied IRF/profile and deviceP1 timing model need checking before measured-data likelihood claims.',
            'The annotation tool uses297ps while the global dataset card says750ps; evaluation in integer bins avoids inventing metric calibration.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(OUT/'reference_windows.npz',pixel=pix,start=begin,end=end,full_histogram_peak=peak,
        observed_window_counts=counts,window_count=n,eligible_two_window_pixels=ids)
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    axes[0].imshow(np.log1p(h.sum(-1)).reshape(192,256),cmap='gray');axes[0].set_title('Log recorded photons')
    axes[1].imshow(n.reshape(192,256),vmin=0,vmax=3,cmap='viridis');axes[1].set_title('Known annotated intervals / pixel')
    for ax in axes[:2]:ax.set_axis_off()
    if len(ids):
        chosen=int(ids[0]);which=np.flatnonzero(pix==chosen)
        axes[2].step(np.arange(672),h[chosen],where='mid',color='#333333',lw=.8)
        for k in which:axes[2].axvspan(begin[k],end[k],color='#176b94',alpha=.2)
        axes[2].set(xlabel='Time bin',ylabel='Recorded photons / bin',title=f'First eligible pixel ({chosen//256},{chosen%256})')
    fig.savefig(OUT/'multireturn_inventory.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
