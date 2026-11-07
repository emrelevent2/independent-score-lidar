"""Inventory of the fixed annotated development pair, not method testing."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'stb_semantic_development';OUT=ROOT/'results'/'stb_semantic_inventory_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=json.loads((DATA/'download_manifest.json').read_text());OUT.mkdir(exist_ok=True)
    for item in manifest['files']:assert sha(DATA/item['local'])==item['sha256']
    labels_file=next(DATA/i['local'] for i in manifest['files'] if i['local'].endswith('.npy'))
    hist_file=next(DATA/i['local'] for i in manifest['files'] if i['local'].endswith('.txt'))
    labels=np.load(labels_file,allow_pickle=False,mmap_mode='r');assert labels.shape==(49152,672) and labels.dtype==np.uint8
    cached=DATA/'frame_histograms.npz'
    if cached.exists():h=np.load(cached)['histograms']
    else:
        h=np.loadtxt(hist_file,dtype=np.int32);assert h.shape==(49152,672) and np.all(h>=0)
        np.savez_compressed(cached,histograms=h.reshape(192,256,672))
    h=h.reshape(49152,672)
    mask=labels>0;starts=mask&~np.pad(mask[:,:-1],((0,0),(1,0)),constant_values=False)
    ends=mask&~np.pad(mask[:,1:],((0,0),(0,1)),constant_values=False)
    windows_per_pixel=starts.sum(-1);pix,begin=np.nonzero(starts);last_pix,end=np.nonzero(ends);assert np.array_equal(pix,last_pix)
    cumulative=np.pad(np.cumsum(h,axis=1,dtype=np.int64),((0,0),(1,0)))
    counts=cumulative[pix,end+1]-cumulative[pix,begin]
    np.savez_compressed(OUT/'annotated_windows.npz',pixel=pix,start=begin,end=end,observed_counts=counts,window_count=windows_per_pixel)
    positive_class_ids=np.unique(labels).tolist();totals=h.sum(-1)
    summary=dict(status='DEVELOPMENT inventory only; no method accuracy or independent hidden-surface truth claim.',
        data_protocol_sha256=manifest['protocol_sha256'],annotation_classes=positive_class_ids,
        annotated_window_count=int(len(pix)),annotated_pixels=int((windows_per_pixel>0).sum()),
        pixels_by_windows={str(k):int((windows_per_pixel==k).sum()) for k in range(int(windows_per_pixel.max())+1)},
        raw_photon_quantiles=np.quantile(totals,[0,.1,.5,.9,1]).tolist(),
        annotated_window_photon_quantiles=np.quantile(counts,[0,.1,.5,.9,1]).tolist(),
        raw_photons_inside_annotations=float(h[mask].sum()/h.sum()),
        derived_histogram_sha256=sha(cached),no_poisson_or_first_event_calibration_claim=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    ids=np.flatnonzero(windows_per_pixel==2);selected=None
    for i in ids:
        indices=np.flatnonzero(pix==i)
        if (end[indices[0]]+8<begin[indices[1]] and (counts[indices]>0).all()
            and 4<=i//256<188 and 4<=i%256<252):selected=int(i);break
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    axes[0].imshow(np.log1p(totals).reshape(192,256),cmap='gray');axes[0].set_title('Log recorded photons')
    axes[1].imshow(windows_per_pixel.reshape(192,256),vmin=0,vmax=3,cmap='viridis');axes[1].set_title('Annotated disjoint intervals')
    for ax in axes[:2]:ax.set_axis_off()
    if selected is not None:
        axes[2].step(np.arange(672),h[selected],where='mid',lw=.8,color='#333333')
        for idx in np.flatnonzero(pix==selected):axes[2].axvspan(begin[idx],end[idx],alpha=.2,color='#176b94')
        axes[2].set_title(f'First separated two-interval pixel ({selected//256},{selected%256})')
        axes[2].set(xlabel='Time bin',ylabel='Recorded photons / bin')
    fig.savefig(OUT/'annotation_inventory.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
