"""Exploratory development data, never a held-out publication benchmark."""
from pathlib import Path
import json
import hashlib
import platform
import time
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]


def first_detection_probabilities(incident_per_cycle):
    lam = np.asarray(incident_per_cycle, dtype=float)
    if np.any(lam < 0):
        raise ValueError('Negative incident rate')
    survival = np.exp(-np.r_[0., np.cumsum(lam)[:-1]])
    bins = survival * (-np.expm1(-lam))
    return np.r_[bins, np.exp(-lam.sum())]


def simulate(seed, secondary=3., size=16, bins=256, cycles=1000,
             primary=30., background=20., separation=12., offset=0.,
             primary_delay_map=None,secondary_mask=None,secondary_delay_map=None,
             pulse_fwhm=5.):
    rng = np.random.default_rng(seed)
    t = np.arange(bins)
    sigma = pulse_fwhm / (2*np.sqrt(2*np.log(2)))
    def pulse(center):
        g = np.exp(-0.5*((t-center)/sigma)**2)
        return g/g.sum()
    h = np.empty((size, size, bins), dtype=np.int32)
    truth = np.empty_like(h, dtype=float)
    # Weak return occupies a square, allowing spatial boundary errors to appear.
    mask = np.zeros((size, size), dtype=bool)
    mask[size//4:3*size//4, size//4:3*size//4] = True
    if secondary_mask is not None:
        mask=np.asarray(secondary_mask,dtype=bool).copy()
        if mask.shape!=(size,size): raise ValueError('Invalid secondary mask')
    centers=np.full((size,size),80+offset) if primary_delay_map is None else np.asarray(primary_delay_map)
    if centers.shape!=(size,size): raise ValueError('Invalid primary delay map')
    secondary_centers=centers+separation if secondary_delay_map is None else np.asarray(secondary_delay_map)
    if secondary_centers.shape!=(size,size): raise ValueError('Invalid secondary delay map')
    for y in range(size):
        for x in range(size):
            incident = primary*pulse(centers[y,x]) + background/bins
            if mask[y,x]:
                incident = incident + secondary*pulse(secondary_centers[y,x])
            truth[y,x] = incident/cycles
            h[y,x] = rng.multinomial(cycles, first_detection_probabilities(incident/cycles))[:-1]
    return h, truth, mask


def coates(h, cycles=1000):
    h = np.asarray(h, dtype=float)
    n = np.maximum(cycles, h.sum(-1, keepdims=True)+1)
    total = np.cumsum(h, axis=-1)
    before = total-h
    return np.log((n-before)/(n-total))


def wavelet_baseline(flux):
    # Deliberately simple reference, not the proposed method.
    spatial = uniform_filter(flux, size=(3,3,1), mode='reflect')
    # Undecimated two-tap Haar averaging and residual details, periodic boundary.
    approximation=spatial.copy()
    details=[]
    for level in range(3):
        smooth=(approximation+np.roll(approximation,2**level,axis=-1))/2
        details.append(approximation-smooth)
        approximation=smooth
    detail = details[0]
    mad = np.median(np.abs(detail-np.median(detail,axis=-1,keepdims=True)),axis=-1,keepdims=True)/0.67448975
    threshold = mad*np.sqrt(2*np.log(flux.shape[-1]))
    recovered=approximation
    for d in details:
        recovered=recovered+np.sign(d)*np.maximum(np.abs(d)-threshold,0)
    return np.maximum(recovered,0)


def self_check():
    p = first_detection_probabilities([0.1,0.2,0.3])
    assert np.isclose(p.sum(),1)
    assert np.isclose(p[0],1-np.exp(-0.1))
    assert np.isclose(p[1],np.exp(-0.1)*(1-np.exp(-0.2)))
    assert np.isclose(p[-1],np.exp(-0.6))
    # No-pileup approximation must agree in the low-rate limit.
    lam = np.array([1e-8,2e-8,3e-8])
    assert np.allclose(first_detection_probabilities(lam)[:-1],lam,rtol=1e-6,atol=0)
    n=1000000
    expected=n*p[:-1]
    assert np.allclose(coates(expected,n),[0.1,0.2,0.3])
    h,_,_=simulate(601)
    h2,_,_=simulate(601)
    assert np.array_equal(h,h2) and np.all(h.sum(-1)<=1000)
    assert np.isfinite(coates(h)).all()
    assert np.isfinite(wavelet_baseline(coates(h))).all()


def main():
    self_check()
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    metadata={'status':'exploratory development only','seed_family':601,
              'cycles':1000,'shape':[16,16,256],'primary_center':80,
              'secondary_center':92,'fwhm_bins':5,'primary_incident_photons':30,
              'background_incident_photons':20,'secondary_incident_photons':3,
              'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'wavelet_baseline':'undecimated Haar averaging residual soft threshold, three levels',
              'checks':'first-detection probabilities, sum, low-flux limit, Coates inversion, seed repeatability, finite outputs passed'}
    for name,secondary in [('null',0.),('two_returns',3.)]:
        h,truth,mask=simulate(601,secondary)
        path=data/f'pilot_{name}.mat'
        savemat(path,{'hst_map_set':h.astype(float),'truth':truth,'weak_mask':mask})
        metadata[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        start=time.perf_counter(); raw=coates(h); wave=wavelet_baseline(raw)
        np.savez_compressed(results/f'pilot_{name}_baselines.npz',raw=raw,wavelet=wave,truth=truth,weak_mask=mask)
        metadata[name+'_raw_wavelet_seconds']=time.perf_counter()-start
    (results/'pilot_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
