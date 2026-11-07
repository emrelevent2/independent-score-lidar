"""Training-free candidate and ablations. Experimental, not a validated detector.

Use the development protocol for tuning. Output calibration is mandatory.
"""
import numpy as np
from scipy.ndimage import convolve1d, uniform_filter
import pywt
from pilot import coates, wavelet_baseline


def remaining_and_variance(h, cycles=1000):
    """Jeffreys-stabilized conditional delta variance for Coates hazards.

    Leading order covariance between hazards vanishes under independent cycles;
    this does not make the finite-count corrected samples exactly independent.
    Never use this approximation as a guaranteed confidence bound.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h<0) or np.any(h.sum(-1)>cycles):
        raise ValueError('Invalid single-detection histogram')
    remaining=cycles-np.cumsum(h,axis=-1)+h
    q=(h+0.5)/(remaining+1)
    variance=q/(np.maximum(remaining,1)*(1-q))
    return remaining, variance


def spatial_pool(x, variance, size=3):
    """Periodic box mean; variance uses squared coefficients exactly.

    Only intended for arrays at least size by size. Periodic boundary behavior
    is explicit and shared by pooled controls; real-data reporting must exclude
    the wrapped border or implement a sensor-appropriate boundary treatment.
    """
    if min(x.shape[:2])<size: raise ValueError('Spatial input too small')
    return (uniform_filter(x,size=(size,size,1),mode='wrap'),
            uniform_filter(variance,size=(size,size,1),mode='wrap')/(size*size))


def spline_filters(levels):
    """Effective filters for undecimated symmetric binomial smoothing.

    Detail = previous effective smoothing minus current smoothing. Returning
    effective filters allows variance propagation without ignoring overlap.
    """
    previous=np.ones(1)
    result=[]
    for level in range(levels):
        stride=2**level
        step=np.zeros(2*stride+1); step[[0,stride,2*stride]]=[.25,.5,.25]
        current=np.convolve(previous,step)
        pad=(len(current)-len(previous))//2
        detail=np.pad(previous,(pad,pad))-current
        result.append((current,detail))
        previous=current
    return result


def firm(d, threshold):
    a=np.abs(d)
    return np.sign(d)*np.where(a<=threshold,0,
                        np.where(a>=2*threshold,a,2*(a-threshold)))


def candidate(h, cycles=1000, levels=3, z=2.5, heteroscedastic=True,
              shrink='firm', pooling=True):
    x=coates(h,cycles)
    _,v=remaining_and_variance(h,cycles)
    if not heteroscedastic:
        v=np.broadcast_to(np.median(v,axis=-1,keepdims=True),v.shape).copy()
    if pooling: x,v=spatial_pool(x,v)
    filters=spline_filters(levels)
    out=convolve1d(x,filters[-1][0],axis=-1,mode='wrap')
    for _,d_filter in filters:
        d=convolve1d(x,d_filter,axis=-1,mode='wrap')
        # This uses the complete effective filter, not a sum of correlated
        # approximation variances from intermediate levels.
        vd=convolve1d(v,d_filter*d_filter,axis=-1,mode='wrap')
        th=z*np.sqrt(np.maximum(vd,0))
        if shrink=='firm': d=firm(d,th)
        elif shrink=='soft': d=np.sign(d)*np.maximum(np.abs(d)-th,0)
        elif shrink=='hard': d=np.where(np.abs(d)>th,d,0)
        else: raise ValueError(shrink)
        out=out+d
    return np.maximum(out,0)


def anscombe_wavelet(h, cycles=1000, levels=3):
    """Classical variance-stabilized control; approximate inverse, not exact.

    Processes raw histograms before Coates inversion. First-detection pileup
    violates the Poisson assumption; include as a conventional comparator.
    """
    a=2*np.sqrt(np.asarray(h,dtype=float)+3/8)
    coefficients=pywt.swt(a,'haar',level=levels,axis=-1)
    th=np.sqrt(2*np.log(h.shape[-1]))
    shrunk=[(ca,pywt.threshold(cd,th,'soft')) for ca,cd in coefficients]
    den=pywt.iswt(shrunk,'haar',axis=-1)
    counts=np.maximum((den/2)**2-3/8,0)
    # Shrinkage inverse can increase total counts; preserve histogram validity.
    total=counts.sum(-1,keepdims=True)
    counts=counts*np.minimum(1,(cycles-1)/np.maximum(total,1))
    return coates(counts,cycles)


def all_methods(h):
    raw=coates(h)
    return {'raw':raw,
            'box3':uniform_filter(raw,size=(3,3,1),mode='wrap'),
            'haar_mad':wavelet_baseline(raw),
            'anscombe_haar':anscombe_wavelet(h),
            'candidate':candidate(h),
            'candidate_constant_variance':candidate(h,heteroscedastic=False),
            'candidate_soft':candidate(h,shrink='soft'),
            'candidate_no_pool':candidate(h,pooling=False)}


def block_wiener_candidate(h, cycles=1000, levels=3, energy_size=5):
    """Revision 2: estimate coefficient energy before shrinking amplitudes.

    This is a development candidate. Local Wiener shrinkage is established
    prior art; the first-detection variance model is the research component.
    The energy window uses spatial correlation without averaging coefficients
    across a depth boundary. It still can bias low-count returns.
    """
    x=coates(h,cycles)
    _,v=remaining_and_variance(h,cycles)
    fs=spline_filters(levels)
    out=convolve1d(x,fs[-1][0],axis=-1,mode='wrap')
    for _,df in fs:
        d=convolve1d(x,df,axis=-1,mode='wrap')
        vd=convolve1d(v,df*df,axis=-1,mode='wrap')
        total=uniform_filter(d*d,size=(energy_size,energy_size,3),mode='wrap')
        noise=uniform_filter(vd,size=(energy_size,energy_size,3),mode='wrap')
        signal=np.maximum(total-noise,0)
        gain=signal/np.maximum(signal+vd,1e-20)
        out=out+gain*d
    return np.maximum(out,0)


def pooled_wavelet_proposer(h,cycles=500,spatial_size=9,levels=3,z=1.5,exclude_center=False):
    """Coarse, shift-invariant wavelet proposal only; not a pixelwise estimate.

    Broad pooling is allowed to share candidate delays. Independent per-pixel
    validation must decide whether those proposed returns actually exist.
    """
    x=coates(h,cycles)
    _,v=remaining_and_variance(h,cycles)
    if exclude_center:
        if spatial_size<=1 or min(x.shape[:2])<spatial_size: raise ValueError('Invalid neighbor support')
        k=spatial_size*spatial_size
        x0=x; v0=v
        x=(uniform_filter(x0,(spatial_size,spatial_size,1),mode='wrap')*k-x0)/(k-1)
        v=np.maximum(uniform_filter(v0,(spatial_size,spatial_size,1),mode='wrap')*k-v0,0)/(k-1)**2
    else: x,v=spatial_pool(x,v,size=spatial_size)
    fs=spline_filters(levels)
    out=convolve1d(x,fs[-1][0],axis=-1,mode='wrap')
    for _,df in fs:
        d=convolve1d(x,df,axis=-1,mode='wrap')
        vd=convolve1d(v,df*df,axis=-1,mode='wrap')
        out+=firm(d,z*np.sqrt(np.maximum(vd,0)))
    return np.maximum(out,0)


def depth_guided_pool(h,cycles=1000,spatial_size=3,range_sigma=2.):
    """Established bilateral-style control using estimated primary depth.

    It is intentionally a strong simple comparator: avoiding foreground depth
    mixing must not be credited as new merely because plain box pooling fails.
    """
    x=coates(h,cycles)
    t=np.arange(-10,11); sigma=5/(2*np.sqrt(2*np.log(2)))
    g=np.exp(-.5*(t/sigma)**2); g/=g.sum()
    delay=np.argmax(convolve1d(x,g,axis=-1,mode='constant'),axis=-1)
    out=np.zeros_like(x); total=np.zeros(x.shape[:2])
    radius=spatial_size//2
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            donor=np.roll(x,(dy,dx),axis=(0,1))
            donor_delay=np.roll(delay,(dy,dx),axis=(0,1))
            weight=np.exp(-.5*((delay-donor_delay)/range_sigma)**2)
            out+=weight[...,None]*donor; total+=weight
    return out/total[...,None]
