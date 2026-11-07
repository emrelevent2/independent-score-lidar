"""Independent-cycle validation of nonlocal candidate returns.

Empirically calibrated development method. No exact p-value claim is made for
the plug-in validation score. Candidate extraction and validation are separate;
the selected delay must never be fitted on the validation half.
"""
import numpy as np
from scipy.ndimage import convolve1d
from pilot import coates
from screen import detect_scores


def propose(flux):
    score=detect_scores(flux)
    secondary=np.argmax(score,axis=-1)
    valid=np.isfinite(np.max(score,axis=-1))
    sigma=5/(2*np.sqrt(2*np.log(2)))
    t=np.arange(-10,11); g=np.exp(-.5*(t/sigma)**2); g/=g.sum()
    primary=np.argmax(convolve1d(flux,g,axis=-1,mode='constant'),axis=-1)
    return primary,secondary,valid


def validate(histogram,cycles,primary,secondary,valid,primary_consistency=False):
    """One-pixel score, with both delays supplied by the other data half.

    Background and primary amplitude are nuisance estimates from the validation
    data; secondary location is fixed. Signal is evaluated without spatially
    borrowing validation photons from neighboring pixels.
    """
    x=coates(histogram,cycles)
    t=np.arange(x.shape[-1]); sigma=5/(2*np.sqrt(2*np.log(2)))
    def pulse(delay):
        d=t-delay[...,None]
        g=np.exp(-.5*(d/sigma)**2)
        return g/g.sum(-1,keepdims=True)
    g1=pulse(primary); g2=pulse(secondary)
    background_mask=(np.abs(t-primary[...,None])>12)&(np.abs(t-secondary[...,None])>12)
    n=background_mask.sum(-1,keepdims=True)
    # Small rate floor avoids division by zero in all-zero control bins.
    background=((x*background_mask).sum(-1,keepdims=True)+.5/cycles)/n
    fit_mask=np.abs(t-secondary[...,None])>7
    norm=(g1*g1*fit_mask).sum(-1,keepdims=True)
    amplitude=np.maximum(((x-background)*g1*fit_mask).sum(-1,keepdims=True)/np.maximum(norm,1e-15),0)
    null=background+amplitude*g1
    remaining=cycles*np.exp(-(np.cumsum(null,axis=-1)-null))
    v=np.expm1(null)/np.maximum(remaining,1)
    numerator=((x-null)*g2).sum(-1)
    denominator=np.sqrt(np.maximum((g2*g2*v).sum(-1),1e-20))
    z=numerator/denominator
    if primary_consistency:
        support=np.arange(-10,11); g=np.exp(-.5*(support/sigma)**2); g/=g.sum()
        measured_primary=np.argmax(convolve1d(x,g,axis=-1,mode='constant'),axis=-1)
        valid=valid&(np.abs(measured_primary-primary)<=2)
    return np.where(valid,z,-np.inf)


def crossfit_scores(left_h,right_h,left_flux,right_flux,cycles=500,primary_consistency=False):
    p_a,s_a,v_a=propose(left_flux); p_b,s_b,v_b=propose(right_flux)
    z_ab=validate(right_h,cycles,p_a,s_a,v_a,primary_consistency)
    z_ba=validate(left_h,cycles,p_b,s_b,v_b,primary_consistency)
    return np.stack([z_ab,z_ba],axis=-1),np.stack([s_a,s_b],axis=-1)


def fuse_agreeing(z,delays,tolerance=2):
    """Development fusion; empirical recalibration required.

    Agreement is data dependent, so summing scores does not create a standard
    normal statistic or a guaranteed p-value. Retain separate directional
    alternatives and add the pooled evidence only when delays agree.
    """
    agree=(np.abs(delays[...,0]-delays[...,1])<=tolerance)&np.isfinite(z).all(-1)
    pooled=np.where(agree,(z[...,0]+z[...,1])/np.sqrt(2),-np.inf)
    pooled_delay=np.mean(delays,axis=-1)
    return np.concatenate([z,pooled[...,None]],axis=-1),np.concatenate([delays,pooled_delay[...,None]],axis=-1)
