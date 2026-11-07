"""Nuisance-projected secondary-return score for close-return evaluation.

Classical linear projection, not a new reconstruction method or an exact
likelihood ratio. Empirical calibration remains necessary after denoising.
"""
import numpy as np
from scipy.ndimage import convolve1d


def secondary_scores(flux,min_separation=3.):
    x=np.asarray(flux,dtype=float)
    n=x.shape[-1]; t=np.arange(n); sigma=5/(2*np.sqrt(2*np.log(2)))
    support=np.arange(-10,11)
    g=np.exp(-.5*(support/sigma)**2); g/=g.sum()
    mf=convolve1d(x,g,axis=-1,mode='constant')
    peak=np.argmax(mf,axis=-1)
    left=np.take_along_axis(mf,np.maximum(peak-1,0)[...,None],axis=-1)[...,0]
    middle=np.take_along_axis(mf,peak[...,None],axis=-1)[...,0]
    right=np.take_along_axis(mf,np.minimum(peak+1,n-1)[...,None],axis=-1)[...,0]
    curvature=left-2*middle+right
    offset=np.clip(.5*(left-right)/np.where(np.abs(curvature)>1e-20,curvature,-1),-.5,.5)
    delay=peak+offset
    dt=t-delay[...,None]
    primary=np.exp(-.5*(dt/sigma)**2)
    primary/=primary.sum(-1,keepdims=True)
    derivative=primary*dt/(sigma*sigma)
    basis=np.stack([np.ones_like(primary),primary,derivative],axis=-1)
    gram=np.einsum('...ti,...tj->...ij',basis,basis)
    inverse=np.linalg.pinv(gram)
    rhs=np.einsum('...ti,...t->...i',basis,x)
    coefficients=np.einsum('...ij,...j->...i',inverse,rhs)
    residual=x-np.einsum('...ti,...i->...t',basis,coefficients)
    numerator=convolve1d(residual,g,axis=-1,mode='constant')
    inner=convolve1d(basis,g,axis=-2,mode='constant')
    correction=np.einsum('...ti,...ij,...tj->...t',inner,inverse,inner)
    norm=np.maximum(np.sum(g*g)-correction,1e-20)
    score=numerator/np.sqrt(norm)
    valid=(np.abs(dt)>=min_separation)&(t>=10)&(t<n-10)
    maxima=(score>=np.roll(score,1,axis=-1))&(score>np.roll(score,-1,axis=-1))
    return np.where(valid&maxima,score,-np.inf)
