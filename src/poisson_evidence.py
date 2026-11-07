"""Measured-IRF prototype for the independent-Poisson observation model.

This is separate from the first-detection experiment: unknown acquisition
cycles are not invented for the public free-running measurement histograms.
The null is one known IRF plus uniform background at any integer delay.
"""
import numpy as np
from scipy.ndimage import correlate1d,uniform_filter
from scipy.optimize import minimize,nnls
from scipy.special import xlogy


def irf_templates(irf,bins):
    irf=np.asarray(irf,float)
    if irf.ndim!=1 or np.any(irf<0) or irf.sum()<=0:
        raise ValueError('Invalid IRF')
    peak=int(np.argmax(irf)); t=np.arange(bins)
    g=np.stack([np.interp(t-d+peak,np.arange(len(irf)),irf,left=0.,right=0.)
                for d in range(bins)])
    return g/g.sum(-1,keepdims=True)


def poisson_log_likelihood(h,mean):
    h=np.asarray(h,float);mean=np.asarray(mean,float)
    if np.any(h<0) or np.any(mean<0) or h.shape!=mean.shape:
        raise ValueError('Invalid Poisson arrays')
    return float(np.sum(xlogy(h,mean)-mean))


def poisson_single_bound(h,pulse):
    h=np.asarray(h,float)
    design=np.stack([pulse,np.ones(len(h))/len(h)],axis=-1)
    initial=np.array([max(h.sum()*.7,1.),max(h.sum()*.3,1.)])
    def value_grad(theta):
        mean=design@theta
        ell=np.sum(xlogy(h,mean)-mean)
        grad=design.T@(h/mean-1)
        return -float(ell),-grad
    fit=minimize(value_grad,initial,jac=True,method='L-BFGS-B',
                 bounds=[(0,None),(1e-10,None)],
                 options={'ftol':1e-13,'gtol':1e-9,'maxiter':150})
    theta=np.maximum(fit.x,[0.,1e-10])
    saturated=float(np.sum(xlogy(h,h)-h))
    for k in range(30):
        point=theta*(1.+1e-7*2.**k)
        n_ell,n_grad=value_grad(point);grad=-n_grad
        if np.all(grad<=-1e-12):
            upper=min(saturated,-n_ell-float(grad@point)+1e-8)
            return dict(upper=upper,lower=-float(fit.fun),fallback=False,
                        optimizer_success=bool(fit.success))
    return dict(upper=saturated,lower=-float(fit.fun),fallback=True,
                optimizer_success=bool(fit.success))


def poisson_null_bound(h,templates):
    fits=[poisson_single_bound(h,g) for g in templates]
    upper=max(f['upper'] for f in fits);lower=max(f['lower'] for f in fits)
    return dict(upper=upper,best_feasible=lower,global_gap=upper-lower,
                fallback_count=sum(f['fallback'] for f in fits),
                failed_optimizers=sum(not f['optimizer_success'] for f in fits))


def poisson_neighbor_alternative(h,irf,size=9):
    """Measured IRF is a calibration input; candidate/nuisances use donors."""
    if size%2!=1 or size<3 or size>min(h.shape[:2]):raise ValueError('Invalid donor support')
    x=np.asarray(h,float); k=size*size
    pooled=np.maximum((uniform_filter(x,(size,size,1),mode='wrap')*k-x)/(k-1),0.)
    irf=np.asarray(irf,float);irf=irf/irf.sum()
    origin=int(np.argmax(irf))-len(irf)//2
    response=correlate1d(pooled,irf,axis=-1,mode='constant',origin=origin)
    p=np.argmax(response,axis=-1)
    g=irf_templates(irf,x.shape[-1]); t=np.arange(x.shape[-1])
    alt=np.empty_like(x);secondary=np.empty(p.shape,int)
    separation=len(irf)
    for y in range(x.shape[0]):
        for col in range(x.shape[1]):
            primary=g[p[y,col]]
            design=np.stack([primary,np.ones(x.shape[-1])/x.shape[-1]],axis=-1)
            coeff,_=nnls(design,pooled[y,col])
            residual=pooled[y,col]-design@coeff
            score=correlate1d(residual,irf,mode='constant',origin=origin)
            valid=(np.abs(t-p[y,col])>=separation)&(t>=len(irf))&(t<x.shape[-1]-len(irf))
            s=int(np.argmax(np.where(valid,score,-np.inf)))
            secondary[y,col]=s
            design=np.stack([primary,g[s],np.ones(x.shape[-1])/x.shape[-1]],axis=-1)
            coeff,_=nnls(design,pooled[y,col])
            alt[y,col]=np.maximum(design@coeff,1e-12)
    return alt,p,secondary
