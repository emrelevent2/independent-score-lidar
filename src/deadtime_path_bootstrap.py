"""Unknown-flux development bootstrap on full dead-time event paths.

Parametric bootstrap and one-step nuisance fitting are established methods.
Plug-in p-values here are approximate,not finite-sample composite-null tests.
The optional Numba kernel changes execution only;random numbers come from
an explicit NumPy Generator outside the kernel and no fastmath is enabled.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from deadtime_exposure import sufficient_statistics
from deadtime_scan import pulse,fit_null,likelihood_parts,rounded_opportunities
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        return lambda f:f


@njit(cache=True)
def events_from_exponentials(rates,cycles,dead,draws):
    k=len(rates);prefix=np.empty(k+1);prefix[0]=0.
    for j in range(k):prefix[j+1]=prefix[j]+rates[j]
    mass=prefix[-1];end=cycles*k;available=0.;out=np.empty(len(draws));n=0
    for draw in draws:
        if available>=end:break
        cycle=np.floor(available/k);phase=available-cycle*k;j=min(int(phase),k-1)
        target=cycle*mass+prefix[j]+rates[j]*(phase-j)+draw
        c=np.floor(target/mass);v=target-c*mass
        lo=0;hi=k
        while lo<hi:
            mid=(lo+hi)//2
            if prefix[mid+1]<=v:lo=mid+1
            else:hi=mid
        j=min(lo,k-1);t=c*k+j+(v-prefix[j])/rates[j]
        if t>=end:break
        out[n]=t;n+=1;available=t+dead
    else:
        if available<end:raise ValueError('Insufficient exponential draws')
    return out[:n]


def simulate(rates,cycles,dead,seed):
    if dead<=0 or np.any(np.asarray(rates)<=0):raise ValueError('Positive rates and dead time required')
    draws=np.random.default_rng(seed).exponential(size=int(cycles*len(rates)/dead)+3)
    return events_from_exponentials(np.asarray(rates,dtype=float),cycles,float(dead),draws)


@lru_cache(maxsize=16)
def fine_kernels(k,sigma,subdivision):
    templates=np.array([pulse(j/subdivision,k,sigma)[0] for j in range(subdivision)])
    return np.conj(np.fft.fft(templates,axis=1)),np.conj(np.fft.fft(templates**2,axis=1))


def fine_correlations(rows,kernels):
    rows=np.atleast_2d(rows)
    v=np.fft.ifft(np.fft.fft(rows,axis=1)[:,None,:]*kernels[None,:,:],axis=2).real
    return v.transpose(0,2,1).reshape(len(rows),-1)


def fixed_fit_scores(counts,exposures,theta,sigma,mode='duration',subdivision=4):
    """Batched nuisance-projected scan at a fixed generating/observed null fit."""
    s=np.atleast_2d(counts).astype(float);e=np.atleast_2d(exposures).astype(float);k=s.shape[1]
    a,tau,b=np.asarray(theta);f,d=pulse(tau,k,sigma);lam=np.maximum(a*f+b/k,1e-14)
    if mode=='ded':u=s/np.expm1(lam)-(e-s);w=e/np.expm1(lam)
    else:u=s/lam-e;w=e/lam
    j=np.column_stack([f,a*d,np.ones(k)/k]);ff,ff2=fine_kernels(k,sigma,subdivision)
    raw=fine_correlations(u,ff);info=fine_correlations(w,ff2)
    cross=np.stack([fine_correlations(w*j[:,i],ff) for i in range(3)],axis=2)
    fisher=np.einsum('ki,bk,kj->bij',j,w,j);inv=np.linalg.pinv(fisher,rcond=1e-12)
    raw-=np.einsum('bmi,bij,bj->bm',cross,inv,u@j)
    info-=np.einsum('bmi,bij,bmj->bm',cross,inv,cross)
    grid=np.arange(k*subdivision)/subdivision;distance=np.abs((grid-tau+k/2)%k-k/2)
    allowed=(distance[None,:]>=2.5*sigma)&(info>1e-8)
    scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(info,1e-8),-np.inf)
    idx=np.argmax(scores,axis=1)
    return scores[np.arange(len(s)),idx],grid[idx]


def fine_scan(s,e,sigma,mode='duration',fit=None,subdivision=4):
    fit=fit_null(s,e,sigma,mode) if fit is None else fit
    score,delay=fixed_fit_scores(s,e,fit['theta'],sigma,mode,subdivision)
    return dict(score=float(score[0]),delay=float(delay[0]),primary=float(fit['theta'][1]),fit=fit)


def refit_from_start(s,e,theta,sigma,mode):
    """Optimize all three null parameters from the generating null fit.

    Numerical local MLE,not certified global. Warm start is shared with the
    one-step control;source-style three-start fits can audit selected draws.
    """
    s=np.asarray(s,dtype=float);e=np.asarray(e,dtype=float);k=len(s);scale=max(1.,s.sum());a,t,b=theta
    def objective(x):
        a,t,b=x;f,d=pulse(t,k,sigma);ll,u,_=likelihood_parts(s,e,a*f+b/k,mode)
        j=np.column_stack([f,a*d,np.ones(k)/k])
        return -ll/scale,-j.T@u/scale
    result=minimize(objective,theta,jac=True,method='L-BFGS-B',bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=120))
    x=result.x.copy();x[1]%=k
    return dict(theta=x,loglik=float(-result.fun*scale),optimizer_success=bool(result.success),evaluations=int(result.nfev))


def one_step(s,e,theta,sigma,mode):
    """Damped Fisher step,with likelihood-increase backtracking;not a new idea."""
    theta=np.asarray(theta,dtype=float);k=len(s);a,t,b=theta;f,d=pulse(t,k,sigma);lam=a*f+b/k
    ll,u,w=likelihood_parts(s,e,lam,mode);j=np.column_stack([f,a*d,np.ones(k)/k])
    delta=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12)@(j.T@u)
    for n in range(21):
        x=theta+delta*2.**(-n);x[0]=np.clip(x[0],0,20);x[2]=np.clip(x[2],1e-6,30);x[1]=np.clip(x[1],t-3*sigma,t+3*sigma)
        f,_=pulse(x[1],k,sigma);candidate=likelihood_parts(s,e,x[0]*f+x[2]/k,mode)[0]
        if candidate>=ll-1e-10:
            x[1]%=k
            return dict(theta=x,loglik=candidate,backtracks=n)
    return dict(theta=theta.copy(),loglik=ll,backtracks=21)


def bootstrap_paths(theta,sigma,cycles,dead,bins,replicates,seed,mode):
    f,_=pulse(theta[1],bins,sigma);rates=theta[0]*f+theta[2]/bins
    ss=[];ee=[];aa=[];hash_parts=[]
    for i in range(replicates):
        events=simulate(rates,cycles,dead,seed+1009*i);s,e=sufficient_statistics(events,cycles,bins,dead)
        ss.append(s);ee.append(e)
        if mode=='ded':aa.append(rounded_opportunities(events,cycles,bins,dead))
        hash_parts.append(events.tobytes())
    return np.asarray(ss),np.asarray(aa if mode=='ded' else ee),hash_parts


def pvalue(observed,bootstrap):
    # Unlike the previous known-regime MC test,this plug-in procedure has no
    # finite-sample exchangeability guarantee under an unknown composite null.
    return float((1+np.count_nonzero(np.asarray(bootstrap)>=observed))/(len(bootstrap)+1))
