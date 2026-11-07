"""Development second-return scans: empirically calibrated,not certified tests.

Nuisance-projected score and multistart likelihood ratio are classical tools.
FFT implements the integer-delay scan; continuous nuisance delays are fitted.
Deadtime-aware and histogram controls have the same Gaussian pulse family.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from deadtime_exposure import periodic_measure


def pulse(tau,bins,sigma):
    r=np.arange(bins)[:,None]-float(tau)%bins+np.arange(-2,3)[None,:]*bins
    f=(ndtr((r+1)/sigma)-ndtr(r/sigma)).sum(axis=1)
    z=lambda x:np.exp(-.5*(x/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    d=(z(r)-z(r+1)).sum(axis=1)
    return f,d


def rounded_opportunities(events,cycles,bins,dead_bins):
    """Source-style whole-bin opportunity count,including partial active bins.

    Initial active state,dead_bins>=1. This is a discretized control,not the
    exact continuous-duration likelihood of the generating simulator.
    """
    if dead_bins<1:raise ValueError('Rounded one-hit-per-bin control needs dead>=1')
    t=np.asarray(events);lo=np.r_[0.,t+dead_bins];hi=np.r_[t,cycles*bins]
    keep=lo<hi;lo=lo[keep];hi=hi[keep]
    # Continuous simulator has probability zero of integer event timestamps.
    ends=np.where(hi==cycles*bins,hi,np.floor(hi)+1)
    return np.rint(periodic_measure(np.floor(lo),ends,bins)).astype(np.int64)


def likelihood_parts(s,e,lam,mode):
    lam=np.maximum(np.asarray(lam),1e-14)
    if mode=='ded':
        p=-np.expm1(-lam);f=e-s
        return float(np.dot(s,np.log(p))-np.dot(f,lam)),s/np.expm1(lam)-f,e/np.expm1(lam)
    return float(np.dot(s,np.log(lam))-np.dot(e,lam)),s/lam-e,e/lam


def corr(x,f):return np.fft.ifft(np.fft.fft(x)*np.conj(np.fft.fft(f))).real


def fit_null(s,e,sigma,mode='duration'):
    s=np.asarray(s,dtype=float);e=np.asarray(e,dtype=float);k=len(s);scale=max(1.,s.sum())
    def objective(x):
        a,tau,b=x;f,d=pulse(tau,k,sigma);lam=a*f+b/k
        ll,u,_=likelihood_parts(s,e,lam,mode);j=np.column_stack([f,a*d,np.ones(k)/k])
        return -ll/scale,-j.T@u/scale
    f0,_=pulse(0,k,sigma);ratio=s/np.maximum(e,1e-9);sm=corr(ratio,f0)
    seeds=[]
    for i in np.argsort(sm)[::-1]:
        if all(abs((int(i)-v+k/2)%k-k/2)>3*sigma for v in seeds):seeds.append(int(i))
        if len(seeds)==3:break
    b0=max(1e-4,float(np.median(ratio)*k));a0=max(.02,float(ratio.sum()-b0));fits=[]
    for t in seeds:
        res=minimize(objective,[min(a0,10),t,min(b0,20)],jac=True,method='L-BFGS-B',bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=120))
        fits.append(res)
    best=min(fits,key=lambda z:z.fun);x=best.x.copy();x[1]%=k
    return dict(theta=x,loglik=float(-best.fun*scale),optimizer_success=bool(best.success),evaluations=int(sum(v.nfev for v in fits)))


def scan(s,e,sigma,mode='duration',adjusted=True,fit=None):
    s=np.asarray(s,dtype=float);e=np.asarray(e,dtype=float);k=len(s)
    fit=fit_null(s,e,sigma,mode) if fit is None else fit
    a,tau,b=fit['theta'];f,d=pulse(tau,k,sigma);f0,_=pulse(0,k,sigma)
    _,u,w=likelihood_parts(s,e,a*f+b/k,mode);j=np.column_stack([f,a*d,np.ones(k)/k])
    raw=corr(u,f0);info=corr(w,f0*f0);cross=np.column_stack([corr(w*j[:,i],f0) for i in range(3)])
    fisher=j.T@(w[:,None]*j);inv=np.linalg.pinv(fisher,rcond=1e-12)
    if adjusted:
        raw=raw-cross@(inv@(j.T@u));info=info-np.einsum('ij,jk,ik->i',cross,inv,cross)
    allowed=(np.abs((np.arange(k)-tau+k/2)%k-k/2)>=2.5*sigma)&(info>1e-8)
    scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(info,1e-8),-np.inf)
    idx=int(np.argmax(scores))
    return dict(score=float(scores[idx]),delay=float(idx),primary=float(tau),theta=fit['theta'],scores=scores,fit=fit,adjusted=adjusted)


def multistart_glrt(s,e,sigma,score_result):
    """Numerical five-parameter two-pulse control;not certified global optimum.

    Six separated scan peaks initialize local alternative fits. Includes the
    fitted null itself,so a negative numerical likelihood ratio is impossible.
    Primary/secondary shifts each vary continuously around their starts.
    """
    s=np.asarray(s,dtype=float);e=np.asarray(e,dtype=float);k=len(s);scale=max(1.,s.sum())
    base=score_result['fit'];a,t,b=base['theta'];candidates=[]
    for idx in np.argsort(score_result['scores'])[::-1]:
        if not np.isfinite(score_result['scores'][idx]):continue
        if all(abs((int(idx)-v+k/2)%k-k/2)>2*sigma for v in candidates):candidates.append(int(idx))
        if len(candidates)==6:break
    def objective(x):
        a,t,b,c,d=x;f,df=pulse(t,k,sigma);g,dg=pulse(d,k,sigma);lam=a*f+b/k+c*g
        ll,u,_=likelihood_parts(s,e,lam,'duration')
        j=np.column_stack([f,a*df,np.ones(k)/k,g,c*dg])
        return -ll/scale,-j.T@u/scale
    best=dict(loglik=base['loglik'],theta=[a,t,b,0.,score_result['delay']],success=True);evals=0
    for d in candidates:
        res=minimize(objective,[a,t,b,.02,d],jac=True,method='L-BFGS-B',bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30),(0,20),(d-3*sigma,d+3*sigma)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=160));evals+=res.nfev
        if -res.fun*scale>best['loglik']:best=dict(loglik=float(-res.fun*scale),theta=res.x.tolist(),success=bool(res.success))
    x=best['theta'];p,d=x[1]%k,x[4]%k
    if x[3]>x[0]:p,d=d,p
    return dict(score=float(2*(best['loglik']-base['loglik'])),delay=float(d),primary=float(p),theta=x,optimizer_success=best['success'],evaluations=evals)
