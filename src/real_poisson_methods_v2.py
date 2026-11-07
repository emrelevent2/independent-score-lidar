"""Working-count likelihood for real histogram comparisons.

No cycle count is inferred. These numerical Poisson scores are empirical
benchmark scores, not certified sensor-calibrated tests. A flexible Gaussian
or independently learned empirical pulse describes an effective return.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares,minimize,nnls
from scipy.special import xlogy
from likelihood_baseline import pulse_derivatives as gaussian_derivatives


class WorkingPulse:
    def __init__(self,kind='gaussian',profile=None):
        if kind not in ['gaussian','empirical']:raise ValueError('Unknown pulse family')
        self.kind=kind
        if kind=='empirical':
            v=np.asarray(profile,float)
            if v.ndim!=1 or np.any(v<0) or not np.isfinite(v).all() or v.max()<=0:raise ValueError('Invalid profile')
            self.values=np.r_[0.,v,0.];self.peak=int(np.argmax(self.values))
            active=np.flatnonzero(self.values>=self.values.max()/2)
            self.base_width=float(active[-1]-active[0]+1)

    def derivatives(self,t,delay,width):
        if self.kind=='gaussian':return gaussian_derivatives(t,delay,width)
        scale=width/self.base_width;u=(np.asarray(t)-delay)/scale+self.peak
        raw=np.interp(u,np.arange(len(self.values)),self.values,left=0.,right=0.)
        j=np.floor(u).astype(int);valid=(j>=0)&(j<len(self.values)-1);slope=np.zeros_like(raw)
        slope[valid]=self.values[j[valid]+1]-self.values[j[valid]]
        d=-slope/scale;w=-slope*(np.asarray(t)-delay)/(scale*width)
        total=raw.sum()
        if total<=0:raise ValueError('Pulse outside window')
        g=raw/total
        return g,(d-g*d.sum())/total,(w-g*w.sum())/total

    @lru_cache(maxsize=4)
    def templates(self,bins,width=8.):
        return np.stack([self.derivatives(np.arange(bins),d,width)[0] for d in range(bins)])


def _peaks(x,n=3):
    mask=(x>=np.roll(x,1))&(x>np.roll(x,-1))
    order=np.argsort(np.where(mask,x,-np.inf))[::-1]
    selected=[int(i) for i in order if np.isfinite(np.where(mask,x,-np.inf)[i])][:n]
    return selected or [int(np.argmax(x))]


def _matched(h,pulse):
    g=pulse.templates(h.shape[-1]);centered=h-np.median(h,axis=-1,keepdims=True)
    return centered@g.T/np.sum(g*g,axis=1)


def _initial(h,pulse,d1,width,d2=None):
    t=np.arange(len(h));columns=[pulse.derivatives(t,d1,width)[0]]
    if d2 is not None:columns.append(pulse.derivatives(t,d2,width)[0])
    columns.append(np.ones(len(h))/len(h));coef,_=nnls(np.stack(columns,axis=-1),np.maximum(h,0))
    coef=np.maximum(coef,1e-9)
    return np.r_[coef,d1,width] if d2 is None else np.r_[coef,d1,d2,width]


def _fit(h,pulse,initial,bounds,two=False):
    t=np.arange(len(h));h=np.asarray(h,float)
    def objective(q):
        if two:a,c,b,d1,d2,w=q
        else:a,b,d1,w=q
        g1,dg1,wg1=pulse.derivatives(t,d1,w);mean=a*g1+b/len(t)
        jac=[g1,np.ones(len(t))/len(t),a*dg1,a*wg1]
        if two:
            g2,dg2,wg2=pulse.derivatives(t,d2,w);mean+=c*g2
            jac=[g1,g2,np.ones(len(t))/len(t),a*dg1,c*dg2,a*wg1+c*wg2]
        derivative=h/mean-1
        return -float(np.sum(xlogy(h,mean)-mean)),-np.array([np.sum(derivative*j) for j in jac])
    return minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
        options={'ftol':1e-11,'gtol':1e-7,'maxiter':160,'maxls':40})


def null_fit(h,pulse,width_range=(3.,16.)):
    h=np.asarray(h)
    if h.ndim!=1 or np.any(h<0) or not np.isfinite(h).all():raise ValueError('Invalid count histogram')
    starts=_peaks(_matched(h,pulse));fits=[]
    for d in starts:
        initial=_initial(h,pulse,d,8.)
        bounds=[(0,None),(1e-9,None),(max(0,d-2.),min(len(h)-1,d+2.)),width_range]
        fits.append(_fit(h,pulse,initial,bounds))
    best=min(fits,key=lambda f:f.fun)
    return dict(parameters=best.x.tolist(),log_likelihood=-float(best.fun),starts=starts,
        optimizer_failures=sum(not f.success for f in fits))


def local_report(h,pulse,width_range=(3.,16.),null=None):
    h=np.asarray(h);null=null_fit(h,pulse,width_range) if null is None else null
    a,b,d1,w=null['parameters'];t=np.arange(len(h))
    residual=h-a*pulse.derivatives(t,d1,w)[0]-b/len(h);score=_matched(residual,pulse)
    allowed=(np.abs(t-d1)>=8)&(t>=4)&(t<len(h)-4)
    starts=_peaks(np.where(allowed,score,-np.inf));fits=[]
    for d2 in starts:
        initial=_initial(h,pulse,d1,w,d2)
        bounds=[(0,None),(0,None),(1e-9,None),(max(0,d1-2),min(len(h)-1,d1+2)),
            (max(0,d2-2),min(len(h)-1,d2+2)),width_range]
        fits.append(_fit(h,pulse,initial,bounds,True))
    best=min(fits,key=lambda f:f.fun)
    return dict(score=max(0.,float(-best.fun-null['log_likelihood'])),primary=float(best.x[3]),delay=float(best.x[4]),
        parameters=best.x.tolist(),optimizer_failures=sum(not f.success for f in fits),calibration_required=True)


def guided_report(h,pulse,proposal,width_range=(3.,16.),null=None):
    h=np.asarray(h);null=null_fit(h,pulse,width_range) if null is None else null
    starts=sorted(set(map(float,null['starts']+[proposal['primary'],null['parameters'][2]])))
    sb=(max(0,proposal['secondary']-2),min(len(h)-1,proposal['secondary']+2));fits=[]
    for d1 in starts:
        for offset in [-1.,0.,1.]:
            d2=float(np.clip(proposal['secondary']+offset,*sb));w=null['parameters'][3]
            initial=_initial(h,pulse,d1,w,d2)
            bounds=[(0,None),(0,None),(1e-9,None),(0,len(h)-1),sb,width_range]
            fits.append(_fit(h,pulse,initial,bounds,True))
    best=min(fits,key=lambda f:f.fun)
    return dict(score=max(0.,float(-best.fun-null['log_likelihood'])),primary=float(best.x[3]),delay=float(best.x[4]),
        parameters=best.x.tolist(),optimizer_failures=sum(not f.success for f in fits),calibration_required=True)


def donor_proposal(h,pulse,row,col,width_range=(3.,16.),size=9):
    radius=size//2
    if not(radius<=row<h.shape[0]-radius and radius<=col<h.shape[1]-radius):raise ValueError('Full donor window required')
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1];keep=(x!=0)|(y!=0)
    donors=np.asarray(h[row-radius:row+radius+1,col-radius:col+radius+1][keep],float)
    x,y=x[keep],y[keep];primary=np.argmax(_matched(donors,pulse),axis=-1)
    anchor=float(np.median(primary[(np.abs(x)<=1)&(np.abs(y)<=1)]))
    log_weights=-.5*((x*x+y*y)/1.5**2+(primary-anchor)**2/2.**2)
    weights=np.exp(log_weights-np.max(log_weights));weights/=weights.sum()
    pooled=weights@donors;p=int(np.argmax(_matched(pooled,pulse)));t=np.arange(len(pooled))
    coef,_=nnls(np.stack([pulse.derivatives(t,p,8.)[0],np.ones(len(t))/len(t)],axis=-1),pooled)
    residual=pooled-coef[0]*pulse.derivatives(t,p,8.)[0]-coef[1]/len(t)
    scores=_matched(residual,pulse);allowed=(np.abs(t-p)>=8)&(t>=4)&(t<len(t)-4)
    s=int(np.argmax(np.where(allowed,scores,-np.inf)));initial=_initial(pooled,pulse,p,8.,s)
    lower=np.array([0.,0.,1e-9,max(0,p-2.),max(0,s-2.),width_range[0]])
    upper=np.array([np.inf,np.inf,np.inf,min(len(t)-1,p+2.),min(len(t)-1,s+2.),width_range[1]])
    initial=np.minimum(np.maximum(initial,lower+1e-8),upper-1e-8)
    def model(q):
        a,c,b,d1,d2,w=q;return a*pulse.derivatives(t,d1,w)[0]+c*pulse.derivatives(t,d2,w)[0]+b/len(t)
    fit=least_squares(lambda q:model(q)-pooled,initial,bounds=(lower,upper),max_nfev=140,
        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    a,c,b,d1,d2,w=fit.x;g=pulse.derivatives(t,d2,w)[0];neff=float(1/np.sum(weights**2))
    variance=(c+b/(len(t)*np.sum(g*g)))/neff
    return dict(primary=float(d1),secondary=float(d2),primary_count=float(a),secondary_count=float(c),
        background_count=float(b),width=float(w),effective_donors=neff,support_proxy=float(c/np.sqrt(max(variance,1e-12))),
        raw_center_excluded=True,optimizer_success=bool(fit.success))


def evaluate_query(h,pulse,row,col):
    q=donor_proposal(h,pulse,row,col);null=null_fit(h[row,col],pulse)
    local=local_report(h[row,col],pulse,null=null);guided=guided_report(h[row,col],pulse,q,null=null)
    return dict(proposal=q,reports=dict(local=local,guided=guided,spatial=dict(guided,delay=q['secondary'])),null=null)
