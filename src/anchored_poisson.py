"""Query-primary anchored empirical donor ablation, not independent evidence.

Only an empirical extension: the query primary selects compatible neighbors.
Original donor-only and certified implementations remain unchanged.
"""
import numpy as np
from scipy.optimize import nnls,least_squares
from real_poisson_methods_v2 import _matched,_initial,_peaks,guided_report


def anchored_proposal(h,pulse,row,col,null,depth_sigma=8.):
    y,x=np.mgrid[-4:5,-4:5];keep=(x!=0)|(y!=0);x,y=x[keep],y[keep]
    donors=np.asarray(h[row-4:row+5,col-4:col+5][keep],float)
    primary=np.argmax(_matched(donors,pulse),axis=-1);anchor=null['parameters'][2]
    logw=-.5*((x*x+y*y)/1.5**2+(primary-anchor)**2/depth_sigma**2)
    weights=np.exp(logw-logw.max());weights/=weights.sum();pooled=weights@donors;t=np.arange(len(pooled))
    coef,_=nnls(np.stack([pulse.derivatives(t,anchor,8.)[0],np.ones(len(t))/len(t)],axis=-1),pooled)
    residual=pooled-coef[0]*pulse.derivatives(t,anchor,8.)[0]-coef[1]/len(t)
    scores=_matched(residual,pulse);allowed=(np.abs(t-anchor)>=8)&(t>=4)&(t<len(t)-4)
    starts=_peaks(np.where(allowed,scores,-np.inf));fits=[]
    for s in starts:
        initial=_initial(pooled,pulse,anchor,8.,s)
        lower=np.array([0.,0.,1e-9,max(0,anchor-2),max(0,s-2),3.])
        upper=np.array([np.inf,np.inf,np.inf,min(len(t)-1,anchor+2),min(len(t)-1,s+2),16.])
        initial=np.minimum(np.maximum(initial,lower+1e-8),upper-1e-8)
        def model(q):
            a,c,b,d1,d2,w=q;return a*pulse.derivatives(t,d1,w)[0]+c*pulse.derivatives(t,d2,w)[0]+b/len(t)
        fit=least_squares(lambda q:model(q)-pooled,initial,bounds=(lower,upper),max_nfev=140,
            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        fits.append(fit)
    fit=min(fits,key=lambda f:np.sum(f.fun**2));a,c,b,d1,d2,w=fit.x
    g=pulse.derivatives(t,d2,w)[0];neff=float(1/np.sum(weights**2));variance=(c+b/(len(t)*np.sum(g*g)))/neff
    return dict(primary=float(d1),secondary=float(d2),primary_count=float(a),secondary_count=float(c),
        background_count=float(b),width=float(w),effective_donors=neff,support_proxy=float(c/np.sqrt(max(variance,1e-12))),
        raw_center_excluded=True,query_primary_used=True,optimizer_success=bool(fit.success))


def evaluate_anchored(h,pulse,row,col,base,depth_sigma=8.):
    q=anchored_proposal(h,pulse,row,col,base['null'],depth_sigma)
    guided=guided_report(h[row,col],pulse,q,null=base['null'])
    return dict(proposal=q,reports=dict(local=base['reports']['local'],guided=guided),null=base['null'])
