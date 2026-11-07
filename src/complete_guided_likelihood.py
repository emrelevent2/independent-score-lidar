"""Additional strong comparator: full primary range and multiple primary starts.

This closes the nuisance-start limitation of the simpler guided control when
noise makes the true secondary brighter than the primary in the queried pixel.
No frozen original source is modified. This remains an empirically calibrated
numerical likelihood score, not an analytic e-value.
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from likelihood_baseline import pulse_derivatives
from adaptive_evidence import feasible_local_null


def complete_guided_likelihood(h,primary_donor,secondary_donor,cycles=1000):
    h=np.asarray(h);t=np.arange(len(h));failures=cycles-np.cumsum(h);null=feasible_local_null(h,cycles)
    if np.any(h<0) or h.sum()>cycles:raise ValueError('Invalid first-event histogram')
    kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2);kernel/=kernel.sum()
    matched=convolve1d(h.astype(float),kernel,mode='constant')
    peaks=(matched>=np.roll(matched,1))&(matched>np.roll(matched,-1))
    starts=np.argsort(np.where(peaks,matched,-np.inf))[-3:][::-1]
    primary_starts=sorted(set([float(x) for x in starts]+[float(primary_donor),null['parameters'][2]]))
    secondary_bounds=(max(0,secondary_donor-2.),min(len(h)-1,secondary_donor+2.))
    def loss(q):
        a,c,b,d1,d2,w=q;g1,dg1,wg1=pulse_derivatives(t,d1,w);g2,dg2,wg2=pulse_derivatives(t,d2,w)
        rate=(a*g1+c*g2+b/len(t))/cycles
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        with np.errstate(over='ignore'):derivative=h/np.expm1(rate)-failures
        jac=[g1,g2,np.ones(len(t))/len(t),a*dg1,c*dg2,a*wg1+c*wg2]
        return -float(ell),-np.asarray([np.sum(derivative*x/cycles) for x in jac])
    fits=[]
    for d1 in primary_starts:
        for offset in [-1.,0.,1.]:
            d2=float(np.clip(secondary_donor+offset,*secondary_bounds));w=null['parameters'][3]
            coef,_=nnls(np.stack([pulse_derivatives(t,d1,w)[0],pulse_derivatives(t,d2,w)[0],np.ones(len(t))/len(t)],axis=-1),h.astype(float))
            initial=[max(coef[0],1e-8),max(coef[1],1e-8),max(coef[2],1e-8),d1,d2,w]
            fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(0,None),(1e-10,None),
                (0,len(h)-1),secondary_bounds,(4.5,5.5)],options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
            fits.append(fit)
    best=min(fits,key=lambda f:f.fun);score=max(0.,float(-best.fun-null['lower']))
    if score<1e-9:score=0.
    return dict(score=score,delay=float(best.x[4]),primary=float(best.x[3]),parameters=best.x.tolist(),
        null_log_likelihood_lower=null['lower'],alternative_log_likelihood=-float(best.fun),
        optimizer_failures=sum(not f.success for f in fits),primary_starts=primary_starts,
        primary_range=[0,len(h)-1],secondary_proposal=secondary_donor,calibration_required=True)
