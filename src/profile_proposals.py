"""Development: separate local existence fitting from spatial localization.

These are empirically calibrated scores, never analytic e-values. Fixed
positions are supplied exclusively by the donor predictor. Free amplitudes,
background and pulse width are fitted on the queried histogram.
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from likelihood_baseline import pulse_derivatives
from adaptive_evidence import feasible_local_null


def profile_at_proposals(h,primary,secondary,mode='fixed_secondary',cycles=1000,null_witness=None):
    if mode not in ['fixed_secondary','fixed_both']:raise ValueError('Unknown profile mode')
    h=np.asarray(h);t=np.arange(len(h));failures=cycles-np.cumsum(h)
    if np.any(h<0) or h.sum()>cycles or not(0<=primary<len(h) and 0<=secondary<len(h)):raise ValueError('Invalid histogram/positions')
    null=feasible_local_null(h,cycles) if null_witness is None else null_witness
    def objective(q):
        if mode=='fixed_secondary':a,c,b,d1,w=q
        else:a,c,b,w=q;d1=primary
        g1,dg1,wg1=pulse_derivatives(t,d1,w);g2,_,wg2=pulse_derivatives(t,secondary,w)
        rate=(a*g1+c*g2+b/len(t))/cycles
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        with np.errstate(over='ignore'):derivative=h/np.expm1(rate)-failures
        jac=[g1,g2,np.ones(len(t))/len(t)]
        if mode=='fixed_secondary':jac.append(a*dg1)
        jac.append(a*wg1+c*wg2)
        return -float(ell),-np.asarray([np.sum(derivative*x/cycles) for x in jac])
    if mode=='fixed_secondary':
        kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2);kernel/=kernel.sum()
        matched=convolve1d(h.astype(float),kernel,mode='constant')
        maxima=(matched>=np.roll(matched,1))&(matched>np.roll(matched,-1))
        starts=np.argsort(np.where(maxima,matched,-np.inf))[-3:]
        starts=sorted(set([float(x) for x in starts]+[float(primary),null['parameters'][2]]))
        pairs=[(d,null['parameters'][3]) for d in starts]
        bounds=[(0,None),(0,None),(1e-10,None),(0,len(h)-1),(4.5,5.5)]
    else:
        pairs=[(primary,w) for w in [4.6,5.,5.4]]
        bounds=[(0,None),(0,None),(1e-10,None),(4.5,5.5)]
    fits=[]
    for d,w in pairs:
        coef,_=nnls(np.stack([pulse_derivatives(t,d,w)[0],pulse_derivatives(t,secondary,w)[0],np.ones(len(t))/len(t)],axis=-1),h.astype(float))
        initial=[max(coef[0],1e-8),max(coef[1],1e-8),max(coef[2],1e-8)]
        if mode=='fixed_secondary':initial.append(d)
        initial.append(w)
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
        fits.append(fit)
    best=min(fits,key=lambda f:f.fun);score=max(0.,float(-best.fun-null['lower']))
    if score<1e-9:score=0.
    d1=float(best.x[3]) if mode=='fixed_secondary' else float(primary)
    return dict(score=score,delay=float(secondary),primary=d1,parameters=best.x.tolist(),
        null_log_likelihood_lower=null['lower'],alternative_log_likelihood=-float(best.fun),
        optimizer_failures=sum(not f.success for f in fits),mode=mode,
        location_source='Donor secondary position, not refitted using the queried photons.',calibration_required=True)
