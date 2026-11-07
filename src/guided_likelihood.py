"""Strong control: donor-guided secondary search, query-fitted nuisances.

This is a numerical likelihood-improvement score requiring calibration. It
does not inherit the proper-mixture e-value guarantee. Secondary centers may
refine within two bins of the donor proposal; primary fits share the ordinary
local-likelihood control's bounds and response family.
"""
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from likelihood_baseline import pulse_derivatives
from adaptive_evidence import feasible_local_null


def guided_likelihood_report(h,secondary,cycles=1000,null_witness=None):
    h=np.asarray(h);t=np.arange(len(h));failures=cycles-np.cumsum(h)
    if np.any(h<0) or h.sum()>cycles or not 0<=secondary<len(h):raise ValueError('Invalid histogram/proposal')
    null=feasible_local_null(h,cycles) if null_witness is None else null_witness
    a,b,d1,w=null['parameters'];d2=float(secondary)
    def objective(q):
        a,c,b,p,s,w=q;g1,dg1,wg1=pulse_derivatives(t,p,w);g2,dg2,wg2=pulse_derivatives(t,s,w)
        rate=(a*g1+c*g2+b/len(t))/cycles
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        with np.errstate(over='ignore'):derivative=h/np.expm1(rate)-failures
        jac=[g1,g2,np.ones(len(t))/len(t),a*dg1,c*dg2,a*wg1+c*wg2]
        return -float(ell),-np.asarray([np.sum(derivative*x/cycles) for x in jac])
    bounds=[(0,None),(0,None),(1e-10,None),(max(0,d1-2.),min(len(t)-1,d1+2.)),
        (max(0,d2-2.),min(len(t)-1,d2+2.)),(4.5,5.5)]
    fits=[]
    for offset in [-1.,0.,1.]:
        s=float(np.clip(d2+offset,*bounds[4]))
        coeff,_=nnls(np.stack([pulse_derivatives(t,d1,w)[0],pulse_derivatives(t,s,w)[0],np.ones(len(t))/len(t)],axis=-1),h.astype(float))
        initial=[max(a,coeff[0]),max(1e-8,coeff[1]),max(1e-8,coeff[2]),d1,s,w]
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
        fits.append(fit)
    best=min(fits,key=lambda x:x.fun);improvement=max(0.,float(-best.fun-null['lower']))
    if improvement<1e-9:improvement=0.
    return dict(score=improvement,delay=float(best.x[4]),primary=float(best.x[3]),
        null_log_likelihood_lower=null['lower'],alternative_log_likelihood=-float(best.fun),
        parameters=best.x.tolist(),optimizer_failures=sum(not f.success for f in fits),
        calibration_required=True,secondary_proposal=d2)
