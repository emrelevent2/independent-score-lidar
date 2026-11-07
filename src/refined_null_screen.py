"""Reject-only refinement of a feasible null point; no global MLE assertion."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from flint import arb
from envelope_evidence import exact_gaussian,certified_log_likelihood


def witness(query,proposal,eta,cap=None,cycles=1000):
    h=np.asarray(query);bins=len(h);t=np.arange(bins);fail=cycles-np.cumsum(h);k=4*np.log(2)
    if not(0<=eta<=.25 and (cap is None or 0<=eta<=cap<=.25)):raise ValueError('Invalid response range')
    d=float(proposal['primary']);w=float(proposal['fitted_fwhm']);g=np.exp(-k*((t-d)/w)**2)
    initial=[max(proposal['primary_incident']/g.sum(),1e-8),max(proposal['background_incident'],1e-8),d,w]
    if cap is not None:initial.append(eta)
    bounds=[(0,None),(1e-10,None),(0,bins-1),(4.5,5.5)]+([(0,cap)] if cap is not None else [])
    def objective(z):
        a,b,d,w=z[:4];e=z[4] if cap is not None else eta
        g=np.exp(-k*((t-d)/w)**2);s=np.exp(-k*((t-(d+16))/w)**2);shape=g+e*s
        rates=(a*shape+b/bins)/cycles
        value=np.sum(xlogy(h,-np.expm1(-rates))-fail*rates)
        with np.errstate(over='ignore'):v=h/np.expm1(rates)-fail
        dg=2*k*(t-d)/w**2*g;ds=2*k*(t-(d+16))/w**2*s
        wg=2*k*(t-d)**2/w**3*g;ws=2*k*(t-(d+16))**2/w**3*s
        columns=[shape,np.ones(bins)/bins,a*(dg+e*ds),a*(wg+e*ws)]
        if cap is not None:columns.append(a*s)
        return -float(value),-np.array([np.sum(v*x/cycles) for x in columns])
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':200,'ftol':1e-12,'gtol':1e-8})
    z=np.array(fit.x if np.all(np.isfinite(fit.x)) else initial)
    for i,(low,high) in enumerate(bounds):z[i]=max(low,z[i]);z[i]=min(high,z[i]) if high is not None else z[i]
    a,b,d,w=map(float,z[:4]);e=float(z[4]) if cap is not None else eta
    g=exact_gaussian(bins,d,w,False);s=exact_gaussian(bins+16,arb(d)+16,w,False)[:bins]
    rates=[(arb(a)*(x+arb(float(e))*y)+arb(b)/bins)/cycles for x,y in zip(g,s)]
    return dict(lower=certified_log_likelihood(h,rates,cycles,'first'),parameters=[a,b,d,w,float(e)],optimizer_success=bool(fit.success),
        scope='Feasible null point only; query data allowed in denominator; cannot authorize acceptance')
