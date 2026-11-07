"""Location-only alternatives; never alter evidence or existence decisions.

Both constructions are standard decision/estimation tools,not new statistical
principles. The mixture's numerical posterior is a working distribution,
not a calibrated location confidence set.
"""
import numpy as np
from scipy.special import logsumexp,xlogy
from scipy.optimize import minimize


def mixture_weights(h,models,cycles=1000):
    rates=models['rates'];h=np.asarray(h);remaining=cycles-np.cumsum(h)
    ell=np.log(-np.expm1(-rates))@h-rates@remaining
    logw=ell+np.log([float(w) for w in models['weights']])
    return np.exp(logw-logsumexp(logw))


def interval_mass(center,global_locations,global_mass,spatial_location,spatial_mass,radius=2.):
    # Global grid values represent uniform unit-width cells. This avoids the
    # artificial inclusion of five grid atoms in a width-four interval.
    centers=np.atleast_1d(center).astype(float)
    lo=np.maximum(centers[:,None]-radius,np.asarray(global_locations)[None,:]-.5)
    hi=np.minimum(centers[:,None]+radius,np.asarray(global_locations)[None,:]+.5)
    mass=np.clip(hi-lo,0.,1.)@np.asarray(global_mass)
    mass+=spatial_mass*(np.abs(centers-spatial_location)<=radius)
    return mass


def window_location(h,models,legacy,radius=2.,cycles=1000):
    if not(radius>0 and np.isfinite(radius)):raise ValueError('Positive tolerance required')
    p=mixture_weights(h,models,cycles);spatial=models['spatial'];locations=models['locations']
    grid=np.array(models['global_delays'],float)
    mass=np.array([p[(~spatial)&(locations==d)].sum() for d in grid])
    sd=float(locations[np.flatnonzero(spatial)[0]]);sm=float(p[spatial].sum())
    # Window mass is piecewise linear between these breakpoints,with only
    # the spatial-atom jump. Thus a maximum occurs at a breakpoint (or any
    # point of a flat interval). Legacy is included for stable tie handling.
    edges=np.r_[grid-.5,grid+.5]
    centers=np.unique(np.r_[edges-radius,edges+radius,sd-radius,sd+radius,float(legacy)])
    centers=centers[(centers>=0)&(centers<=models['rates'].shape[1]-1)]
    values=interval_mass(centers,grid,mass,sd,sm,radius);maximum=float(values.max())
    tied=np.flatnonzero(values>=maximum-1e-13)
    chosen=min(tied,key=lambda i:(abs(centers[i]-legacy),centers[i]))
    return dict(secondary=float(centers[chosen]),working_window_mass=float(values[chosen]),
        old_working_window_mass=float(interval_mass(legacy,grid,mass,sd,sm,radius)[0]),
        radius=float(radius),global_smoothing='uniform unit-width grid cells',
        scope='Working-posterior window-loss location only; no location coverage guarantee')


def two_value_gradient(z,h,eta,cycles=1000):
    a,c,b,d,s,w=map(float,z);t=np.arange(len(h));k=4*np.log(2)
    # Use the SAME finite-bin normalization as shoulder_alternative.pulse.
    def pulse(center):
        g=np.exp(-k*((t-center)/w)**2);side=np.exp(-k*((t-(center+16))/w)**2)
        total=g.sum();shape=(g+eta*side)/total
        dg=2*k*(t-center)/w**2*g;ds=2*k*(t-(center+16))/w**2*side
        wg=2*k*(t-center)**2/w**3*g;ws=2*k*(t-(center+16))**2/w**3*side
        dd=(dg+eta*ds)/total-shape*dg.sum()/total
        dw=(wg+eta*ws)/total-shape*wg.sum()/total
        return shape,dd,dw
    g,gd,gw=pulse(d);q,qd,qw=pulse(s)
    rate=(a*g+c*q+b/len(h))/cycles;fail=cycles-np.cumsum(h)
    value=np.sum(xlogy(h,-np.expm1(-rate))-fail*rate)
    with np.errstate(over='ignore'):v=h/np.expm1(rate)-fail
    columns=[g,q,np.ones(len(h))/len(h),a*gd,c*qd,a*gw+c*qw]
    return float(value),np.array([np.sum(v*x/cycles) for x in columns])


def profile_location(h,proposal,eta,legacy,head,cycles=1000):
    if head!='global':return dict(secondary=float(legacy),used=False,reason='Spatial proposal retained')
    bins=len(h);d=float(proposal['primary']);w=float(proposal['fitted_fwhm'])
    lower=[0.,0.,1e-10,max(0.,d-1.5),max(4.,float(legacy)-5.5),4.5]
    upper=[np.inf,np.inf,np.inf,min(bins-1.,d+1.5),min(bins-4.,float(legacy)+5.5),5.5]
    # Keep secondary and primary separated by the original admissibility8.
    if legacy>d:lower[4]=max(lower[4],upper[3]+8.)
    else:upper[4]=min(upper[4],lower[3]-8.)
    if lower[4]>=upper[4]:return dict(secondary=float(legacy),used=False,reason='Empty local admissible interval')
    best=None
    for c in [8.,16.,32.]:
        x=np.array([proposal['primary_incident'],c,proposal['background_incident'],d,legacy,w])
        x=np.minimum(np.maximum(x,np.array(lower)+1e-9),np.array(upper)-1e-9)
        def loss(z):
            value,grad=two_value_gradient(z,np.asarray(h),eta,cycles);return -value,-grad
        fit=minimize(loss,x,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),options={'maxiter':200,'ftol':1e-12,'gtol':1e-7})
        if np.all(np.isfinite(fit.x)) and np.isfinite(fit.fun) and (best is None or fit.fun<best.fun):best=fit
    if best is None:return dict(secondary=float(legacy),used=False,reason='Numerical fallback')
    return dict(secondary=float(best.x[4]),used=True,parameters=best.x.tolist(),optimizer_success=bool(best.success),log_likelihood=float(-best.fun),
        scope='Local numerical two-pulse first-event likelihood fit; no global optimum or location coverage claim; donor primary remains reported')
