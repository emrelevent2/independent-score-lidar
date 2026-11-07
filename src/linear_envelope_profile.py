"""Arb-checked concave upper profile for arbitrary nonnegative rate bases.

Input matrix endpoints must enclose the physical coefficient vectors. The
optimizer never certifies a bound. A returned saturated fallback is valid;
matrix midpoints are NOT asserted to be feasible physical nuisance models.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from flint import arb
from envelope_evidence import _observations,outward_float


def value_gradient(theta,h,remaining,lo,hi,scale):
    lower=(lo@theta)/scale;upper=(hi@theta)/scale
    optimum=np.full(len(h),np.inf)
    np.log1p(np.divide(h,remaining,out=np.zeros(len(h),float),where=remaining>0),
             out=optimum,where=remaining>0)
    optimum[(h==0)&(remaining==0)]=0.
    rate=np.clip(optimum,lower,upper)
    value=np.sum(xlogy(h,-np.expm1(-rate))-remaining*rate)
    with np.errstate(over='ignore',invalid='ignore'):
        derivative=np.divide(h,np.expm1(rate),out=np.zeros(len(h)),where=rate>0)-remaining
    active=np.where((optimum<lower)[:,None],lo,np.where((optimum>upper)[:,None],hi,0.))/scale
    return float(value),np.sum(derivative[:,None]*active,axis=0)


def certify_tangent(theta,h,remaining,lo,hi,scale):
    theta=[arb(float(x)) for x in theta];gradient=[arb(0) for _ in theta];phi=arb(0)
    for count,fail,lower_vector,upper_vector in zip(h,remaining,lo,hi):
        count,fail=int(count),int(fail)
        dl=[arb(float(x))/scale for x in lower_vector];du=[arb(float(x))/scale for x in upper_vector]
        low=sum((a*b for a,b in zip(theta,dl)),arb(0));high=sum((a*b for a,b in zip(theta,du)),arb(0))
        if count==0 and fail==0:continue
        if fail==0:rate,active=high,du
        else:
            optimum=(arb(count)/fail).log1p() if count else arb(0)
            if optimum<low:rate,active=low,dl
            elif optimum>high:rate,active=high,du
            elif optimum>=low and optimum<=high:rate,active=optimum,None
            else:return None
        if count:phi+=count*(-(-rate).expm1()).log()
        phi-=fail*rate
        derivative=arb(count)/rate.expm1()-fail if count else arb(-fail)
        if active is not None:
            for j,x in enumerate(active):gradient[j]+=derivative*x
    if all(x<=0 for x in gradient):
        return outward_float(phi-sum((a*b for a,b in zip(gradient,theta)),arb(0)))
    return None


def profile_upper(h,lo,hi,cycles,initial):
    h,remaining,scale,saturated=_observations(h,cycles,'first')
    lo=np.asarray(lo,float);hi=np.asarray(hi,float);initial=np.asarray(initial,float)
    if not(lo.shape==hi.shape==(len(h),len(initial)) and np.all(np.isfinite(lo)) and
           np.all(np.isfinite(hi)) and np.all(lo>=0) and np.all(hi>=lo) and np.all(np.isfinite(initial)) and np.all(initial>=0)):
        raise ValueError('Finite nonnegative matrix enclosure and initialization required')
    def loss(theta):
        value,gradient=value_gradient(theta,h,remaining,lo,hi,scale);return -value,-gradient
    bounds=[(0,None)]*(len(initial)-1)+[(1e-10,None)]
    fit=minimize(loss,np.maximum(initial,[0.]*(len(initial)-1)+[1e-10]),jac=True,method='L-BFGS-B',bounds=bounds,
                 options={'ftol':1e-12,'gtol':1e-8,'maxiter':160})
    theta=np.maximum(fit.x,[0.]*(len(initial)-1)+[1e-10]);certificate=None
    for k in range(16):
        certificate=certify_tangent(theta*(1+1e-6*2**k),h,remaining,lo,hi,scale)
        if certificate is not None:break
    return dict(upper=saturated if certificate is None else min(saturated,certificate),
                tangent_certified=certificate is not None,optimizer_success=bool(fit.success),theta=theta.tolist())
