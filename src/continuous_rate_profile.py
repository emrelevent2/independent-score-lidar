"""Certified concave relaxation of a continuous-time partial log likelihood.

For PROVIDED exact phase exposures E and envelopes f_lo<=f<=f_hi,bound
sum S log(a*f+B/K)-E*(a*f+B/K) over ALL nonnegative a,B. This is a
conditional algebraic certificate,not a certificate of raw timestamp
arithmetic,envelope construction,physical detector model or a complete test.
Concave tangent bounds are established optimization tools.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize
from flint import arb,ctx


def outward_upper(v):return math.nextafter(float(v.upper()),math.inf)


def validate(s,e,lo,hi):
    raw=np.asarray(s)
    if np.any(~np.isfinite(raw)) or np.any(raw<0) or np.any(raw!=np.floor(raw)):raise ValueError('Integer counts required')
    s=np.asarray(s,dtype=np.int64);e=np.asarray(e,dtype=float);lo=np.asarray(lo,dtype=float);hi=np.asarray(hi,dtype=float)
    if not (s.shape==e.shape==lo.shape==hi.shape) or s.ndim!=1:raise ValueError('Shape mismatch')
    if np.any(s<0) or np.any(~np.isfinite(e)) or np.any(e<0) or np.any(lo<0) or np.any(hi<lo) or np.any(~np.isfinite(hi)) or np.any(~np.isfinite(lo)):raise ValueError('Invalid profile inputs')
    if np.any((s>0)&(e<=0)):raise ValueError('Positive count requires positive exposure')
    return s,e,lo,hi


def numeric_value_gradient(s,e,lo,hi,a,b):
    k=len(s);l=a*lo+b/k;h=a*hi+b/k;target=np.divide(s,e,out=np.zeros_like(e),where=e>0);lam=np.clip(target,l,h)
    positive=s>0
    if np.any(lam[positive]<=0):return -np.inf,np.array([np.inf,np.inf])
    value=np.dot(s[positive],np.log(lam[positive]))-np.dot(e,lam)
    upper=h<target;lower=l>target;active=upper|lower;deriv=np.zeros_like(e)
    deriv[active]=s[active]/np.maximum(lam[active],1e-300)-e[active]
    # Zero-count terms are -E*lower_endpoint,including exact lower_endpoint0.
    deriv[~positive]=-e[~positive];shape=np.where(upper,hi,lo)
    return float(value),np.array([np.dot(deriv,shape),deriv.sum()/k])


def certified_value_gradient(s,e,lo,hi,a,b):
    aa=arb(float(a));bb=arb(float(b));k=len(s);value=arb(0);ga=arb(0);gb=arb(0);uncertain=0
    for count,exposure,flo,fhi in zip(s,e,lo,hi):
        n=arb(int(count));ee=arb(float(exposure));fl=arb(float(flo));fh=arb(float(fhi));l=aa*fl+bb/k;h=aa*fh+bb/k
        if count==0:
            value-=ee*l;ga-=ee*fl;gb-=ee/k;continue
        target=n/ee
        if h<target:
            value+=n*h.log()-ee*h;deriv=n/h-ee;ga+=deriv*fh;gb+=deriv/k
        elif l>target:
            value+=n*l.log()-ee*l;deriv=n/l-ee;ga+=deriv*fl;gb+=deriv/k
        elif l<=target and target<=h:
            value+=n*target.log()-n
        else:
            # Only a rounding-indeterminate clipping boundary. The saturated
            # value is an upper bound;hull all possible endpoint gradients.
            value+=n*target.log()-n;uncertain+=1
            al=(n/l-ee)*fl;ah=(n/h-ee)*fh;bl=(n/l-ee)/k;bh=(n/h-ee)/k
            ga+=arb(0).union(al).union(ah);gb+=arb(0).union(bl).union(bh)
    return value,ga,gb,uncertain


def saturated_upper(s,e):
    result=arb(0)
    for n,x in zip(s,e):
        if n:result+=arb(int(n))*(arb(int(n))/arb(float(x))).log()-int(n)
    return result


def profile_upper(s,e,lo,hi,initial=None):
    """Global in a,B without caps: require both tangent slopes nonpositive.

    If g_a,g_B<=0,the affine upper tangent attains its maximum on the positive
    orthant at(0,0). Value-g_a*a-g_B*B bounds the full unbounded nuisance domain.
    Numerical optimization only proposes a tangent location;Arb checks it.
    """
    s,e,lo,hi=validate(s,e,lo,hi);k=len(s);scale=max(1,int(s.sum()))
    if ctx.prec<128:raise RuntimeError('At least128-bit Arb precision required')
    if s.sum()==0:return dict(upper=0.,method='zero_count_supremum',tangent_certified=True,scope='Provided exact exposures/envelopes only')
    if initial is None:
        rates=np.divide(s,e,out=np.zeros_like(e),where=e>0);b=max(1e-5,float(np.median(rates))*k);a=max(1e-5,float(rates.sum()-b));initial=[a,b]
    def objective(x):
        v,g=numeric_value_gradient(s,e,lo,hi,*x);return -v/scale,-g/scale
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(1e-12,None)],options=dict(ftol=1e-13,gtol=1e-11,maxiter=200))
    a,b=np.maximum(fit.x,[0.,1e-12]);sat=saturated_upper(s,e);step=max(1.,a,b)*1e-10;certificate=None
    for iteration in range(64):
        v,ga,gb,uncertain=certified_value_gradient(s,e,lo,hi,a,b)
        if ga<=0 and gb<=0:
            bound=v-ga*arb(float(a))-gb*arb(float(b));certificate=bound;break
        a+=step;b+=step;step*=4
    bound=sat if certificate is None else certificate.min(sat)
    return dict(upper=outward_upper(bound),method='orthant_concave_tangent' if certificate is not None else 'saturated_fallback',tangent_certified=certificate is not None,parameters=[float(a),float(b)],certified_slopes=[str(ga),str(gb)],uncertain_clipping_bins=uncertain,nudges=iteration,numeric_optimizer_success=bool(fit.success),scope='Global nonnegative unbounded amplitude/background for provided exact dyadic exposures and envelope endpoints. No raw-data,physical,envelope-construction or complete-test guarantee.')
