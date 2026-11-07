"""Same confidence construction; robust outward bracketing of tail roots.

The frozen v1 sometimes returned the conservative endpoint 1 after 256 ULP
steps at N=100000. Here the proposal step grows geometrically and remains
Arb-checked. No unchecked root or stronger statistical assumption is used.
"""
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from scipy.stats import beta
from flint import arb
from calibration_confidence import (binomial_cdf,window_counts,leakage_ratio_upper,_fraction_arb)
from envelope_evidence import outward_float


@lru_cache(maxsize=16384)
def binomial_endpoint(k,n,delta=Fraction(1,3000),upper=True):
    if int(n)!=n or int(k)!=k or not 0<=k<=n:raise ValueError('Invalid binomial data')
    n,k=int(n),int(k);delta=Fraction(delta)
    if not 0<delta<1:raise ValueError('Invalid confidence budget')
    if upper and (n==0 or k==n):return 1.
    if not upper and (n==0 or k==0):return 0.
    proposal=float(beta.ppf(1-float(delta),k+1,n-k) if upper
                   else beta.ppf(float(delta),k,n-k+1))
    if not math.isfinite(proposal):return 1. if upper else 0.
    direction=1. if upper else 0.;p=float(np.nextafter(proposal,direction))
    for j in range(32):
        tail=binomial_cdf(k,n,p) if upper else 1-binomial_cdf(k-1,n,p)
        if tail<=_fraction_arb(delta):return p
        step=np.finfo(float).eps*2**j
        p=min(1.,proposal+step) if upper else max(0.,proposal-step)
    return direction


def lobe_confidence(h,cycles,delta=Fraction(1,1000),cap=.25,shift=16,
                    main=(27,37),dark=(128,255),delay=(31.5,32.5),width=(4.5,5.5)):
    h=np.asarray(h);bins=len(h);delta=Fraction(delta)
    if not (0<delta<1 and 0<=cap<=1 and isinstance(shift,int) and shift>0):raise ValueError('Invalid calibration settings')
    lobe=(main[0]+shift,main[1]+shift)
    if not(main[1]<lobe[0] and lobe[1]<dark[0]):raise ValueError('Separated ordered windows required')
    counts={name:window_counts(h,cycles,*w) for name,w in [('main',main),('lobe',lobe),('dark',dark)]}
    p0=binomial_endpoint(*counts['main'],delta/3,False)
    p1=binomial_endpoint(*counts['lobe'],delta/3,True)
    pd=binomial_endpoint(*counts['dark'],delta/3,True)
    result=dict(eta_upper=float(cap),delta=str(delta),cap=float(cap),cycles=int(cycles),detected=int(h.sum()),
        window_counts={k:list(v) for k,v in counts.items()},
        probability_endpoints=dict(main_lower=p0,lobe_upper=p1,dark_upper=pd),informative=False,
        scope='model-conditional one-sided calibration confidence; geometric Arb-checked endpoint bracketing')
    if p1>=1 or pd>=1:return result
    main_l=-(-arb(p0)).log1p();lobe_u=-(-arb(p1)).log1p();dark_u=-(-arb(pd)).log1p()
    denominator=main_l-dark_u*arb(main[1]-main[0]+1)/(dark[1]-dark[0]+1)
    if not denominator>0:return result
    leakage=leakage_ratio_upper(bins,shift,tuple(main),tuple(delay),tuple(width))
    upper=outward_float(lobe_u*(1+arb(float(cap))*arb(leakage))/denominator)
    result.update(eta_upper=min(float(cap),upper),informative=upper<float(cap),
        leakage_ratio_upper=leakage,main_signal_hazard_lower=outward_float(denominator,False))
    return result
