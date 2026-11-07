"""Finite-cycle first-event calibration of an optical response lobe.

SciPy quantiles propose endpoints only. Every accepted binomial endpoint is
checked with Arb against its exact rational one-sided tail budget. Three
conditional-binomial window bounds are combined by a union bound. This is a
standard confidence-set construction, not a new coverage theorem.
"""
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from scipy.stats import beta
from flint import arb
from envelope_evidence import gaussian_envelope, outward_float


def _fraction_arb(q):
    q=Fraction(q)
    return arb(q.numerator)/q.denominator


def binomial_cdf(k,n,p):
    """Arb CDF with p the exact supplied IEEE endpoint."""
    if k<0:return arb(0)
    if k>=n:return arb(1)
    if p==0:return arb(1)
    if p==1:return arb(0)
    x=arb(float(p));term=(1-x)**n;total=term
    odds=x/(1-x)
    for j in range(k):
        term*=arb(n-j)/(j+1)*odds
        total+=term
    return total


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
    for _ in range(256):
        tail=binomial_cdf(k,n,p) if upper else 1-binomial_cdf(k-1,n,p)
        if tail<=_fraction_arb(delta):return p
        p=float(np.nextafter(p,direction))
    return direction  # Conservative endpoint, never an unchecked quantile.


def window_counts(h,cycles,start,stop):
    h=np.asarray(h)
    if (h.ndim!=1 or np.any(h<0) or np.any(h!=np.floor(h)) or
            int(cycles)!=cycles or cycles<0 or h.sum()>cycles or
            not 0<=start<=stop<len(h)):
        raise ValueError('Invalid first-event calibration/window')
    return int(h[start:stop+1].sum()),int(cycles-h[:start].sum())


@lru_cache(maxsize=64)
def leakage_ratio_upper(bins=256,shift=16,main=(27,37),delay=(31.5,32.5),width=(4.5,5.5)):
    lo,_=gaussian_envelope(bins,*delay,*width,normalized=False)
    sd0=max(0.,outward_float(arb(float(delay[0]))+shift,False))
    sd1=min(float(bins+shift-1),outward_float(arb(float(delay[1]))+shift))
    _,shift_hi=gaussian_envelope(bins+shift,sd0,sd1,*width,normalized=False)
    a=sum((arb(float(v)) for v in lo[main[0]:main[1]+1]),arb(0))
    e=sum((arb(float(v)) for v in shift_hi[main[0]:main[1]+1]),arb(0))
    if not a>0:raise ValueError('No positive calibrated main-window mass')
    return outward_float(e/a)


def lobe_confidence(h,cycles,delta=Fraction(1,1000),cap=.25,shift=16,
                    main=(27,37),dark=(128,255),delay=(31.5,32.5),width=(4.5,5.5)):
    """Upper confidence bound on relative optical-lobe height eta.

    Required calibration law: independent cycles with first recorded event,
    rate a*(g_d+eta*g_(d+shift))+b/T, nonnegative a,b, known d/w intervals,
    uniform background and 0<=eta<=cap. No Gaussian truncation approximation:
    shifted windows cancel their main/lobe Gaussian mass exactly. This does
    not cover detector afterpulsing, unknown cycle counts, or calibration drift.
    """
    h=np.asarray(h);bins=len(h);delta=Fraction(delta)
    if not (0<delta<1 and 0<=cap<=1 and isinstance(shift,int) and shift>0):
        raise ValueError('Invalid lobe calibration settings')
    lobe=(main[0]+shift,main[1]+shift)
    if not (main[1]<lobe[0] and lobe[1]<dark[0]):raise ValueError('Separated ordered windows required')
    counts={name:window_counts(h,cycles,*w) for name,w in [('main',main),('lobe',lobe),('dark',dark)]}
    budget=delta/3
    p0=binomial_endpoint(*counts['main'],budget,False)
    p1=binomial_endpoint(*counts['lobe'],budget,True)
    pd=binomial_endpoint(*counts['dark'],budget,True)
    result=dict(eta_upper=float(cap),delta=str(delta),cap=float(cap),cycles=int(cycles),
        detected=int(h.sum()),window_counts={k:list(v) for k,v in counts.items()},
        probability_endpoints=dict(main_lower=p0,lobe_upper=p1,dark_upper=pd),
        informative=False,scope='model-conditional one-sided calibration confidence')
    if p1>=1 or pd>=1:return result
    # Each float endpoint has already been checked in the conservative direction.
    main_l=-(-arb(p0)).log1p()
    lobe_u=-(-arb(p1)).log1p();dark_u=-(-arb(pd)).log1p()
    m=main[1]-main[0]+1;dark_m=dark[1]-dark[0]+1
    denominator=main_l-dark_u*arb(m)/dark_m
    if not denominator>0:return result
    leakage=leakage_ratio_upper(bins,shift,tuple(main),tuple(delay),tuple(width))
    upper=outward_float(lobe_u*(1+arb(float(cap))*arb(leakage))/denominator)
    result.update(eta_upper=min(float(cap),upper),informative=upper<float(cap),
                  leakage_ratio_upper=leakage,main_signal_hazard_lower=outward_float(denominator,False))
    return result
