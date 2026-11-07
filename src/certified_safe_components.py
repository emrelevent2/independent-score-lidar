"""Exact recorded-time risk arithmetic and periodic Gaussian rate envelopes.

Recorded IEEE timestamps/lag are treated as exact dyadics of the declared
idealized continuous-time,piecewise-constant phase-rate model. This does NOT
certify physical timestamp quantization,jitter or uncertain detector recovery.
"""
from __future__ import annotations
from functools import lru_cache
from fractions import Fraction
import math
import numpy as np
from flint import arb,ctx


def outward(value,upper):
    x=value.upper() if upper else value.lower();v=float(x)
    if math.isfinite(v):v=math.nextafter(v,math.inf if upper else -math.inf)
    if math.isfinite(v):assert arb(v)>=x if upper else arb(v)<=x
    return v


def exact_safe_statistics(observed,cycles,bins,lag,previous=None):
    """Integer common-denominator reconstruction,including unscored hits.

    O(number of observed events + bins),with arbitrary-precision integer
    ticks. All mask decisions/additions/modulo operations are exact. Return
    exact phase exposure ticks plus outward float endpoints for optimizers.
    """
    t=np.asarray(observed,dtype=float).reshape(-1)
    if int(cycles)!=cycles or cycles<=0 or int(bins)!=bins or bins<=0:raise ValueError('Invalid horizon')
    if not np.isfinite(lag) or lag<0 or np.any(~np.isfinite(t)) or np.any(np.diff(t)<=0):raise ValueError('Invalid times')
    if previous is not None and (not np.isfinite(previous) or previous>=0):raise ValueError('Previous must precede block')
    ratios=[float(x).as_integer_ratio() for x in t];lr=float(lag).as_integer_ratio();pr=None if previous is None else float(previous).as_integer_ratio()
    denominator=max([1,lr[1]]+[d for n,d in ratios]+([] if pr is None else [pr[1]]));scale=int(denominator);k=int(bins);horizon=int(cycles)*k*scale
    ticks=[n*(scale//d) for n,d in ratios];delay=lr[0]*(scale//lr[1]);prev=None if pr is None else pr[0]*(scale//pr[1])
    if any(v<0 or v>=horizon for v in ticks):raise ValueError('Hit outside block')
    initial=0 if prev is None else max(0,prev+delay);opens=[];ends=[];s=[0]*k;open_at=initial;eligible=[]
    for v in ticks:
        score=v>=open_at;eligible.append(score)
        if score:s[(v//scale)%k]+=1
        if v>open_at:opens.append(open_at);ends.append(v)
        open_at=v+delay
    if open_at<horizon:opens.append(open_at);ends.append(horizon)
    def endpoint_ramps(values):
        count=[0]*k;fraction=[0]*k;cycles_sum=0
        for value in values:
            q,r=divmod(value,k*scale);j,f=divmod(r,scale);cycles_sum+=q;count[j]+=1;fraction[j]+=f
        prefix=0;ramps=[]
        for j in range(k):
            prefix+=count[j];ramps.append((len(values)-prefix)*scale+fraction[j])
        return cycles_sum,ramps
    qc,rc=endpoint_ramps(ends);qo,ro=endpoint_ramps(opens)
    e=[(qc-qo)*scale+a-b for a,b in zip(rc,ro)]
    assert all(0<=v<=int(cycles)*scale for v in e) and sum(e)==sum(b-a for a,b in zip(opens,ends))
    values=[arb(v)/arb(scale) for v in e]
    lo=np.array([max(0.,outward(v,False)) for v in values]);hi=np.array([outward(v,True) for v in values])
    return dict(counts=np.array(s,dtype=np.int64),exposure_ticks=e,denominator=scale,exposure=values,exposure_lower=lo,exposure_upper=hi,eligible=np.array(eligible,dtype=bool),scored_events=sum(s),observed_events=len(t))


def normal_bin_mass(distance,sigma):
    dist=distance if isinstance(distance,arb) else arb(distance);sig=sigma if isinstance(sigma,arb) else arb(sigma);root=arb(2).sqrt()
    return (((dist-arb(1)/2)/(sig*root)).erfc()-((dist+arb(1)/2)/(sig*root)).erfc())/2


@lru_cache(maxsize=32768)
def periodic_envelope(bins,d0,d1,s0,s1,images=2):
    if ctx.prec<128:raise RuntimeError('Arb128required')
    if not (0<=d0<=d1<=bins and 0<s0<=s1 and images>=1):raise ValueError('Invalid Gaussian box')
    sigma=arb(float(s0)).union(arb(float(s1)));tail=(arb(images*bins)/(arb(float(s1))*arb(2).sqrt())).erfc();lower=[];upper=[]
    for r in range(bins):
        lo=arb(0);hi=arb(0)
        for j in range(-images,images+1):
            center=Fraction(2*r+1,2)+j*bins
            x=abs(arb(center.numerator)/center.denominator-arb(float(d0)));y=abs(arb(center.numerator)/center.denominator-arb(float(d1)))
            far=x.max(y);near=arb(0) if d0<=center<=d1 else x.min(y)
            # Normal interval mass decreases with distance from its center at
            # every positive sigma. Interval sigma encloses width uncertainty.
            lo+=normal_bin_mass(far,sigma);hi+=normal_bin_mass(near,sigma)
        lower.append(max(0.,outward(lo,False)));upper.append(min(1.,outward(hi+tail,True)))
    a=np.array(lower);b=np.array(upper);assert np.all(a<=b);a.setflags(write=False);b.setflags(write=False)
    return a,b


def periodic_point_rates(bins,a,delay,background,sigma,images=2):
    """Enclose infinite wrapped normal masses;omitted positive tails retained."""
    d=arb(float(delay));sig=arb(float(sigma));tail=(arb(images*bins)/(sig*arb(2).sqrt())).erfc();out=[]
    if not (a>=0 and background>=0 and 0<=delay<=bins and sigma>0):raise ValueError('Invalid point')
    for r in range(bins):
        m=sum((normal_bin_mass(abs(arb(r)+arb(1)/2+j*bins-d),sig) for j in range(-images,images+1)),arb(0))
        # The actual omitted contribution is in[0,tail],not an arbitrary
        # renormalization. No false claim that a finite image sum is exact.
        m=m+arb(0).union(tail)
        out.append(arb(float(a))*m+arb(float(background))/bins)
    return out


def log_likelihood_interval(stats,rates):
    if len(rates)!=len(stats['counts']):raise ValueError('Rate dimension mismatch')
    result=arb(0)
    for count,exposure,rate in zip(stats['counts'],stats['exposure'],rates):
        q=rate if isinstance(rate,arb) else arb(float(rate))
        if not q>0:raise ValueError('Strictly positive rates needed')
        if count:result+=int(count)*q.log()
        result-=exposure*q
    return result


def mixture_log_lower(stats,rate_vectors,weights):
    weights=[Fraction(v) for v in weights]
    if len(weights)!=len(rate_vectors) or sum(weights)!=1 or any(v<=0 for v in weights):raise ValueError('Proper fixed weights required')
    ll=[log_likelihood_interval(stats,v) for v in rate_vectors];shift=max(float(v.mid()) for v in ll);total=arb(0)
    for value,w in zip(ll,weights):total+=arb(w.numerator)/w.denominator*(value-arb(shift)).exp()
    return outward(arb(shift)+total.log(),False)
