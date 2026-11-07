"""Continuous-delay/width null relaxation with outward-rounded Arb bounds.

Every amplitude/background profile is upper-bounded by a concave tangent.
Adaptive subdivision tightens a complete cover; stopping early is conservative.
The statistical construction remains universal inference, not a new e-value
principle. Gaussian response envelopes are a specific, explicit model class.
"""
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import numpy as np
from flint import arb,ctx
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from scipy.ndimage import convolve
from pilot import coates
from crossfit import propose
from likelihood_evidence import gaussian_templates

ctx.prec=128


def outward_float(x,upper=True):
    """Enclose an Arb endpoint by an IEEE double, checking the direction."""
    endpoint=x.upper() if upper else x.lower()
    f=float(endpoint)
    if not math.isfinite(f):return f
    f=float(np.nextafter(f,np.inf if upper else -np.inf))
    assert (arb(f)>=endpoint) if upper else (arb(f)<=endpoint)
    return f


@lru_cache(maxsize=16384)
def gaussian_envelope(bins,d0,d1,f0,f1,normalized=True):
    if not (0<=d0<=d1<=bins-1 and 0<f0<=f1):
        raise ValueError('Invalid Gaussian response box')
    coefficient=4*arb(2).log()
    low=[];high=[]
    for t in range(bins):
        # Endpoints are represented exactly as supplied IEEE doubles.
        a,b=arb(d0)-t,arb(d1)-t
        max_distance=max(abs(a),abs(b))
        if d0<=t<=d1:min_distance=arb(0)
        else:min_distance=min(abs(a),abs(b))
        low.append((-coefficient*(max_distance/arb(f0))**2).exp())
        high.append((-coefficient*(min_distance/arb(f1))**2).exp())
    if normalized:
        lower_sum=sum(low,arb(0));upper_sum=sum(high,arb(0))
        if not lower_sum>0:raise ValueError('Degenerate envelope normalization')
        low=[a/upper_sum for a in low];high=[a/lower_sum for a in high]
    # With an unrestricted signal amplitude the unknown normalization can be
    # absorbed into that amplitude. Unnormalized envelopes cover the SAME
    # Gaussian null family without the extra normalization relaxation.
    lo=np.array([max(0.,outward_float(a,False)) for a in low])
    hi=np.array([min(1.,outward_float(a,True)) for a in high])
    lo.setflags(write=False);hi.setflags(write=False)
    return lo,hi


def exact_gaussian(bins,delay,fwhm,normalized=True):
    coefficient=4*arb(2).log()/arb(fwhm)**2
    values=[(-coefficient*(arb(t)-arb(delay))**2).exp() for t in range(bins)]
    if not normalized:return values
    total=sum(values,arb(0));return [v/total for v in values]


def _observations(h,cycles,family):
    h=np.asarray(h)
    if h.ndim!=1 or np.any(h<0) or np.any(h!=np.floor(h)):
        raise ValueError('Integer count vector required')
    h=h.astype(np.int64)
    if family=='first':
        if cycles is None or int(cycles)!=cycles or h.sum()>cycles:raise ValueError('Invalid cycle count')
        failures=int(cycles)-np.cumsum(h)
        saturated=arb(0)
        for count,fail in zip(h,failures):
            if count and fail:
                rate=(arb(int(count))/arb(int(fail))).log1p()
                saturated+=int(count)*(-(-rate).expm1()).log()-int(fail)*rate
        scale=int(cycles)
    elif family=='poisson':
        failures=np.ones_like(h)
        saturated=sum((int(v)*arb(int(v)).log()-int(v) for v in h if v),arb(0))
        scale=1
    else:raise ValueError(family)
    return h,failures,scale,outward_float(saturated)


def certified_log_likelihood(h,rates,cycles=None,family='first',upper=False):
    counts,failures,_,_=_observations(h,cycles,family)
    total=arb(0)
    for count,fail,rate in zip(counts,failures,rates):
        rate=rate if isinstance(rate,arb) else arb(float(rate))
        if not rate>=0:raise ValueError('Negative or indeterminate rate')
        if family=='first':
            if count:total+=int(count)*(-(-rate).expm1()).log()
            total-=int(fail)*rate
        else:
            if count:total+=int(count)*rate.log()
            total-=rate
    return outward_float(total,upper)


def relaxed_value_gradient(theta,h,failures,lo,hi,scale,family):
    """Floating-point search only; the returned bound is certified separately."""
    t=len(h)
    dl=np.stack([lo,np.ones(t)/t],axis=-1)/scale
    du=np.stack([hi,np.ones(t)/t],axis=-1)/scale
    lower=dl@theta;upper=du@theta
    if family=='first':
        optimum=np.full(t,np.inf)
        np.log1p(np.divide(h,failures,out=np.zeros(t,float),where=failures>0),
                 out=optimum,where=failures>0)
        optimum[(h==0)&(failures==0)]=0.
    else:optimum=np.asarray(h,float)
    rate=np.clip(optimum,lower,upper)
    if family=='first':
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        # Very large trial rates can overflow expm1; the reciprocal limit is
        # exactly zero. This only guides the optimizer, never the certificate.
        with np.errstate(over='ignore'):
            derivative=h/np.expm1(rate)-failures
    else:
        ell=np.sum(xlogy(h,rate)-rate)
        derivative=h/rate-1
    active=np.where((optimum<lower)[:,None],dl,
                    np.where((optimum>upper)[:,None],du,0.))
    gradient=np.sum(derivative[:,None]*active,axis=0)
    return float(ell),gradient


def certify_tangent(theta,h,failures,lo,hi,scale,family):
    t=len(h);aa,bb=arb(float(theta[0])),arb(float(theta[1]))
    phi=arb(0);ga=arb(0);gb=arb(0)
    base=arb(1)/(t*scale)
    for count,fail,lower_pulse,upper_pulse in zip(h,failures,lo,hi):
        count,fail=int(count),int(fail)
        dl=arb(float(lower_pulse))/scale;du=arb(float(upper_pulse))/scale
        lower=aa*dl+bb*base;upper=aa*du+bb*base
        if family=='first' and count==0 and fail==0:continue
        if family=='first' and fail==0:
            rate=upper;active=du
        else:
            optimum=((arb(count)/fail).log1p() if count else arb(0)) if family=='first' else arb(count)
            if optimum<lower:
                rate=lower;active=dl
            elif optimum>upper:
                rate=upper;active=du
            elif optimum>=lower and optimum<=upper:
                rate=optimum;active=None
            else:
                return None  # Interval ambiguity; never assume a derivative.
        if family=='first':
            if count:phi+=count*(-(-rate).expm1()).log()
            phi-=fail*rate
            derivative=arb(count)/rate.expm1()-fail if count else arb(-fail)
        else:
            if count:phi+=count*rate.log()
            phi-=rate
            derivative=arb(count)/rate-1 if count else arb(-1)
        if active is not None:
            ga+=derivative*active;gb+=derivative*base
    if ga<=0 and gb<=0:
        return outward_float(phi-ga*aa-gb*bb)
    return None


@dataclass
class Box:
    d0:float
    d1:float
    f0:float
    f1:float
    upper:float
    lower:float
    certified:bool
    optimizer_success:bool


def bound_box(h,failures,scale,family,d0,d1,f0,f1,saturated,normalized=True):
    lo,hi=gaussian_envelope(len(h),d0,d1,f0,f1,normalized)
    initial=np.array([max(h.sum()*.7,1.),max(h.sum()*.3,1.)])
    if not normalized:
        mid=np.exp(-4*np.log(2)*((np.arange(len(h))-(d0+d1)/2)/((f0+f1)/2))**2)
        initial[0]/=mid.sum()
    def loss(theta):
        ell,gradient=relaxed_value_gradient(theta,h,failures,lo,hi,scale,family)
        return -ell,-gradient
    fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(1e-10,None)],
                 options={'ftol':1e-12,'gtol':1e-8,'maxiter':120})
    theta=np.maximum(fit.x,[0.,1e-10])
    certificate=None
    for k in range(16):
        certificate=certify_tangent(theta*(1+1e-6*2**k),h,failures,lo,hi,scale,family)
        if certificate is not None:break
    upper=saturated if certificate is None else min(saturated,certificate)
    pulse=exact_gaussian(len(h),(d0+d1)/2,(f0+f1)/2,normalized)
    rates=[(arb(float(theta[0]))*g+arb(float(theta[1]))/len(h))/scale for g in pulse]
    lower=certified_log_likelihood(h,rates,scale if family=='first' else None,family)
    return Box(d0,d1,f0,f1,upper,lower,certificate is not None,bool(fit.success))


def continuous_null_bound(h,cycles=1000,fwhm_range=(5.,5.),family='first',
                          initial_cells=32,max_boxes=256,gap_tolerance=.01,
                          normalized_envelope=True):
    counts,failures,scale,saturated=_observations(h,cycles,family)
    if max_boxes<initial_cells or initial_cells<1:raise ValueError('Invalid subdivision budget')
    f0,f1=map(float,fwhm_range)
    if not 0<f0<=f1:raise ValueError('Invalid pulse-width interval')
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    queue=[];counter=0;best_lower=-np.inf;fallbacks=0;failures_count=0
    def add(d0,d1,w0,w1):
        nonlocal counter,best_lower,fallbacks,failures_count
        box=bound_box(counts,failures,scale,family,d0,d1,w0,w1,saturated,normalized_envelope)
        counter+=1;best_lower=max(best_lower,box.lower)
        fallbacks+=not box.certified;failures_count+=not box.optimizer_success
        heapq.heappush(queue,(-box.upper,counter,box))
    for d0,d1 in zip(edges[:-1],edges[1:]):add(float(d0),float(d1),f0,f1)
    while counter+2<=max_boxes and queue:
        upper=-queue[0][0]
        if upper-best_lower<=gap_tolerance:break
        _,_,box=heapq.heappop(queue)
        if box.upper<best_lower:continue
        delay_span=(box.d1-box.d0)/((box.f0+box.f1)/2)
        width_span=3*(box.f1-box.f0)/((box.f0+box.f1)/2)
        if delay_span>=width_span:
            mid=(box.d0+box.d1)/2
            if mid in (box.d0,box.d1):
                heapq.heappush(queue,(-box.upper,counter,box));break
            add(box.d0,mid,box.f0,box.f1);add(mid,box.d1,box.f0,box.f1)
        else:
            mid=(box.f0+box.f1)/2
            add(box.d0,box.d1,box.f0,mid);add(box.d0,box.d1,mid,box.f1)
    # An empty cover is mathematically inconsistent with valid upper/lower
    # bounds; retain the saturated bound if such a numerical failure occurs.
    upper=max(best_lower,-queue[0][0]) if queue else saturated
    return dict(upper=upper,feasible_lower=best_lower,gap=upper-best_lower,
                boxes_evaluated=counter,conservative_fallbacks=fallbacks,
                optimizer_failures=failures_count,arb_precision_bits=ctx.prec,
                fwhm_range=[f0,f1],family=family,
                parameterization='normalized' if normalized_envelope else 'peak amplitude')


def independent_alternative(h,cycles=1000,size=9):
    """Direct zero-center convolution prevents floating cancellation leakage."""
    if size<3 or size%2!=1 or size>min(h.shape[:2]):raise ValueError('Invalid support')
    x=coates(h,cycles)
    weights=np.ones((size,size,1))/(size*size-1)
    weights[size//2,size//2,0]=0.
    pooled=convolve(x,weights,mode='wrap')
    p,s,valid=propose(pooled);g=gaussian_templates(h.shape[-1]);alt=np.empty_like(x)
    for y in range(x.shape[0]):
        for col in range(x.shape[1]):
            design=np.stack([g[p[y,col]],g[s[y,col]],np.ones(x.shape[-1])/x.shape[-1]],axis=-1)
            coeff,_=nnls(design,pooled[y,col]);alt[y,col]=np.maximum(design@coeff,1e-12)
    return alt,p,s,valid


def robust_evidence(h,alternative,cycles=1000,fwhm_range=(5.,5.),**kwargs):
    bound=continuous_null_bound(h,cycles,fwhm_range,**kwargs)
    numerator=certified_log_likelihood(h,alternative,cycles)
    log_e_lower=outward_float(arb(numerator)-arb(bound['upper']),False)
    threshold=outward_float(arb(100).log())
    return dict(log_e_lower=log_e_lower,log_threshold_upper=threshold,
                declared=bool(log_e_lower>=threshold),numerator_lower=numerator,**bound)


def certified_mixture_likelihood(h,alternatives,cycles=1000,family='first'):
    if not alternatives:raise ValueError('Empty mixture')
    terms=[certified_log_likelihood(h,a,cycles,family) for a in alternatives]
    shift=max(terms)
    mixture=sum(((arb(x)-arb(shift)).exp() for x in terms),arb(0))/len(terms)
    return outward_float(arb(shift)+mixture.log(),False)


def robust_mixture_evidence(h,alternatives,cycles=1000,fwhm_range=(4.5,5.5),**kwargs):
    bound=continuous_null_bound(h,cycles,fwhm_range,**kwargs)
    numerator=certified_mixture_likelihood(h,alternatives,cycles)
    log_e=outward_float(arb(numerator)-arb(bound['upper']),False)
    threshold=outward_float(arb(100).log())
    return dict(log_e_lower=log_e,log_threshold_upper=threshold,
                declared=bool(log_e>=threshold),numerator_lower=numerator,
                mixture_components=len(alternatives),**bound)
