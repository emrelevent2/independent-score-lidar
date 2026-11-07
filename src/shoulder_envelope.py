"""Continuous-delay/width null including an uncertain optical IRF lobe.

For eta in [0,M], a*(g+eta*shift(g)) equals u*g+v*(g+M*shift(g)),
u,v>=0. Thus a response-shape interval adds a nonnegative nuisance coordinate,
not another cover dimension. Complete d/w covers and Arb-checked concave
tangents give conservative bounds even when stopped early. Gaussian/lobe
shape, fixed shift and independent-cycle recording are explicit assumptions.
"""
import heapq
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from flint import arb,ctx
from envelope_evidence import (Box,_observations,gaussian_envelope,exact_gaussian,
                               outward_float,certified_log_likelihood)


def response_basis(bins,d0,d1,w0,w1,cap,shift):
    if not (np.isfinite(cap) and cap>=0 and isinstance(shift,int) and shift>0):
        raise ValueError('Invalid optical-lobe model')
    lo,hi=gaussian_envelope(bins,d0,d1,w0,w1,normalized=False)
    # Adding an integer to an IEEE delay can round. Enclose the exact real
    # shifted endpoints, rather than silently changing the physical model.
    sd0=max(0.,outward_float(arb(float(d0))+shift,False))
    sd1=min(float(bins+shift-1),outward_float(arb(float(d1))+shift,True))
    sl,sh=gaussian_envelope(bins+shift,sd0,sd1,w0,w1,normalized=False)
    lower=np.array([max(0.,outward_float(arb(float(a))+arb(float(cap))*arb(float(b)),False))
                    for a,b in zip(lo,sl[:bins])])
    upper=np.array([outward_float(arb(float(a))+arb(float(cap))*arb(float(b)))
                    for a,b in zip(hi,sh[:bins])])
    return np.stack([lo,lower],axis=1),np.stack([hi,upper],axis=1)


def relaxed_value_gradient(theta,h,failures,lo,hi,scale,family):
    t=len(h);dl=np.column_stack([lo,np.full(t,1/t)])/scale
    du=np.column_stack([hi,np.full(t,1/t)])/scale
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
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            derivative=np.divide(h,np.expm1(rate),out=np.zeros(t),where=rate>0)-failures
    else:
        ell=np.sum(xlogy(h,rate)-rate)
        derivative=np.divide(h,rate,out=np.zeros(t),where=rate>0)-1
    active=np.where((optimum<lower)[:,None],dl,np.where((optimum>upper)[:,None],du,0.))
    return float(ell),np.sum(derivative[:,None]*active,axis=0)


def certify_tangent(theta,h,failures,lo,hi,scale,family):
    if np.any(np.asarray(theta)<0) or not np.all(np.isfinite(theta)):return None
    theta=[arb(float(v)) for v in theta];phi=arb(0);gradient=[arb(0) for _ in theta]
    base=arb(1)/(len(h)*scale)
    for count,fail,lp,up in zip(h,failures,lo,hi):
        count,fail=int(count),int(fail)
        dl=[arb(float(x))/scale for x in lp]+[base]
        du=[arb(float(x))/scale for x in up]+[base]
        lower=sum((x*y for x,y in zip(theta,dl)),arb(0))
        upper=sum((x*y for x,y in zip(theta,du)),arb(0))
        if family=='first' and count==0 and fail==0:continue
        if family=='first' and fail==0:rate,active=upper,du
        else:
            optimum=((arb(count)/fail).log1p() if count else arb(0)) if family=='first' else arb(count)
            if optimum<lower:rate,active=lower,dl
            elif optimum>upper:rate,active=upper,du
            elif optimum>=lower and optimum<=upper:rate,active=optimum,None
            else:return None
        if family=='first':
            if count:phi+=count*(-(-rate).expm1()).log()
            phi-=fail*rate
            derivative=arb(count)/rate.expm1()-fail if count else arb(-fail)
        else:
            if count:phi+=count*rate.log()
            phi-=rate;derivative=arb(count)/rate-1 if count else arb(-1)
        if active is not None:
            for j in range(3):gradient[j]+=derivative*active[j]
    if all(v<=0 for v in gradient):
        return outward_float(phi-sum((v*x for v,x in zip(gradient,theta)),arb(0)))
    return None


def exact_rates(bins,d,w,theta,cap,shift,scale):
    main=exact_gaussian(bins,d,w,False)
    side=exact_gaussian(bins+shift,arb(float(d))+shift,w,False)[:bins]
    u,v,b=map(lambda x:arb(float(x)),theta)
    return [(u*g+v*(g+arb(float(cap))*s)+b/bins)/scale for g,s in zip(main,side)]


def bound_box(h,failures,scale,family,d0,d1,w0,w1,saturated,cap,shift):
    lo,hi=response_basis(len(h),d0,d1,w0,w1,cap,shift)
    initial=np.array([max(h.sum()*.35,1.)/(5.*1.0645)]*2+[max(h.sum()*.3,1.)])
    def loss(theta):
        ell,gradient=relaxed_value_gradient(theta,h,failures,lo,hi,scale,family)
        return -ell,-gradient
    fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(0,None),(1e-10,None)],
                 options={'ftol':1e-12,'gtol':1e-8,'maxiter':120})
    theta=np.maximum(fit.x,[0.,0.,1e-10]);certificate=None
    for k in range(16):
        certificate=certify_tangent(theta*(1+1e-6*2**k),h,failures,lo,hi,scale,family)
        if certificate is not None:break
    upper=saturated if certificate is None else min(saturated,certificate)
    rates=exact_rates(len(h),(d0+d1)/2,(w0+w1)/2,theta,cap,shift,scale)
    lower=certified_log_likelihood(h,rates,scale if family=='first' else None,family)
    return Box(d0,d1,w0,w1,upper,lower,certificate is not None,bool(fit.success))


def shoulder_null_bound(h,cycles=1000,fwhm_range=(4.5,5.5),family='first',
                       cap=.25,shift=16,initial_cells=32,max_boxes=256,gap_tolerance=.01,
                       decision_level=None):
    counts,failures,scale,saturated=_observations(h,cycles,family)
    if not (np.isfinite(cap) and cap>=0 and isinstance(shift,int) and shift>0):raise ValueError('Invalid optical model')
    if not (1<=initial_cells<=max_boxes):raise ValueError('Invalid subdivision budget')
    w0,w1=map(float,fwhm_range)
    if not 0<w0<=w1:raise ValueError('Invalid width interval')
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    queue=[];counter=0;best=-np.inf;fallbacks=0;optimizer_failures=0
    def add(d0,d1,f0,f1,parent_upper=None):
        nonlocal counter,best,fallbacks,optimizer_failures
        box=bound_box(counts,failures,scale,family,d0,d1,f0,f1,saturated,cap,shift)
        if parent_upper is not None:box.upper=min(box.upper,parent_upper)
        counter+=1;best=max(best,box.lower);fallbacks+=not box.certified
        optimizer_failures+=not box.optimizer_success
        if box.lower>box.upper:raise ArithmeticError('Contradictory feasible/certified likelihood bounds')
        heapq.heappush(queue,(-box.upper,counter,box))
    for d0,d1 in zip(edges[:-1],edges[1:]):add(float(d0),float(d1),w0,w1)
    while counter+2<=max_boxes and queue:
        if -queue[0][0]-best<=gap_tolerance:break
        # Query-dependent stopping keeps the entire current cover and its
        # certified upper bound; it changes computation, never validity.
        if decision_level is not None and -queue[0][0]<=decision_level:break
        _,_,box=heapq.heappop(queue)
        if box.upper<best:continue
        ds=(box.d1-box.d0)/((box.f0+box.f1)/2)
        ws=3*(box.f1-box.f0)/((box.f0+box.f1)/2)
        if ds>=ws:
            mid=(box.d0+box.d1)/2
            if mid in (box.d0,box.d1):heapq.heappush(queue,(-box.upper,counter,box));break
            add(box.d0,mid,box.f0,box.f1,box.upper);add(mid,box.d1,box.f0,box.f1,box.upper)
        else:
            mid=(box.f0+box.f1)/2
            if mid in (box.f0,box.f1):heapq.heappush(queue,(-box.upper,counter,box));break
            add(box.d0,box.d1,box.f0,mid,box.upper);add(box.d0,box.d1,mid,box.f1,box.upper)
    upper=max(best,-queue[0][0]) if queue else saturated
    return dict(upper=upper,feasible_lower=best,gap=upper-best,boxes_evaluated=counter,
                conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,
                arb_precision_bits=ctx.prec,fwhm_range=[w0,w1],family=family,
                optical_lobe_cap=float(cap),optical_lobe_shift=shift,
                parameterization='three nonnegative peak-amplitude nuisance coordinates')
