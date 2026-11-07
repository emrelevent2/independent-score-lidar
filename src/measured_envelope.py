"""Certified continuous-shift/stretch null for an independently supplied IRF.

The calibration samples define a piecewise-linear pulse, padded by one zero
sample at each end. Delay is its peak position, not its leading edge. A fixed
relative shape band is a MODEL TOLERANCE, not a measured calibration confidence
interval. This module does not modify the frozen Gaussian experiment.
"""
from functools import lru_cache
import heapq,math
import numpy as np
from flint import arb,ctx
from scipy.optimize import minimize,nnls,least_squares
from envelope_evidence import (outward_float,_observations,certified_log_likelihood,
    relaxed_value_gradient,certify_tangent,Box,certified_mixture_likelihood)


class MeasuredResponse:
    def __init__(self,irf,lower_irf=None,upper_irf=None):
        raw=np.asarray(irf,float)
        if raw.ndim!=1 or len(raw)<2 or not np.all(np.isfinite(raw)) or np.any(raw<0) or raw.max()<=0:
            raise ValueError('Finite nonnegative one-dimensional IRF required')
        self.samples=np.r_[0.,raw,0.];self.samples.setflags(write=False)
        low=raw if lower_irf is None else np.asarray(lower_irf,float)
        high=raw if upper_irf is None else np.asarray(upper_irf,float)
        if low.shape!=raw.shape or high.shape!=raw.shape or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low<0) or np.any(low>raw) or np.any(high<raw):
            raise ValueError('Calibration knot intervals must contain the nominal response')
        self.lower_samples=np.r_[0.,low,0.];self.upper_samples=np.r_[0.,high,0.]
        self.lower_samples.setflags(write=False);self.upper_samples.setflags(write=False)
        self.peak=int(np.argmax(self.samples))
        active=np.flatnonzero(raw>=raw.max()/2)
        self.fwhm=int(active[-1]-active[0]+1)

    def _at_float(self,u,samples=None):
        samples=self.samples if samples is None else samples
        if u<=0 or u>=len(samples)-1:return arb(0)
        j=math.floor(u)
        return arb(float(samples[j]))+(arb(float(u))-j)*(arb(float(samples[j+1]))-arb(float(samples[j])))

    def _range(self,u0,u1,samples=None):
        """Extrema of a continuous piecewise-linear curve over an interval."""
        samples=self.samples if samples is None else samples
        if u1<=0 or u0>=len(samples)-1:return arb(0),arb(0)
        vals=[self._at_float(u0,samples),self._at_float(u1,samples)]
        j0=max(0,math.ceil(u0));j1=min(len(samples)-1,math.floor(u1))
        if j0<=j1:
            values=samples[j0:j1+1]
            vals.extend([arb(float(values.min())),arb(float(values.max()))])
        return min(v.lower() for v in vals),max(v.upper() for v in vals)

    @lru_cache(maxsize=2048)
    def envelope(self,bins,d0,d1,s0,s1,shape_tolerance=0.):
        if not (0<=d0<=d1<=bins-1 and 0<s0<=s1 and 0<=shape_tolerance<1):
            raise ValueError('Invalid shift/stretch/shape box')
        lo=np.zeros(bins);hi=np.zeros(bins)
        minus=arb(1)-arb(float(shape_tolerance));plus=arb(1)+arb(float(shape_tolerance))
        left=arb(float(d0))-self.peak*arb(float(s1))
        right=arb(float(d1))+(len(self.samples)-1-self.peak)*arb(float(s1))
        start=max(0,math.floor(outward_float(left,False)))
        stop=min(bins,math.ceil(outward_float(right,True))+1)
        for t in range(start,stop):
            corners=[(arb(t)-arb(float(d)))/arb(float(s))+self.peak for d in [d0,d1] for s in [s0,s1]]
            u0=outward_float(min(v.lower() for v in corners),False)
            u1=outward_float(max(v.upper() for v in corners),True)
            low,_=self._range(u0,u1,self.lower_samples)
            _,high=self._range(u0,u1,self.upper_samples)
            lo[t]=max(0.,outward_float(low*minus,False))
            hi[t]=max(0.,outward_float(high*plus,True))
        lo.setflags(write=False);hi.setflags(write=False)
        return lo,hi

    def pulse(self,bins,delay,stretch=1.,normalized=False):
        if stretch<=0:raise ValueError('Positive stretch required')
        u=(np.arange(bins)-delay)/stretch+self.peak
        values=np.interp(u,np.arange(len(self.samples)),self.samples,left=0.,right=0.)
        if normalized:
            if values.sum()<=0:raise ValueError('Pulse outside time window')
            values=values/values.sum()
        return values

    def exact_pulse(self,bins,delay,stretch=1.):
        out=[arb(0) for _ in range(bins)]
        left=arb(float(delay))-self.peak*arb(float(stretch))
        right=arb(float(delay))+(len(self.samples)-1-self.peak)*arb(float(stretch))
        start=max(0,math.floor(outward_float(left,False)))
        stop=min(bins,math.ceil(outward_float(right,True))+1)
        for t in range(start,stop):
            u=(arb(t)-arb(float(delay)))/arb(float(stretch))+self.peak
            lo,hi=self._range(outward_float(u,False),outward_float(u,True))
            out[t]=lo.union(hi)
        return out

    @lru_cache(maxsize=4)
    def templates(self,bins):
        out=np.stack([self.pulse(bins,d,normalized=True) for d in range(bins)])
        out.setflags(write=False);return out


def measured_box(h,response,d0,d1,s0,s1,saturated,shape_tolerance=0.,family='poisson',cycles=None):
    counts,failures,scale,_=_observations(h,cycles,family)
    lo,hi=response.envelope(len(h),d0,d1,s0,s1,shape_tolerance)
    pulse=response.pulse(len(h),(d0+d1)/2,(s0+s1)/2)
    initial=np.array([max(h.sum()*.7,1.)/max(pulse.sum(),1e-12),max(h.sum()*.3,1.)])
    def loss(theta):
        value,gradient=relaxed_value_gradient(theta,counts,failures,lo,hi,scale,family)
        return -value,-gradient
    fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(1e-10,None)],
        options={'ftol':1e-12,'gtol':1e-8,'maxiter':120})
    theta=np.maximum(fit.x,[0.,1e-10]);certificate=None
    for k in range(16):
        certificate=certify_tangent(theta*(1+1e-6*2**k),counts,failures,lo,hi,scale,family)
        if certificate is not None:break
    upper=saturated if certificate is None else min(saturated,certificate)
    exact=response.exact_pulse(len(h),(d0+d1)/2,(s0+s1)/2)
    rates=[(arb(float(theta[0]))*g+arb(float(theta[1]))/len(h))/scale for g in exact]
    lower=certified_log_likelihood(h,rates,cycles,family)
    return Box(d0,d1,s0,s1,upper,lower,certificate is not None,bool(fit.success))


def measured_null_bound(h,response,stretch_range=(.9,1.1),shape_tolerance=.05,
                        initial_cells=32,max_boxes=256,gap_tolerance=.01,
                        family='poisson',cycles=None):
    counts,_,_,saturated=_observations(h,cycles,family)
    s0,s1=map(float,stretch_range)
    if not 0<s0<=s1 or max_boxes<initial_cells or initial_cells<1:raise ValueError('Invalid budget/stretch interval')
    queue=[];counter=0;best_lower=-np.inf;fallbacks=0;failed=0
    def add(d0,d1,w0,w1):
        nonlocal counter,best_lower,fallbacks,failed
        box=measured_box(counts,response,d0,d1,w0,w1,saturated,shape_tolerance,family,cycles)
        counter+=1;best_lower=max(best_lower,box.lower);fallbacks+=not box.certified;failed+=not box.optimizer_success
        heapq.heappush(queue,(-box.upper,counter,box))
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    for a,b in zip(edges[:-1],edges[1:]):add(float(a),float(b),s0,s1)
    while counter+2<=max_boxes and queue:
        if -queue[0][0]-best_lower<=gap_tolerance:break
        _,_,box=heapq.heappop(queue)
        if box.upper<best_lower:continue
        if (box.d1-box.d0)>=response.fwhm*(box.f1-box.f0):
            mid=(box.d0+box.d1)/2
            if mid in (box.d0,box.d1):heapq.heappush(queue,(-box.upper,counter,box));break
            add(box.d0,mid,box.f0,box.f1);add(mid,box.d1,box.f0,box.f1)
        else:
            mid=(box.f0+box.f1)/2
            if mid in (box.f0,box.f1):heapq.heappush(queue,(-box.upper,counter,box));break
            add(box.d0,box.d1,box.f0,mid);add(box.d0,box.d1,mid,box.f1)
    upper=max(best_lower,-queue[0][0]) if queue else saturated
    return dict(upper=upper,feasible_lower=best_lower,gap=upper-best_lower,boxes_evaluated=counter,
        conservative_fallbacks=fallbacks,optimizer_failures=failed,arb_precision_bits=ctx.prec,
        family=family,stretch_range=[s0,s1],shape_tolerance=shape_tolerance,
        response_model='piecewise-linear supplied samples, one zero padding knot each end; free peak amplitude')


def measured_alternative_at(h,response,row,col,size=9,stretch_range=(.9,1.1),factors=(.5,1.,2.,4.)):
    """All parameters from distinct donors; no floating subtraction of center."""
    h=np.asarray(h);radius=size//2
    if size%2!=1 or size<3 or not(radius<=row<h.shape[0]-radius and radius<=col<h.shape[1]-radius):
        raise ValueError('Complete distinct donor window required')
    patch=h[row-radius:row+radius+1,col-radius:col+radius+1].reshape(size*size,h.shape[-1])
    donor_mask=np.ones(size*size,bool);donor_mask[size*size//2]=False
    pooled=np.mean(patch[donor_mask],axis=0,dtype=np.float64)
    bins=len(pooled);g=response.templates(bins);base=np.ones(bins)/bins
    energy=np.sum(g*g,axis=1);centered=pooled-np.median(pooled)
    primary=int(np.argmax((g@centered)/energy))
    coef,_=nnls(np.stack([g[primary],base],axis=-1),pooled)
    residual=pooled-coef[0]*g[primary]-coef[1]*base
    score=(g@residual)/energy;t=np.arange(bins)
    separation=max(8,int(np.ceil(1.6*response.fwhm)))
    allowed=(np.abs(t-primary)>=separation)&(t>=response.fwhm)&(t<bins-response.fwhm)
    valid=bool(allowed.any());secondary=int(np.argmax(np.where(allowed,score,-np.inf)))
    coef,_=nnls(np.stack([g[primary],g[secondary],base],axis=-1),pooled)
    def model(q):
        a,c,b,d1,d2,stretch=q
        return a*response.pulse(bins,d1,stretch,True)+c*response.pulse(bins,d2,stretch,True)+b/bins
    lower=np.array([0.,0.,1e-10,max(0.,primary-1.5),max(0.,secondary-1.5),stretch_range[0]])
    upper=np.array([np.inf,np.inf,np.inf,min(bins-1.,primary+1.5),min(bins-1.,secondary+1.5),stretch_range[1]])
    start=np.r_[coef,primary,secondary,np.mean(stretch_range)]
    start=np.minimum(np.maximum(start,lower+1e-9),upper-1e-9)
    fit=least_squares(lambda q:model(q)-pooled,start,bounds=(lower,upper),
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=120)
    a,c,b,d1,d2,stretch=fit.x
    means=[np.maximum(a*response.pulse(bins,d1,stretch,True)+factor*c*response.pulse(bins,d2,stretch,True)+b/bins,1e-12) for factor in factors]
    return means,dict(primary=float(d1),secondary=float(d2),primary_count=float(a),secondary_count=float(c),
        background_count=float(b),stretch=float(stretch),minimum_proposal_separation=separation,
        valid=valid,optimizer_success=bool(fit.success),donors=int(donor_mask.sum()),amplitude_factors=list(factors))


def measured_evidence(h,alternatives,response,alpha=.01,**kwargs):
    h=np.asarray(h)
    if not alternatives or any(np.shape(a)!=h.shape or np.any(np.asarray(a)<=0) or not np.all(np.isfinite(a)) for a in alternatives):
        raise ValueError('Proper positive finite Poisson alternatives required')
    bound=measured_null_bound(h,response,**kwargs)
    if bound['family']!='poisson':raise ValueError('This public entry point uses the Poisson count model')
    numerator=certified_mixture_likelihood(h,alternatives,cycles=None,family='poisson')
    log_e=outward_float(arb(numerator)-arb(bound['upper']),False)
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    threshold=outward_float((arb(1)/arb(str(alpha))).log(),True)
    return dict(log_e_lower=log_e,log_threshold_upper=threshold,declared=bool(log_e>=threshold),
        numerator_lower=numerator,mixture_components=len(alternatives),alpha=alpha,**bound)
