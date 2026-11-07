"""Calibrated point-IRF control, ignoring finite calibration uncertainty.

Bounds certify its specified fixed-eta likelihood, not coverage when the
estimated calibration eta differs from the physical eta. This is a stronger
response-aware comparison than an intentionally misspecified Gaussian null.
"""
import heapq
import numpy as np
from scipy.optimize import minimize,least_squares
from flint import arb
from pilot import coates
from guided_alternatives import guided_pool
from shoulder_alternative import proposal_from_pool,components,pulse
from shoulder_envelope import response_basis,relaxed_value_gradient,certify_tangent
from envelope_evidence import Box,_observations,exact_gaussian,certified_log_likelihood


def estimated_eta(calibration,cycles=100000):
    return proposal_from_pool(coates(np.asarray(calibration),cycles),cycles=cycles)['eta_donor']


def fixed_proposal(h,eta,row=4,col=4,cycles=1000):
    pool,meta=guided_pool(h,row,col,cycles=cycles)
    q=proposal_from_pool(pool,cycles,cap=eta);t=np.arange(len(pool));obs=pool*cycles
    def model(z):
        a,c,b,d,s,w=z;return a*pulse(t,d,w,eta)+c*pulse(t,s,w,eta)+b/len(t)
    low=np.array([0.,0.,1e-10,max(0.,q['primary']-1.5),max(0.,q['secondary']-1.5),4.5])
    high=np.array([np.inf]*3+[min(len(t)-1.,q['primary']+1.5),min(len(t)-1.,q['secondary']+1.5),5.5])
    x=np.array([q[k] for k in ['primary_incident','secondary_incident','background_incident','primary','secondary','fitted_fwhm']])
    x=np.minimum(np.maximum(x,low+1e-9),high-1e-9)
    fit=least_squares(lambda z:model(z)-obs,x,bounds=(low,high),max_nfev=120,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    q.update({k:float(v) for k,v in zip(['primary_incident','secondary_incident','background_incident','primary','secondary','fitted_fwhm'],fit.x)})
    q.update(meta);q.update(eta_donor=float(eta),optimizer_success=bool(fit.success))
    return q


def fixed_components(q,eta):
    m=components(q,cap=eta);n=len(m['rates'])//3
    # Collapse identical eta components exactly; do not waste sparse work on
    # duplicated shapes or alter their total prior mass.
    m.update(rates=m['rates'][-n:].copy(),weights=[3*w for w in m['weights'][-n:]],
             locations=m['locations'][-n:].copy(),spatial=m['spatial'][-n:].copy())
    assert sum(m['weights'])==1
    return m


def fixed_box(h,failures,scale,d0,d1,w0,w1,saturated,eta):
    lo,hi=response_basis(len(h),d0,d1,w0,w1,eta,16)
    lo=np.repeat(lo[:,1,None],2,axis=1);hi=np.repeat(hi[:,1,None],2,axis=1)
    initial=np.array([max(h.sum()*.35,1.)/(5*1.0645)]*2+[max(h.sum()*.3,1.)])
    def loss(theta):
        ell,grad=relaxed_value_gradient(theta,h,failures,lo,hi,scale,'first');return -ell,-grad
    fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(0,None),(1e-10,None)],
                 options={'ftol':1e-12,'gtol':1e-8,'maxiter':120})
    theta=np.maximum(fit.x,[0.,0.,1e-10]);certificate=None
    for k in range(16):
        certificate=certify_tangent(theta*(1+1e-6*2**k),h,failures,lo,hi,scale,'first')
        if certificate is not None:break
    upper=saturated if certificate is None else min(saturated,certificate)
    d=(d0+d1)/2;w=(w0+w1)/2;g=exact_gaussian(len(h),d,w,False)
    side=exact_gaussian(len(h)+16,arb(d)+16,w,False)[:len(h)]
    amp=arb(float(theta[0]))+arb(float(theta[1]));bg=arb(float(theta[2]))
    rates=[(amp*(x+arb(float(eta))*y)+bg/len(h))/scale for x,y in zip(g,side)]
    lower=certified_log_likelihood(h,rates,scale,'first')
    return Box(d0,d1,w0,w1,upper,lower,certificate is not None,bool(fit.success))


def fixed_null_bound(h,eta,cycles=1000,initial_cells=32,max_boxes=256,decision_level=None):
    if not (np.isfinite(eta) and 0<=eta<=.25):raise ValueError('Invalid fixed IRF')
    h,failures,scale,saturated=_observations(h,cycles,'first');counter=0;best=-np.inf
    queue=[];fallbacks=0
    def add(d0,d1,w0,w1,parent=None):
        nonlocal counter,best,fallbacks
        b=fixed_box(h,failures,scale,d0,d1,w0,w1,saturated,eta)
        if parent is not None:b.upper=min(b.upper,parent)
        assert b.lower<=b.upper
        counter+=1;best=max(best,b.lower);fallbacks+=not b.certified
        heapq.heappush(queue,(-b.upper,counter,b))
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    for x,y in zip(edges[:-1],edges[1:]):add(float(x),float(y),4.5,5.5)
    while counter+2<=max_boxes and queue:
        upper=-queue[0][0]
        if upper-best<=.01 or (decision_level is not None and upper<=decision_level):break
        _,_,b=heapq.heappop(queue)
        if b.upper<best:continue
        if b.d1-b.d0>=3*(b.f1-b.f0):
            mid=(b.d0+b.d1)/2
            if mid in (b.d0,b.d1):heapq.heappush(queue,(-b.upper,counter,b));break
            add(b.d0,mid,b.f0,b.f1,b.upper);add(mid,b.d1,b.f0,b.f1,b.upper)
        else:
            mid=(b.f0+b.f1)/2
            if mid in (b.f0,b.f1):heapq.heappush(queue,(-b.upper,counter,b));break
            add(b.d0,b.d1,b.f0,mid,b.upper);add(b.d0,b.d1,mid,b.f1,b.upper)
    upper=max(best,-queue[0][0]) if queue else saturated
    return dict(upper=upper,feasible_lower=best,gap=upper-best,boxes_evaluated=counter,
                conservative_fallbacks=fallbacks,fixed_eta=float(eta),
                scope='Fixed estimated IRF only; physical finite-calibration coverage is not asserted')
