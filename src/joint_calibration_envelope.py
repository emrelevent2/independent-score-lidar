"""Complete shared-eta/delay/width cover for calibration + query likelihood.

Calibration is observed in six contiguous first-event bins. Its known-target
phase/width range is enclosed by normalized window-mass bounds. Neither a
plug-in shape nor a confidence-set truncation authorizes a declaration.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from flint import arb
from envelope_evidence import _observations,outward_float
from linear_envelope_profile import profile_upper
from joint_calibration_model import calibration_basis,query_basis


@dataclass
class Cell:
    eta0:float
    eta1:float
    d0:float
    d1:float
    w0:float
    w1:float
    upper:float


def joint_null_bound(calibration,query,calibration_cycles,query_cycles=1000,
                     cap=.25,initial_eta=8,initial_delay=16,max_boxes=1024,
                     decision_level=None):
    cal=np.asarray(calibration)
    if cal.shape!=(6,):raise ValueError('Six contiguous calibration bins required')
    _observations(cal,calibration_cycles,'first')
    def provider(e0,e1):
        lo,hi=calibration_basis(e0,e1,cap)
        return profile_upper(cal,lo,hi,calibration_cycles,[max(float(cal[1]),1.),max(float(cal[-1])*2,1.)])
    return shared_response_cover(query,provider,query_cycles,cap,initial_eta,initial_delay,max_boxes,decision_level,6)


def shared_response_cover(query,calibration_provider,query_cycles=1000,cap=.25,
                          initial_eta=8,initial_delay=16,max_boxes=1024,
                          decision_level=None,calibration_observed_bins=1):
    query=np.asarray(query)
    if not(0<cap<=.25 and initial_eta>=1 and initial_delay>=1 and max_boxes>=initial_eta*initial_delay):
        raise ValueError('Invalid complete shared-response cover')
    _,_,_,query_saturated=_observations(query,query_cycles,'first')
    cache={};queue=[];counter=0;query_profiles=0;fallbacks=0;optimizer_failures=0;pruned=0
    def cal_upper(e0,e1):
        nonlocal fallbacks,optimizer_failures
        key=(e0,e1)
        if key not in cache:
            fit=calibration_provider(e0,e1)
            cache[key]=fit
            fallbacks+=not fit.get('profile_certified',fit.get('tangent_certified',False))
            optimizer_failures+=fit.get('optimizer_success') is False
        return cache[key]['upper']
    def add(e0,e1,d0,d1,w0,w1,parent_cap=None):
        nonlocal counter,query_profiles,fallbacks,optimizer_failures,pruned
        cu=cal_upper(e0,e1)
        upper=outward_float(arb(cu)+arb(query_saturated))
        # Saturation alone already covers this whole cell. The complete cover
        # is retained even when expensive query profiling is unnecessary.
        if decision_level is not None and upper<=decision_level:pruned+=1
        else:
            lo,hi=query_basis(len(query),d0,d1,w0,w1,e0,e1)
            fit=profile_upper(query,lo,hi,query_cycles,
                [max(float(query.sum())*.7,1.)/(5*1.0645),max(float(query.sum())*.3,1.)])
            upper=outward_float(arb(cu)+arb(fit['upper']));query_profiles+=1
            fallbacks+=not fit['tangent_certified'];optimizer_failures+=not fit['optimizer_success']
        if parent_cap is not None:upper=min(upper,parent_cap)
        counter+=1;cell=Cell(e0,e1,d0,d1,w0,w1,upper)
        heapq.heappush(queue,(-upper,counter,cell))
    etas=np.linspace(0.,cap,initial_eta+1);delays=np.linspace(0.,len(query)-1.,initial_delay+1)
    for e0,e1 in zip(etas[:-1],etas[1:]):
        for d0,d1 in zip(delays[:-1],delays[1:]):add(float(e0),float(e1),float(d0),float(d1),4.5,5.5)
    while queue and counter+2<=max_boxes:
        if decision_level is not None and -queue[0][0]<=decision_level:break
        _,_,c=heapq.heappop(queue)
        mean_width=(c.w0+c.w1)/2
        spans=[(c.eta1-c.eta0)/(cap/16),(c.d1-c.d0)/mean_width,3*(c.w1-c.w0)/mean_width]
        axis=int(np.argmax(spans))
        if axis==0:
            mid=(c.eta0+c.eta1)/2
            if mid in (c.eta0,c.eta1):heapq.heappush(queue,(-c.upper,counter,c));break
            add(c.eta0,mid,c.d0,c.d1,c.w0,c.w1,c.upper);add(mid,c.eta1,c.d0,c.d1,c.w0,c.w1,c.upper)
        elif axis==1:
            mid=(c.d0+c.d1)/2
            if mid in (c.d0,c.d1):heapq.heappush(queue,(-c.upper,counter,c));break
            add(c.eta0,c.eta1,c.d0,mid,c.w0,c.w1,c.upper);add(c.eta0,c.eta1,mid,c.d1,c.w0,c.w1,c.upper)
        else:
            mid=(c.w0+c.w1)/2
            if mid in (c.w0,c.w1):heapq.heappush(queue,(-c.upper,counter,c));break
            add(c.eta0,c.eta1,c.d0,c.d1,c.w0,mid,c.upper);add(c.eta0,c.eta1,c.d0,c.d1,mid,c.w1,c.upper)
    if not queue:raise ArithmeticError('Complete cover unexpectedly empty')
    return dict(upper=-queue[0][0],boxes_evaluated=counter,query_profiles=query_profiles,
        calibration_profiles=len(cache),calibration_pruned_query_profiles=pruned,
        conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,
        shared_eta_range=[0.,cap],calibration_observed_bins=calibration_observed_bins,
        scope='Upper bound over a complete shared-eta/query-delay/query-width cover; calibration geometry enclosed, not plugged in')
