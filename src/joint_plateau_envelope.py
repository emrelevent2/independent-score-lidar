"""Cone query profile with exact calibration-plateau-aware subdivision.

When the calibration mode is feasible for EVERY eta in a cell, its profile
is constant there. The cone already profiles a single shared query eta over
that entire interval. In exact arithmetic,eta subdivision cannot tighten it;
refine query geometry instead. The complete eta domain is still retained.
"""
import heapq
import numpy as np
from flint import arb
from joint_calibration_envelope import Cell
from joint_cone_envelope import query_cone_basis
from conditional_calibration import calibration_upper
from linear_envelope_profile import profile_upper
from envelope_evidence import _observations,outward_float


def calibration_flat(k,r,eta0,eta1,cf):
    k,r=int(k),int(r)
    if not(0<=k<=r and 0<=eta0<=eta1<=.25):raise ValueError('Invalid plateau interval/counts')
    if r==0 or not cf['informative']:return True
    fail=r-k
    if fail==0:return False
    mode=(arb(k)/fail).log1p() if k else arb(0)
    # The minimum possible hazard increases with eta; so does its maximum.
    # The binomial mode must be attainable at both extreme ends of the cell.
    lower_at_right=arb(cf['signal_main_lower'])*(arb(float(eta1))+arb(cf['forward_leakage_lower']))+11*arb(cf['background_lower'])
    upper_at_left=arb(cf['signal_main_upper'])*(arb(float(eta0))+arb(cf['forward_leakage_upper']))+11*arb(cf['background_upper'])
    return bool(mode>=lower_at_right and mode<=upper_at_left)


def plateau_shared_cover(query,k,r,confidence,query_cycles=1000,cap=.25,
                         initial_eta=8,initial_delay=16,max_boxes=1024,
                         decision_level=None):
    query=np.asarray(query)
    if not(0<cap<=.25 and initial_eta>=1 and initial_delay>=1 and max_boxes>=initial_eta*initial_delay):raise ValueError('Invalid plateau-cover settings')
    _,_,_,saturated=_observations(query,query_cycles,'first')
    cache={};queue=[];counter=profiles=fallbacks=optimizer_failures=pruned=0;skipped=0;splits=[0,0,0]
    def cal_info(e0,e1):
        nonlocal fallbacks
        key=(e0,e1)
        if key not in cache:
            info=calibration_upper(k,r,e0,e1,confidence)
            info['flat_for_entire_eta_interval']=calibration_flat(k,r,e0,e1,confidence)
            cache[key]=info;fallbacks+=not info.get('profile_certified',False)
        return cache[key]
    def add(e0,e1,d0,d1,w0,w1,parent=None):
        nonlocal counter,profiles,fallbacks,optimizer_failures,pruned
        info=cal_info(e0,e1);cu=info['upper'];upper=outward_float(arb(cu)+arb(saturated))
        if decision_level is not None and upper<=decision_level:pruned+=1
        else:
            lo,hi=query_cone_basis(len(query),d0,d1,w0,w1,e0,e1)
            a=max(float(query.sum())*.35,1.)/(5*1.0645);b=max(float(query.sum())*.3,1.)
            fit=profile_upper(query,lo,hi,query_cycles,[a,a,b]);profiles+=1
            upper=outward_float(arb(cu)+arb(fit['upper']));fallbacks+=not fit['tangent_certified'];optimizer_failures+=not fit['optimizer_success']
        if parent is not None:upper=min(upper,parent)
        counter+=1;c=Cell(e0,e1,d0,d1,w0,w1,upper);heapq.heappush(queue,(-upper,counter,c))
    etas=np.linspace(0.,cap,initial_eta+1);delays=np.linspace(0.,len(query)-1.,initial_delay+1)
    for e0,e1 in zip(etas[:-1],etas[1:]):
        for d0,d1 in zip(delays[:-1],delays[1:]):add(float(e0),float(e1),float(d0),float(d1),4.5,5.5)
    while queue and counter+2<=max_boxes:
        if decision_level is not None and -queue[0][0]<=decision_level:break
        _,_,c=heapq.heappop(queue);mean=(c.w0+c.w1)/2
        spans=[(c.eta1-c.eta0)/(cap/16),(c.d1-c.d0)/mean,3*(c.w1-c.w0)/mean]
        if cal_info(c.eta0,c.eta1)['flat_for_entire_eta_interval']:
            skipped+=int(np.argmax(spans)==0);spans[0]=0.
        axis=int(np.argmax(spans));splits[axis]+=1
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
    if not queue:raise ArithmeticError('Complete plateau cover unexpectedly empty')
    return dict(upper=-queue[0][0],boxes_evaluated=counter,query_profiles=profiles,calibration_profiles=len(cache),
        calibration_pruned_query_profiles=pruned,conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,
        calibration_plateau_eta_cells=sum(v['flat_for_entire_eta_interval'] for v in cache.values()),
        redundant_eta_splits_skipped=skipped,splits=dict(eta=splits[0],delay=splits[1],width=splits[2]),
        shared_eta_range=[0.,cap],scope='Same confidence-box coupled null; exact endpoint cone and checked calibration-plateau geometry refinement')
