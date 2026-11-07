"""Decision-directed Gaussian null bounds,with an ordinary strong control.

The contracted backend uses only Gaussian delay/width subdivision. Zero-bin
likelihoods bound background;window mass and outside likelihoods contract
signal amplitude. All removed parameter domains remain covered by the supplied
decision-level floor. No statistical alpha or additional photons are spent.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize,minimize_scalar
from flint import arb
from envelope_evidence import (_observations,gaussian_envelope,exact_gaussian,
    bound_box,certified_log_likelihood,outward_float)
from coupled_corner_profile import query_tangent,amplitude_cap,gaussian_mass_lower
from query_likelihood_contractor import (zero_background_cap,subset_saturated,
    hazard_interval,gaussian_mass_upper)
from linear_envelope_profile import value_gradient


@dataclass
class Cell:
    d0:float;d1:float;w0:float;w1:float;upper:float;lower:float


def checked_witness(h,cycles,witness,widths):
    if witness is None:return -np.inf
    a,b,d,w=map(float,witness['parameters'])
    if not(all(np.isfinite([a,b,d,w])) and a>=0 and b>=0 and 0<=d<=len(h)-1 and widths[0]<=w<=widths[1]):raise ValueError('Feasible normalized-amplitude null witness required')
    g=exact_gaussian(len(h),d,w,True);rates=[(arb(a)*x+arb(b)/len(h))/cycles for x in g]
    return certified_log_likelihood(h,rates,cycles,'first')


def basis(bins,d0,d1,w0,w1):
    gl,gh=gaussian_envelope(bins,d0,d1,w0,w1,False)
    bl=max(0.,outward_float(arb(1)/bins,False));bu=outward_float(arb(1)/bins)
    return np.column_stack([gl,np.zeros(bins),np.full(bins,bl)]),np.column_stack([gh,np.zeros(bins),np.full(bins,bu)])


def window_contract(h,remaining,cycles,A,B,window,level,d0,d1,w0,w1):
    lo,hi=basis(len(h),d0,d1,w0,w1);start,stop=window
    mask=np.ones(len(h),bool);mask[start:stop]=False
    leak=[outward_float(arb(A)*arb(float(x))) for x in hi[mask,0]]
    lower=np.column_stack([np.zeros(mask.sum()),np.zeros(mask.sum()),lo[mask,2]])
    upper=np.column_stack([leak,np.zeros(mask.sum()),hi[mask,2]])
    hs,fs=h[mask],remaining[mask];outside=outward_float(subset_saturated(hs,fs));cert=False
    if mask.any():
        def objective(b):return -value_gradient([1.,0.,b],hs,fs,lower,upper,cycles)[0]
        fit=minimize_scalar(objective,bounds=(1e-10,max(B,2e-10)),method='bounded',options={'maxiter':100,'xatol':1e-10})
        b=float(np.clip(fit.x,1e-10,B));t=query_tangent([1.,0.,b],hs,fs,lower,upper,cycles)
        if t is not None:
            v,g=t;outside=min(outside,max(outward_float(v-g[2]*arb(b)),outward_float(v+g[2]*(arb(B)-arb(b)))));cert=True
    else:outside=0.;cert=True
    D=outward_float(arb(outside)+subset_saturated(h[start:stop],remaining[start:stop])-arb(level))
    result=dict(outside_upper=outside,outside_certified=cert,deviance_budget=D)
    if D<0:return dict(**result,empty=True)
    kw=int(h[start:stop].sum());rw=int(cycles-h[:start].sum());HL,HU=hazard_interval(kw,rw,D)
    gl=sum((arb(float(x)) for x in lo[start:stop,0]),arb(0));gh=sum((arb(float(x)) for x in hi[start:stop,0]),arb(0))
    outside_g=sum((arb(float(x)) for x in hi[mask,0]),arb(0));global_lower=gaussian_mass_lower(len(h));global_upper=gaussian_mass_upper(len(h))
    if global_lower-outside_g>gl:gl=global_lower-outside_g
    if global_upper<gh:gh=global_upper
    a0=0.;a1=A
    if np.isfinite(HU) and gl>0:a1=min(a1,outward_float(cycles*arb(HU)/gl))
    if HL>0 and gh>0:a0=max(0.,outward_float((cycles*arb(HL)-arb(B)*(stop-start)/len(h))/gh,False))
    return dict(**result,empty=a0>=a1,amplitude_range=[a0,a1],hazard_interval=[HL,HU])


def bounded_box(h,remaining,cycles,saturated,A,B,window,level,d0,d1,w0,w1):
    contract=window_contract(h,remaining,cycles,A,B,window,level,d0,d1,w0,w1)
    if contract['empty']:return Cell(d0,d1,w0,w1,level,-np.inf),dict(pruned=True,certified=True,optimizer_success=True,contract=contract)
    a0,a1=contract['amplitude_range'];lo,hi=basis(len(h),d0,d1,w0,w1)
    initial=[float(np.clip(max(float(h.sum())*.7,1.)/(5*1.0645),a0,a1)),float(np.clip(max(float(h.sum())*.3,1.),1e-10,B))]
    def objective(z):
        value,g=value_gradient([z[0],0.,z[1]],h,remaining,lo,hi,cycles);return -value,-np.array([g[0],g[2]])
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(a0,a1),(1e-10,B)],options={'maxiter':120,'ftol':1e-12,'gtol':1e-8})
    z=fit.x if np.all(np.isfinite(fit.x)) else initial;a=float(np.clip(z[0],a0,a1));b=float(np.clip(z[1],1e-10,B))
    t=query_tangent([a,0.,b],h,remaining,lo,hi,cycles);upper=saturated
    if t is not None:
        phi,g=t;C=phi-g[0]*arb(a)-g[2]*arb(b)
        upper=min(saturated,max(outward_float(C+g[0]*arb(aa)+g[2]*arb(bb)) for aa in [a0,a1] for bb in [0.,B]))
    g=exact_gaussian(len(h),(d0+d1)/2,(w0+w1)/2,False);rates=[(arb(a)*x+arb(b)/len(h))/cycles for x in g]
    lower=certified_log_likelihood(h,rates,cycles,'first')
    return Cell(d0,d1,w0,w1,max(level,upper),lower),dict(pruned=False,certified=t is not None,optimizer_success=bool(fit.success),contract=contract)


def decision_bound(h,level,backend='contracted',cycles=1000,fwhm_range=(4.5,5.5),initial_cells=32,max_boxes=256,gap_tolerance=.01,witness=None):
    counts,remaining,_,sat=_observations(h,cycles,'first');w0,w1=map(float,fwhm_range)
    if backend not in ['ordinary','contracted'] or not(np.isfinite(level) and 4.5<=w0<=w1<=5.5 and 1<=initial_cells<=max_boxes):raise ValueError('Valid threshold,backend,budget and trusted widths required')
    best=checked_witness(counts,cycles,witness,(w0,w1))
    if best>level:return dict(upper=None,feasible_lower=best,boxes_evaluated=0,decision='fixed_numerator_rejected_by_feasible_null',backend=backend)
    A=B=None;tail=None;window=None
    if backend=='contracted':
        bg=zero_background_cap(counts,remaining,cycles,sat,0.,level)
        if bg is None:return dict(upper=float('inf'),boxes_evaluated=0,decision='safe_unbounded_fallback',backend=backend)
        if bg['entire_domain_below_level']:return dict(upper=bg['upper'],boxes_evaluated=0,decision='accepted_by_saturation',backend=backend)
        B=bg['background_cap'];tail=amplitude_cap(counts,cycles,0,1,level)
        if tail is None:return dict(upper=float('inf'),boxes_evaluated=0,decision='safe_unbounded_fallback',backend=backend)
        A=tail['cap'];kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2);center=int(np.argmax(convolve1d(counts.astype(float),kernel,mode='constant')))
        window=(max(0,center-8),min(len(h),center+9))
    queue=[];counter=pruned=outside_profiles=profiles=fallbacks=optimizer_failures=0
    def add(d0,d1,f0,f1,parent=None):
        nonlocal counter,best,pruned,outside_profiles,profiles,fallbacks,optimizer_failures
        if backend=='ordinary':
            old=bound_box(counts,remaining,cycles,'first',d0,d1,f0,f1,sat,False);box=Cell(d0,d1,f0,f1,old.upper,old.lower);info=dict(certified=old.certified,optimizer_success=old.optimizer_success,pruned=False);profiles+=1
        else:
            box,info=bounded_box(counts,remaining,cycles,sat,A,B,window,level,d0,d1,f0,f1);outside_profiles+=1;profiles+=not info['pruned']
        if parent is not None:box.upper=min(box.upper,parent)
        if box.lower>box.upper:raise ArithmeticError('Cell upper contradicts feasible physical point')
        best=max(best,box.lower);pruned+=info['pruned'];fallbacks+=not info['certified'];optimizer_failures+=not info['optimizer_success'];counter+=1;heapq.heappush(queue,(-box.upper,counter,box))
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    for d0,d1 in zip(edges[:-1],edges[1:]):add(float(d0),float(d1),w0,w1)
    reason='budget_undecided'
    while queue:
        U=-queue[0][0]
        if backend=='contracted':U=max(level,tail['tail_upper'],U)
        if U<best:raise ArithmeticError('Full cover upper contradicts feasible null witness')
        if U<=level:reason='accepted';break
        if best>level:reason='fixed_numerator_rejected_by_feasible_null';break
        if U-best<=gap_tolerance:reason='gap_undecided';break
        if counter+2>max_boxes:break
        _,_,box=heapq.heappop(queue)
        if box.upper<best:continue
        ds=(box.d1-box.d0)/((box.w0+box.w1)/2);ws=3*(box.w1-box.w0)/((box.w0+box.w1)/2)
        if ds>=ws:
            mid=(box.d0+box.d1)/2
            if mid in (box.d0,box.d1):heapq.heappush(queue,(-box.upper,counter,box));reason='precision_undecided';break
            add(box.d0,mid,box.w0,box.w1,box.upper);add(mid,box.d1,box.w0,box.w1,box.upper)
        else:
            mid=(box.w0+box.w1)/2
            if mid in (box.w0,box.w1):heapq.heappush(queue,(-box.upper,counter,box));reason='precision_undecided';break
            add(box.d0,box.d1,box.w0,mid,box.upper);add(box.d0,box.d1,mid,box.w1,box.upper)
    upper=max(best,-queue[0][0]) if queue else sat
    if backend=='contracted':upper=max(level,tail['tail_upper'],upper)
    return dict(upper=upper,feasible_lower=None if not np.isfinite(best) else best,gap=None if not np.isfinite(best) else upper-best,boxes_evaluated=counter,query_profiles=profiles,outside_profiles=outside_profiles,pruned_boxes=pruned,conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,decision=reason,backend=backend,amplitude_cap=tail,background_cap=B,signal_window=window,
        scope='Full continuous Gaussian single-return family;contracted backend retains decision-level floor for all removed unbounded domains')
