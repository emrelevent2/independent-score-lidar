"""Threshold-safe query likelihood contraction for null optimization.

Zero-event bins bound background. A window's aggregated binomial likelihood
and a certified outside-window profile bound the primary amplitude. These are
deterministic likelihood inequalities,not new confidence events. Parameters
removed by the contractor have joint likelihood <=the supplied decision level;
the final cover MUST retain that level as a global upper-bound floor.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar,brentq
from scipy.ndimage import convolve1d
from flint import arb
from coupled_corner_profile import query_basis,query_tangent,gaussian_mass_lower
from linear_envelope_profile import value_gradient
from conditional_calibration import saturated_binomial
from envelope_evidence import outward_float


@lru_cache(maxsize=128)
def gaussian_mass_upper(bins,max_width=5.5):
    # Closest integer contributes <=1; each other integer at offsetj has
    # distance >=j-.5,with at mosttwo integers at each positive offset.
    k=4*arb(2).log()
    return arb(1)+2*sum(((-k*((arb(j)-arb(1)/2)/arb(float(max_width)))**2).exp() for j in range(1,bins+17)),arb(0))


def phi(k,r,H):
    value=-(r-k)*arb(H)
    if k:value+=k*(-(-arb(H)).expm1()).log()
    return value


def hazard_interval(k,r,deviance):
    """Enclose the binomial-hazard superlevel set using checked endpoints."""
    if not(0<=k<=r and deviance>=0):raise ValueError('Invalid window/deviance')
    if not r or deviance==0 or k==r or not np.isfinite(deviance):return 0.,float('inf')
    target=saturated_binomial(k,r)-arb(float(deviance));mode=float((arb(k)/(r-k)).log1p()) if k else 0.
    if not k:return 0.,outward_float(arb(float(deviance))/r)
    target_float=float(target)
    def f(x):return k*np.log(-np.expm1(-x))-(r-k)*x-target_float
    if f(mode)<=0:return 0.,float('inf')
    lower=brentq(f,1e-300,mode,xtol=1e-13) if f(1e-300)<0 else 0.;upper=max(1.,mode*2)
    while f(upper)>0:upper*=2
    upper=brentq(f,mode,upper,xtol=1e-13)
    def checked(proposal,isupper):
        margin=max(1e-12,abs(proposal)*1e-10)
        for _ in range(64):
            value=max(0.,proposal-margin) if not isupper else proposal+margin
            if (isupper and arb(value)>=arb(mode) or not isupper and arb(value)<=arb(mode)) and phi(k,r,value)<=target:return value
            margin*=2
        return float('inf') if isupper else 0.
    return checked(lower,False),checked(upper,True)


def subset_saturated(h,remaining):
    value=arb(0)
    for k,f in zip(h,remaining):
        k,f=int(k),int(f)
        if k and f:
            H=(arb(k)/f).log1p();value+=k*(-(-H).expm1()).log()-f*H
    return value


def zero_background_cap(h,remaining,cycles,query_sat,cal_sat,level):
    D=outward_float(arb(query_sat)+arb(cal_sat)-arb(float(level)))
    zeros=sum(int(f) for k,f in zip(h,remaining) if not k)
    if D<0:return dict(entire_domain_below_level=True,upper=outward_float(arb(query_sat)+arb(cal_sat)))
    if not zeros:return None
    B=outward_float(arb(D)*cycles*len(h)/zeros)
    return dict(entire_domain_below_level=False,background_cap=max(B,1e-10),deviance_budget=D,zero_bin_failure_sum=zeros)


def signal_window(h):
    kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    center=int(np.argmax(convolve1d(np.asarray(h,float),kernel,mode='constant')))
    return max(0,center-8),min(len(h),center+25)


def outside_profile(h,remaining,lo,hi,A,B,window,cycles):
    mask=np.ones(len(h),bool);mask[window[0]:window[1]]=False
    low=np.column_stack([np.zeros(mask.sum()),np.zeros(mask.sum()),lo[mask,2]])
    # a<=A and eta<=.25; the offset is fixed,not an invented free nuisance.
    leakage=[outward_float(arb(A)*(arb(float(x))+arb(.25)*arb(float(y)))) for x,y in zip(hi[mask,0],hi[mask,1])]
    high=np.column_stack([leakage,np.zeros(mask.sum()),hi[mask,2]])
    hs,fs=h[mask],remaining[mask]
    saturated=outward_float(subset_saturated(hs,fs))
    if not mask.any():return dict(upper=0.,tangent_certified=True)
    def objective(b):return -value_gradient([1.,0.,b],hs,fs,low,high,cycles)[0]
    fit=minimize_scalar(objective,bounds=(1e-10,max(B,2e-10)),method='bounded',options={'xatol':1e-10,'maxiter':100})
    b=float(np.clip(fit.x,1e-10,B));t=query_tangent([1.,0.,b],hs,fs,low,high,cycles)
    if t is None:return dict(upper=saturated,tangent_certified=False)
    value,g=t;gb=g[2];candidates=[value-gb*arb(b),value+gb*(arb(B)-arb(b))]
    return dict(upper=min(saturated,max(outward_float(x) for x in candidates)),tangent_certified=True)


def geometry_contract(h,remaining,cycles,k,r,level,A,B,window,d0,d1,w0,w1):
    lo,hi=query_basis(len(h),d0,d1,w0,w1)
    outside=outside_profile(h,remaining,lo,hi,A,B,window,cycles)
    start,stop=window;inside=subset_saturated(h[start:stop],remaining[start:stop]);cs=saturated_binomial(k,r)
    D=outward_float(arb(outside['upper'])+inside+cs-arb(float(level)))
    result=dict(outside=outside,window=list(window),deviance_budget=D)
    if D<0:return dict(**result,empty_above_level=True)
    kw=int(h[start:stop].sum());rw=int(cycles-h[:start].sum());HL,HU=hazard_interval(kw,rw,D)
    gm=gaussian_mass_lower(len(h));gu=gaussian_mass_upper(len(h))
    def masses(column,can_use_global_lower):
        lower=sum((arb(float(x)) for x in lo[start:stop,column]),arb(0));upper=sum((arb(float(x)) for x in hi[start:stop,column]),arb(0))
        if can_use_global_lower:
            outside_upper=sum((arb(float(x)) for x in np.r_[hi[:start,column],hi[stop:,column]]),arb(0))
            candidate=gm-outside_upper
            if candidate>lower:lower=candidate
        upper=upper if upper<gu else gu
        return max(0.,outward_float(lower,False)),outward_float(upper)
    gl,gh=masses(0,True);sl,sh=masses(1,bool(arb(d1)+16<=len(h)-1))
    return dict(**result,empty_above_level=False,hazard_interval=[HL,HU],main_mass=[gl,gh],side_mass=[sl,sh])


def contract_amplitude(info,a0,a1,e0,e1,B,window_length,bins,cycles):
    if info['empty_above_level']:return None
    HL,HU=info['hazard_interval'];gl,gh=info['main_mass'];sl,sh=info['side_mass']
    lower=arb(float(gl))+arb(float(e0))*arb(float(sl));upper=arb(float(gh))+arb(float(e1))*arb(float(sh))
    lo,hi=a0,a1
    if np.isfinite(HU) and lower>0:hi=min(hi,outward_float(cycles*arb(HU)/lower))
    if HL>0 and upper>0:
        candidate=(cycles*arb(HL)-arb(B)*window_length/bins)/upper
        lo=max(lo,outward_float(candidate,False))
    return None if lo>=hi else (max(0.,lo),hi)
