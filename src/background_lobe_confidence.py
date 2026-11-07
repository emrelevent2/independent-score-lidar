"""Five-endpoint optical-lobe confidence bound with bounded background.

The old three-endpoint construction retained every lobe-window count as
possible signal. This version also obtains a background lower bound, after
bounding nonzero Gaussian leakage into the dark window. Five conditional
binomial endpoints share the SAME total delta. This is an application of
standard confidence-set/union-bound reasoning, not a new coverage principle.
"""
from fractions import Fraction
from functools import lru_cache
import numpy as np
from flint import arb
from calibration_confidence import window_counts,leakage_ratio_upper
from calibration_confidence_v2 import binomial_endpoint
from envelope_evidence import gaussian_envelope,outward_float


@lru_cache(maxsize=64)
def shape_bounds(bins=256,shift=16,cap=.25,main=(27,37),dark=(128,255),
                 delay=(31.5,32.5),width=(4.5,5.5)):
    lo,hi=gaussian_envelope(bins,*delay,*width,normalized=False)
    sd0=max(0.,outward_float(arb(float(delay[0]))+shift,False))
    sd1=min(float(bins+shift-1),outward_float(arb(float(delay[1]))+shift))
    _,side_hi=gaussian_envelope(bins+shift,sd0,sd1,*width,normalized=False)
    main_lower=sum((arb(float(x)) for x in lo[main[0]:main[1]+1]),arb(0))
    dark_upper=sum((arb(float(x))+arb(float(cap))*arb(float(y))
                    for x,y in zip(hi[dark[0]:dark[1]+1],side_hi[dark[0]:dark[1]+1])),arb(0))
    if not main_lower>0:raise ValueError('No positive main-window envelope mass')
    return dict(main_mass_lower=outward_float(main_lower,False),
                dark_response_mass_upper=outward_float(dark_upper),
                backward_leakage_ratio_upper=leakage_ratio_upper(bins,shift,tuple(main),tuple(delay),tuple(width)))


def bound_from_hazards(main_lower,main_upper,lobe_upper,dark_lower,dark_upper,
                       geometry,main_size=11,dark_size=128,cap=.25):
    """Conservative algebra on certified one-sided hazard endpoints.

    Rate: a*(g+eta*shift(g))+beta, a,beta>=0. Let A be main-window
    main-pulse mass and D dark-window total response mass. Then a<=U0/A_L,
    beta >= (LD-a_U*D_U)/|WD|, beta<=UD/|WD|. Subtract only this
    certified background lower bound from the lobe-window upper hazard.
    """
    values=list(map(arb,[main_lower,main_upper,lobe_upper,dark_lower,dark_upper]))
    if not all(x>=0 for x in values):raise ValueError('Nonnegative hazard endpoints required')
    l0,u0,u1,ld,ud=values
    if not(0<main_size and 0<dark_size and 0<=cap<=1):raise ValueError('Invalid geometry')
    result=dict(eta_upper=float(cap),informative=False)
    a_lower=arb(float(geometry['main_mass_lower']))
    d_upper=arb(float(geometry['dark_response_mass_upper']))
    if not (a_lower>0 and d_upper>=0):raise ValueError('Invalid shape enclosure')
    amplitude_upper=u0/a_lower
    beta_upper=ud/dark_size
    beta_lower_candidate=(ld-amplitude_upper*d_upper)/dark_size
    # Arb interval ambiguity retains zero, rather than an unchecked positive
    # background subtraction. The same conservative rule covers no data.
    beta_lower=beta_lower_candidate if beta_lower_candidate>0 else arb(0)
    denominator=l0-main_size*beta_upper
    numerator=u1-main_size*beta_lower
    result.update(background_per_bin_lower=max(0.,outward_float(beta_lower,False)),
        background_per_bin_upper=outward_float(beta_upper),
        calibration_peak_height_upper=outward_float(amplitude_upper),
        main_signal_hazard_lower=outward_float(denominator,False),
        lobe_signal_hazard_upper=outward_float(numerator))
    # Inconsistent endpoints can occur off the joint confidence event. An
    # uninformative fallback is preferable to reporting a negative shape bound.
    if not denominator>0 or not numerator>=0:return result
    epsilon=arb(float(geometry['backward_leakage_ratio_upper']))
    upper=outward_float(numerator*(1+arb(float(cap))*epsilon)/denominator)
    result.update(eta_upper=min(float(cap),upper),informative=upper<float(cap))
    return result


def lobe_confidence(h,cycles,delta=Fraction(1,1000),cap=.25,shift=16,
                    main=(27,37),dark=(128,255),delay=(31.5,32.5),width=(4.5,5.5)):
    h=np.asarray(h);bins=len(h);delta=Fraction(delta)
    if not(0<delta<1 and 0<=cap<=1 and isinstance(shift,int) and shift>0):raise ValueError('Invalid calibration settings')
    lobe=(main[0]+shift,main[1]+shift)
    if not(main[1]<lobe[0] and lobe[1]<dark[0]):raise ValueError('Separated ordered windows required')
    counts={name:window_counts(h,cycles,*w) for name,w in [('main',main),('lobe',lobe),('dark',dark)]}
    specification=[('main_lower','main',False),('main_upper','main',True),
                   ('lobe_upper','lobe',True),('dark_lower','dark',False),('dark_upper','dark',True)]
    endpoints={name:binomial_endpoint(*counts[window],delta/5,upper) for name,window,upper in specification}
    result=dict(eta_upper=float(cap),informative=False,delta=str(delta),one_sided_budget=str(delta/5),
        endpoint_count=5,cap=float(cap),cycles=int(cycles),detected=int(h.sum()),
        window_counts={k:list(v) for k,v in counts.items()},probability_endpoints=endpoints,
        scope='Model-conditional five-endpoint optical-lobe confidence with uniform-background lower and upper bounds')
    if any(endpoints[k]>=1 for k in ['main_upper','lobe_upper','dark_upper']):return result
    hazards={name:-(-arb(value)).log1p() for name,value in endpoints.items()}
    geometry=shape_bounds(bins,shift,cap,tuple(main),tuple(dark),tuple(delay),tuple(width))
    bound=bound_from_hazards(hazards['main_lower'],hazards['main_upper'],hazards['lobe_upper'],
        hazards['dark_lower'],hazards['dark_upper'],geometry,main[1]-main[0]+1,dark[1]-dark[0]+1,cap)
    result.update(bound,geometry=geometry)
    return result
