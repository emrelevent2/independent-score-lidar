"""Conditionally scored calibration lobe, with independently bounded nuisances.

Only the high-count main and dark windows form the training confidence set.
The other half's lobe count is scored conditionally on its first-event risk
set. Its predictive beta-binomial density is proper for every risk-set size.
Twofold averaging requires a UNION failure budget for the two training sets;
the average is not an e-value conditional on all observed calibration data.
"""
from fractions import Fraction
import numpy as np
from flint import arb
from calibration_confidence import window_counts
from calibration_confidence_v2 import binomial_endpoint
from background_lobe_confidence import shape_bounds
from envelope_evidence import gaussian_envelope,outward_float


def nuisance_confidence(h,cycles,delta=Fraction(1,2000),cap=.25):
    delta=Fraction(delta)
    if not 0<delta<1:raise ValueError('Invalid nuisance confidence budget')
    counts=dict(main=window_counts(h,cycles,27,37),dark=window_counts(h,cycles,128,255))
    spec=[('main_lower','main',False),('main_upper','main',True),('dark_lower','dark',False),('dark_upper','dark',True)]
    endpoints={name:binomial_endpoint(*counts[window],delta/4,upper) for name,window,upper in spec}
    result=dict(informative=False,delta=str(delta),one_sided_budget=str(delta/4),
        endpoints=endpoints,window_counts={k:list(v) for k,v in counts.items()},
        scope='Training-half bounds on main-window signal hazard and uniform per-bin background; no lobe-count bound')
    if endpoints['main_upper']>=1 or endpoints['dark_upper']>=1:return result
    geometry=shape_bounds(cap=cap)
    h0l=-(-arb(endpoints['main_lower'])).log1p();h0u=-(-arb(endpoints['main_upper'])).log1p()
    hdl=-(-arb(endpoints['dark_lower'])).log1p();hdu=-(-arb(endpoints['dark_upper'])).log1p()
    beta_u=hdu/128;a_u=h0u/arb(geometry['main_mass_lower'])
    beta_l=(hdl-a_u*arb(geometry['dark_response_mass_upper']))/128
    beta_l=beta_l if beta_l>0 else arb(0)
    signal_l=(h0l-11*beta_u)/(1+arb(float(cap))*arb(geometry['backward_leakage_ratio_upper']))
    signal_l=signal_l if signal_l>0 else arb(0)
    lo,hi=gaussian_envelope(256,31.5,32.5,4.5,5.5,False)
    al=sum((arb(float(x)) for x in lo[27:38]),arb(0));au=sum((arb(float(x)) for x in hi[27:38]),arb(0))
    forward_l=sum((arb(float(x)) for x in lo[43:54]),arb(0))/au
    forward_u=sum((arb(float(x)) for x in hi[43:54]),arb(0))/al
    result.update(informative=True,signal_main_lower=max(0.,outward_float(signal_l,False)),signal_main_upper=outward_float(h0u),
        background_lower=max(0.,outward_float(beta_l,False)),background_upper=outward_float(beta_u),
        forward_leakage_lower=max(0.,outward_float(forward_l,False)),forward_leakage_upper=outward_float(forward_u))
    if result['signal_main_lower']>result['signal_main_upper'] or result['background_lower']>result['background_upper']:
        result['informative']=False
    return result


def predictive_log_interval(train_k,train_r,test_k,test_r):
    if not (0<=train_k<=train_r and 0<=test_k<=test_r):raise ValueError('Invalid calibration risk/counts')
    a=arb(int(train_k))+arb(1)/2;b=arb(int(train_r-train_k))+arb(1)/2
    def logbeta(x,y):return x.lgamma()+y.lgamma()-(x+y).lgamma()
    logq=logbeta(a+int(test_k),b+int(test_r-test_k))-logbeta(a,b)
    # Both calibration numerator and binomial null omit the SAME choose(R,K).
    return dict(lower=outward_float(logq,False),upper=outward_float(logq),
                prior='Jeffreys beta(1/2,1/2)',binomial_constant_omitted=True)


def saturated_binomial(k,r):
    fail=r-k
    if k and fail:
        optimum=(arb(k)/fail).log1p()
        return k*(-(-optimum).expm1()).log()-fail*optimum
    return arb(0)


def binomial_interval_upper(k,r,rate_lower,rate_upper):
    k,r=int(k),int(r)
    if not 0<=k<=r:raise ValueError('Invalid validation lobe count/risk')
    low=arb(max(0.,outward_float(arb(rate_lower),False)))
    high=arb(outward_float(arb(rate_upper)));fail=r-k
    if not(low>=0 and high>=low):raise ValueError('Invalid lobe-hazard enclosure')
    saturated=saturated_binomial(k,r)
    if r==0:return dict(upper=0.,profile_certified=True,bound_kind='analytic-binomial-profile')
    if fail==0:rate=high
    else:
        optimum=(arb(k)/fail).log1p() if k else arb(0)
        if optimum<low:rate=low
        elif optimum>high:rate=high
        elif optimum>=low and optimum<=high:rate=optimum
        else:return dict(upper=outward_float(saturated),profile_certified=False,bound_kind='saturated-binomial')
    value=arb(0)
    if k:value+=k*(-(-rate).expm1()).log()
    value-=fail*rate
    return dict(upper=min(outward_float(saturated),outward_float(value)),profile_certified=True,bound_kind='analytic-binomial-profile')


def calibration_upper(k,r,eta0,eta1,confidence):
    if not confidence['informative']:
        # Saturated binomial bound, not an invented finite nuisance interval.
        return dict(upper=outward_float(saturated_binomial(int(k),int(r))),profile_certified=False,bound_kind='saturated-binomial')
    c=confidence
    low=arb(c['signal_main_lower'])*(arb(float(eta0))+arb(c['forward_leakage_lower']))+11*arb(c['background_lower'])
    high=arb(c['signal_main_upper'])*(arb(float(eta1))+arb(c['forward_leakage_upper']))+11*arb(c['background_upper'])
    return binomial_interval_upper(k,r,low,high)
