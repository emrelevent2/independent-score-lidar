"""A proper donor-dependent spatial + full-delay alternative mixture.

All component distributions and weights are fixed by donors before the query
is scored. Query-adaptive sparse evaluation RETAINS original mixture weights:
the selected sum is a pointwise lower bound on that proper fixed mixture.
Renormalizing selected weights would invalidate that argument.
"""
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp
from flint import arb
from continuous_alternative import amplitude_components
from support_routing import donor_support
from envelope_evidence import certified_log_likelihood,outward_float


def coverage_components(proposal,bins=256,cycles=1000,minimum_separation=8.,amplitudes=(1.,2.,4.,8.,16.,32.)):
    t=np.arange(bins);w=float(proposal['fitted_fwhm']);primary=float(proposal['primary'])
    if not(0<=primary<=bins-1 and 4.5<=w<=5.5):raise ValueError('Invalid donor primary/width')
    if min(proposal[k] for k in ['primary_incident','secondary_incident','background_incident'])<0:raise ValueError('Negative donor count')
    support=donor_support(proposal,bins,cycles);spatial_mass=Fraction(9,10) if support>=3.5 else Fraction(1,10)
    spatial=np.stack(amplitude_components(proposal,bins,cycles));delays=t[(t>=4)&(t<bins-4)&(np.abs(t-primary)>=minimum_separation)]
    if not len(delays) or not len(amplitudes):raise ValueError('No full-delay component support')
    g=np.exp(-4*np.log(2)*((t-primary)/w)**2);g/=g.sum()
    base=proposal['primary_incident']*g+proposal['background_incident']/bins
    pulse=np.exp(-4*np.log(2)*((t[None,:]-delays[:,None])/w)**2);pulse/=pulse.sum(1,keepdims=True)
    amplitudes=np.asarray(amplitudes,float)
    if np.any(amplitudes<=0) or not np.isfinite(amplitudes).all():raise ValueError('Positive finite amplitude grid required')
    dense=np.maximum((base[None,None,:]+amplitudes[None,:,None]*pulse[:,None,:])/cycles,1e-12).reshape(-1,bins)
    weights=[spatial_mass/4]*4+[(1-spatial_mass)/(len(delays)*len(amplitudes))]*len(dense)
    assert sum(weights,Fraction(0))==1
    locations=np.r_[np.full(4,proposal['secondary']),np.repeat(delays,len(amplitudes))]
    return dict(rates=np.vstack([spatial,dense]),weights=weights,locations=locations,
        spatial_components=4,global_delays=delays,amplitudes=amplitudes,support_proxy=support,
        spatial_mass=spatial_mass,raw_center_excluded=proposal['raw_center_excluded'])


def numeric_scores(h,models,cycles=1000):
    h=np.asarray(h);rates=models['rates']
    if h.ndim!=1 or rates.shape[1]!=len(h) or h.sum()>cycles or np.any(h<0):raise ValueError('Invalid first-event query')
    remaining=cycles-np.cumsum(h)
    likelihood=np.log(-np.expm1(-rates))@h-rates@remaining
    logw=np.log(np.array([float(w) for w in models['weights']]))
    weighted=likelihood+logw
    return likelihood,weighted,float(logsumexp(weighted))


def selected_indices(weighted,global_top=16):
    if global_top<1:raise ValueError('At least one global component must be retained')
    indices=np.argsort(weighted[4:])[::-1][:global_top]+4
    return np.r_[np.arange(4),indices].astype(int)


def weighted_log_lower(component_lower,weights,indices):
    indices=list(map(int,indices))
    if not indices or len(set(indices))!=len(indices):raise ValueError('Unique nonempty subset required')
    terms=[]
    for i in indices:
        w=weights[i]
        if not isinstance(w,Fraction) or w<=0:raise ValueError('Positive exact rational component weights required')
        terms.append(arb(float(component_lower[i]))+(arb(w.numerator)/w.denominator).log())
    shift=max(float(x) for x in terms)
    value=arb(shift)+sum(((x-arb(shift)).exp() for x in terms),arb(0)).log()
    return outward_float(value,False)


def sparse_numerator_lower(h,models,indices,cycles=1000):
    indices=list(map(int,indices));lower={i:certified_log_likelihood(h,models['rates'][i],cycles,'first') for i in indices}
    return dict(lower=weighted_log_lower(lower,models['weights'],indices),indices=indices,
        evaluated_components=len(indices),full_components=len(models['weights']),
        retained_prior_mass=str(sum((models['weights'][i] for i in indices),Fraction(0))),
        original_weights_retained=True,component_lower={str(i):v for i,v in lower.items()})


def location_report(models,weighted):
    spatial=float(logsumexp(weighted[:4]));j=len(models['global_delays']);k=len(models['amplitudes'])
    global_mass=logsumexp(weighted[4:].reshape(j,k),axis=1);index=int(np.argmax(global_mass))
    if spatial>=global_mass[index]:
        return dict(delay=float(models['locations'][0]),head='spatial')
    mode=float(models['global_delays'][index]);near=np.abs(models['global_delays']-mode)<=2
    probability=np.exp(global_mass[near]-logsumexp(global_mass[near]))
    return dict(delay=float(probability@models['global_delays'][near]),head='global',mode=mode)
