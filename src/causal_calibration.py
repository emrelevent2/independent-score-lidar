"""Proper full-lobe prediction conditional on the earlier calibration prefix.

Only bins0..42 train calibration Q; bins43..53 are scored in FULL. Conditional
on the complete past and its surviving-cycle risk,the future lobe count is
binomial. Donor-only eta may inform a fixed prior. Query Q may subsequently
use ALL calibration,by ordered normalization. No new probability theorem.
"""
from fractions import Fraction
import numpy as np
from flint import arb
from physical_calibration_predictive import fixed_hazard_log
from envelope_evidence import outward_float


def prefix_features(prefix,cycles,donor_eta):
    h=np.asarray(prefix)
    if h.shape!=(43,) or np.any(h<0) or np.any(h!=np.floor(h)) or h.sum()>cycles or not(0<=donor_eta<=.25):raise ValueError('Only the earlier43bins and donor-only eta are admissible')
    failure=int(cycles)-np.cumsum(h);kp=int(h[:20].sum());fp=int(failure[:20].sum())
    p=(arb(kp)+arb(1)/2)/(arb(kp+fp)+1);beta=-(-p).log1p()
    km=int(h[27:38].sum());rm=int(cycles-h[:27].sum())
    pm=(arb(km)+arb(1)/2)/(arb(rm)+1);signal=-(-pm).log1p()-11*beta
    signal=signal if signal>0 else arb(0)
    t=np.arange(256);g=np.exp(-4*np.log(2)*((t-32.)/5.)**2);F=float(g[43:54].sum()/g[27:38].sum())
    return dict(main_signal=max(0.,float(signal)),background_hazard=float(beta),forward_leakage=F,
        main_count=km,main_risk=rm,early_background_count=kp,early_background_failures=fp,
        scored_lobe_risk=int(cycles-h.sum()),donor_eta=float(donor_eta),training_bins=[0,42],scored_bins=[43,53],
        scope='Prefix-only point features for a proper predictive mixture;not calibrated physical parameter estimates')


def score_lobe(prefix,cycles,donor_eta,k):
    f=prefix_features(prefix,cycles,donor_eta);r=f['scored_lobe_risk'];k=int(k)
    if not 0<=k<=r:raise ValueError('Invalid scored lobe count')
    etas=[(0.,Fraction(1,2)),(float(donor_eta),Fraction(1,4))]+[((2*j+1)/64.,Fraction(1,32)) for j in range(8)]
    backgrounds=[(.5,Fraction(1,4)),(1.,Fraction(1,2)),(2.,Fraction(1,4))]
    components=[]
    for eta,we in etas:
        for factor,wb in backgrounds:
            rate=arb(f['main_signal'])*(arb(eta)+arb(f['forward_leakage']))+11*arb(f['background_hazard'])*arb(factor)
            logq=fixed_hazard_log(k,r,rate);w=we*wb
            components.append(dict(eta=eta,background_factor=factor,weight=str(w),log_probability=logq))
    assert sum((Fraction(c['weight']) for c in components),Fraction(0))==1
    def mix(endpoint,up):
        terms=[arb(c['log_probability'][endpoint])+(arb(Fraction(c['weight']).numerator)/Fraction(c['weight']).denominator).log() for c in components]
        shift=max(float(x) for x in terms);value=arb(shift)+sum(((x-arb(shift)).exp() for x in terms),arb(0)).log();return outward_float(value,up)
    return dict(lower=mix('lower',False),upper=mix('upper',True),features=f,components=components,scored_count=k,scored_risk=r,
        binomial_constant_omitted=True,scope='All lobe calibration cycles scored;Qcal sees only earlier prefix and independent donors;positive exact rational mixture weights')
