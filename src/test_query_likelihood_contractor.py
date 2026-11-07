import numpy as np
from flint import arb
from conditional_calibration import saturated_binomial
from envelope_evidence import exact_gaussian,certified_log_likelihood,_observations
from pilot import first_detection_probabilities
from coupled_corner_profile import amplitude_cap
from query_likelihood_contractor import gaussian_mass_upper,hazard_interval,phi,zero_background_cap,geometry_contract,contract_amplitude,signal_window
from contracted_corner_envelope import contracted_cover

CF=dict(informative=True,signal_main_lower=.04,signal_main_upper=.06,background_lower=.00001,background_upper=.00003,forward_leakage_lower=0.,forward_leakage_upper=.000002)


def fixture():
    def rates(a,eta,b,d,w):
        g=exact_gaussian(64,d,w,False);s=exact_gaussian(80,arb(d)+16,w,False)[:64]
        return [(arb(a)*(x+arb(eta)*y)+arb(b)/64)/1000 for x,y in zip(g,s)]
    rng=np.random.default_rng(49003);h=rng.multinomial(1000,first_detection_probabilities([float(x) for x in rates(20.,.1,5.,31.4,5.2)]))[:-1]
    return h,rates


def test_checked_hazard_intervals_exclude_only_low_likelihoods():
    for k,r,D in [(0,1000,10.),(10,1000,3.),(300,900,20.),(500,500,5.)]:
        lo,hi=hazard_interval(k,r,D);target=saturated_binomial(k,r)-D
        if lo>0:assert phi(k,r,lo)<=target
        if np.isfinite(hi):assert phi(k,r,hi)<=target
        for H in np.geomspace(1e-5,10.,300):
            if phi(k,r,float(H))>target:assert lo<=H<=hi


def test_mass_and_window_contractors_retain_above_threshold_physical_point():
    h,rates=fixture();counts,remaining,_,sat=_observations(h,1000,'first')
    cal=saturated_binomial(100,20000)
    # Actual calibration hazard in the confidence box,not its saturated fit.
    cr=arb(.05)*(arb(.1)+arb(.000001))+11*arb(.00002)
    joint=arb(certified_log_likelihood(h,rates(20.,.1,5.,31.4,5.2),1000,'first'))+100*(-(-cr).expm1()).log()-(20000-100)*cr
    level=float(joint)-20;A=amplitude_cap(h,1000,100,20000,level)['cap'];B=zero_background_cap(h,remaining,1000,sat,float(cal),level)['background_cap'];assert B>=5.
    W=signal_window(h);info=geometry_contract(h,remaining,1000,100,20000,level,A,B,W,31.3,31.5,5.1,5.3)
    assert not info['empty_above_level'];restricted=contract_amplitude(info,0.,A,.04,.14,B,W[1]-W[0],64,1000)
    assert restricted is not None and restricted[0]<=20<=restricted[1]
    for d in [0.,.5,31.4,63.,79.]:assert sum(exact_gaussian(64,d,5.5,False),arb(0))<=gaussian_mass_upper(64)


def test_full_contracted_cover_accounts_for_excluded_domains():
    h,rates=fixture();level=-1500.
    result=contracted_cover(h,100,20000,CF,initial_eta=4,initial_delay=4,max_boxes=64,decision_level=level)
    assert result['upper']>=level
    rng=np.random.default_rng(998)
    for _ in range(40):
        a,eta,b,d,w=rng.uniform(0,600),rng.uniform(0,.25),rng.uniform(0,200),rng.uniform(0,63),rng.uniform(4.5,5.5)
        ll=arb(certified_log_likelihood(h,rates(a,eta,b,d,w),1000,'first',upper=True));cr=arb(.05)*(arb(eta)+arb(.000001))+11*arb(.00002)
        ll+=100*(-(-cr).expm1()).log()-(20000-100)*cr
        assert arb(result['upper'])>=ll
