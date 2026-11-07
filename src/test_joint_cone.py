from fractions import Fraction
import numpy as np
from flint import arb
from joint_cone_envelope import query_cone_basis,cone_shared_cover
from envelope_evidence import exact_gaussian,certified_log_likelihood
from conditional_calibration import calibration_upper
from pilot import first_detection_probabilities


def test_eta_interval_positive_cone_is_exact_and_encloses_geometry():
    lower,upper=query_cone_basis(64,30.2,32.1,4.5,5.5,.03125,.125)
    for eta in [Fraction(1,32),Fraction(1,12),Fraction(1,8)]:
        l,u=Fraction(1,32),Fraction(1,8);a=Fraction(7)
        x=a*(u-eta)/(u-l);y=a*(eta-l)/(u-l)
        assert x+y==a and x*l+y*u==a*eta and x>=0 and y>=0
    for d,w in [(30.2,4.5),(31.73,5.13),(32.1,5.5)]:
        g=exact_gaussian(64,d,w,False);side=exact_gaussian(80,arb(d)+16,w,False)[:64]
        for eta,j in [(.03125,0),(.125,1)]:
            for t,(x,y) in enumerate(zip(g,side)):
                v=x+arb(eta)*y;assert v>=arb(float(lower[t,j])) and v<=arb(float(upper[t,j]))


def test_stopped_cone_cover_contains_physical_joint_null_witness():
    rng=np.random.default_rng(7520);eta=.0625
    g=exact_gaussian(64,31.73,5.13,False);side=exact_gaussian(80,arb(31.73)+16,5.13,False)[:64]
    rates=[(arb(10)*(x+arb(eta)*y)+arb(5)/64)/1000 for x,y in zip(g,side)]
    h=rng.multinomial(1000,first_detection_probabilities([float(x) for x in rates]))[:-1]
    cf=dict(informative=True,signal_main_lower=.04,signal_main_upper=.06,background_lower=.00001,background_upper=.00003,
        forward_leakage_lower=0.,forward_leakage_upper=.000002)
    calrate=arb(.05)*(arb(eta)+arb(.000001))+11*arb(.00002)
    k=int(rng.binomial(500,float(-(-calrate).expm1())))
    ll=arb(certified_log_likelihood(h,rates,1000,'first',upper=True))
    if k:ll+=k*(-(-calrate).expm1()).log()
    ll-=(500-k)*calrate
    provider=lambda e0,e1:calibration_upper(k,500,e0,e1,cf)
    bound=cone_shared_cover(h,provider,initial_eta=2,initial_delay=4,max_boxes=16)
    assert arb(bound['upper'])>=ll
