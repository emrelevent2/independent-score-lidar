from fractions import Fraction
from math import comb
import numpy as np
from flint import arb
from conditional_calibration import nuisance_confidence,predictive_log_interval,binomial_interval_upper,calibration_upper
from envelope_evidence import exact_gaussian
from pilot import first_detection_probabilities


def rising(a,k):
    x=Fraction(1)
    for i in range(k):x*=a+i
    return x


def test_predictive_is_exactly_normalized_for_every_test_risk():
    for kt,rt in [(0,0),(0,8),(3,8),(8,8)]:
        a=Fraction(2*kt+1,2);b=Fraction(2*(rt-kt)+1,2)
        for r in [0,1,6]:
            total=Fraction(0)
            for k in range(r+1):
                q=rising(a,k)*rising(b,r-k)/rising(a+b,r)
                total+=comb(r,k)*q
                x=predictive_log_interval(kt,rt,k,r);exact=(arb(q.numerator)/q.denominator).log()
                assert arb(x['lower'])<=exact and arb(x['upper'])>=exact
            assert total==1


def test_binomial_profile_upper_covers_all_rates_in_a_continuous_interval():
    for k,r in [(0,0),(0,20),(3,20),(20,20)]:
        upper=binomial_interval_upper(k,r,.001,.3)['upper']
        for rate in [.001,.013,.1,.3]:
            x=arb(rate);ll=arb(0)
            if k:ll+=k*(-(-x).expm1()).log()
            ll-=(r-k)*x
            assert arb(upper)>=ll
    assert calibration_upper(20,20,0.,.25,{'informative':False})['upper']>=0


def test_nuisance_interval_algebra_has_compatible_hazard_enclosures():
    # Data-derived intervals are checked against their own exact rational
    # endpoint budget; actual physical coverage is a theorem, not this sample.
    t=np.arange(256);rates=.01*np.exp(-4*np.log(2)*((t-32.1)/5.1)**2)+.005/256
    h=np.random.default_rng(7518).multinomial(50000,first_detection_probabilities(rates))[:-1]
    c=nuisance_confidence(h,50000)
    assert c['informative'] and Fraction(c['one_sided_budget'])*4==Fraction(c['delta'])
    assert c['signal_main_lower']>=0 and c['signal_main_upper']>=c['signal_main_lower']
    # Any chosen nuisance point in the confidence box is covered, regardless
    # of which eta/geometry point produced it.
    eta=.073;s=(c['signal_main_lower']+c['signal_main_upper'])/2
    b=(c['background_lower']+c['background_upper'])/2
    f=(c['forward_leakage_lower']+c['forward_leakage_upper'])/2
    rate=arb(s)*(arb(eta)+arb(f))+11*arb(b)
    r=40000;k=20
    ll=k*(-(-rate).expm1()).log()-(r-k)*rate
    assert arb(calibration_upper(k,r,.0625,.125,c)['upper'])>=ll


def test_crossfold_error_budget_and_good_event_truncation_argument():
    # Each E_j*1(G_j) has mean<=1. Thus on the intersection, the average
    # remains dominated by their half-weighted sum; independence of G_j is
    # unnecessary. Two training confidence failures must BOTH be budgeted.
    delta_fold=Fraction(1,2000);alpha_query=Fraction(9,1000)
    assert alpha_query+2*delta_fold==Fraction(1,100)
    for e1,e2,g1,g2 in [(100.,0.,True,False),(20.,200.,True,True),(1.,50.,False,True)]:
        left=(e1+e2)/2*bool(g1 and g2);right=(e1*bool(g1)+e2*bool(g2))/2
        assert left<=right
