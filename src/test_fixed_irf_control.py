from fractions import Fraction
import numpy as np
from flint import arb
from fixed_irf_control import fixed_null_bound,fixed_proposal,fixed_components
from envelope_evidence import exact_gaussian,certified_log_likelihood
from pilot import first_detection_probabilities


def test_fixed_shape_complete_cover_contains_feasible_likelihood():
    g=exact_gaussian(64,31.73,5.1,False);side=exact_gaussian(80,arb(31.73)+16,5.1,False)[:64]
    for eta in [0.,.125]:
        rates=[(arb(10)*(x+arb(eta)*y)+arb(5)/64)/1000 for x,y in zip(g,side)]
        h=np.random.default_rng(7414).multinomial(1000,first_detection_probabilities([float(x) for x in rates]))[:-1]
        b=fixed_null_bound(h,eta,initial_cells=4,max_boxes=16)
        assert b['upper']>=certified_log_likelihood(h,rates,1000,'first',upper=True)


def test_point_irf_mixture_mass_and_center_exclusion():
    rng=np.random.default_rng(7415);rate=np.full(256,.0001);rate[80]=.05;rate[96]=.008
    h=rng.multinomial(1000,first_detection_probabilities(rate),size=(9,9))[...,:-1]
    q=fixed_proposal(h,.125);changed=h.copy();changed[4,4]=0
    assert fixed_proposal(changed,.125)==q
    m=fixed_components(q,.125)
    assert sum(m['weights'],Fraction(0))==1 and m['raw_center_excluded']
    assert np.all(m['rates']>0) and len(m['rates'])==len(m['locations'])
