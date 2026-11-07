from fractions import Fraction
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import poisson
from real_coverage_mixture import components,report


def proposal(support=4.):
    return dict(primary=13.2,secondary=40.1,primary_count=15.,secondary_count=3.,background_count=5.,width=5.,support_proxy=support,raw_center_excluded=True)


def test_real_mixture_has_exact_unit_mass_and_fixed_incident_counts():
    for support,rho in [(4.,Fraction(9,10)),(1.,Fraction(1,10))]:
        m=components(proposal(support),64)
        assert sum(m['weights'],Fraction(0))==1 and m['spatial_mass']==str(rho)
        assert np.all(m['means']>0)
        np.testing.assert_allclose(m['means'][:4].sum(1),20+3*np.array([.5,1,2,4]),atol=1e-8)
        assert np.all(abs(m['global_delays']-13.2)>=8)


def test_real_numeric_mixture_matches_normalized_poisson_distributions():
    q=proposal();h=np.zeros(64,int);h[13]=8;h[39]=4
    m=components(q,64);terms=poisson.logpmf(h[None,:],m['means']).sum(1)+np.log([float(w) for w in m['weights']])
    result=report(h,q,-50.)
    np.testing.assert_allclose(result['log_mixture'],logsumexp(terms)+gammaln(h+1).sum(),atol=1e-12)
    assert result['calibration_required'] and not result['analytic_guarantee']
