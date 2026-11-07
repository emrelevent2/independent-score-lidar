from fractions import Fraction
from math import factorial,comb
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import poisson
from thinned_predictive import split_query,predict,components,score,POLICIES


def donor():
    return dict(primary=15.,secondary=42.,primary_count=20.,secondary_count=4.,background_count=8.,width=5.,
        support_proxy=4.,effective_donors=12.,raw_center_excluded=True)


def test_poisson_split_joint_mass_factorizes_after_common_exponential_removed():
    # Exact algebra,not a Monte-Carlo significance test. The canceled
    # exp(-lambda) factor equals exp(-p*lambda)*exp(-(1-p)*lambda).
    p=Fraction(1,4);rate=Fraction(3,2)
    for a in range(8):
        for b in range(8):
            joint=rate**(a+b)/factorial(a+b)*comb(a+b,a)*p**a*(1-p)**b
            product=(p*rate)**a/factorial(a)*((1-p)*rate)**b/factorial(b)
            assert joint==product


def test_split_preserves_parent_and_is_reproducible_without_cycle_count():
    h=np.array([0,5,1007,0,8]);a,b=split_query(h,.25,17);c,d=split_query(h,.25,17)
    np.testing.assert_array_equal(a+b,h);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    assert np.all(a>=0) and np.all(b>=0)


def test_predictor_truthfully_distinguishes_training_from_scored_counts():
    q=donor();fit=dict(parameters=[5.,2.,18.2,7.])
    for policy in POLICIES:
        prediction=predict(q,fit,.25,policy)
        assert prediction['scored_counts_excluded']
        assert prediction['query_training_used']==(policy!='donor')
        assert prediction['raw_center_excluded']==(policy=='donor')
        fallback=predict(q,None,.25,policy);assert not fallback['query_training_used']
    assert q==donor()  # Original frozen donor record is never mutated.


def test_predictive_density_matches_normalized_poisson_mixture():
    q=predict(donor(),dict(parameters=[4.,2.,16.,6.]),.25,'shrunken_nuisance')
    h=np.zeros(64,int);h[16]=8;h[41]=3;amps=np.array([1,2,4,8,16,32])*.75
    m=components(q,64,amps);assert sum(m['weights'],Fraction(0))==1
    weighted=poisson.logpmf(h[None,:],m['means']).sum(1)+np.log([float(w) for w in m['weights']])
    result=score(h,q,-50.,amps)
    np.testing.assert_allclose(result['score'],logsumexp(weighted)+gammaln(h+1).sum()+50,atol=1e-12)
    assert result['calibration_required'] and not result['analytic_guarantee']
