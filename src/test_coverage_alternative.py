from fractions import Fraction
import numpy as np
from flint import arb
from coverage_alternative import coverage_components,numeric_scores,selected_indices,sparse_numerator_lower,weighted_log_lower
from envelope_evidence import outward_float,certified_log_likelihood


def proposal(secondary=8.):
    return dict(primary=20.2,secondary=43.4,primary_incident=50.,secondary_incident=secondary,
        background_incident=10.,fitted_fwhm=5.,effective_donors=30.,raw_center_excluded=True)


def test_donor_only_family_has_exact_unit_mass_and_covers_isolated_locations():
    for c,expected in [(8.,Fraction(9,10)),(.001,Fraction(1,10))]:
        m=coverage_components(proposal(c),64)
        assert sum(m['weights'],Fraction(0))==1 and m['spatial_mass']==expected
        assert np.all(m['rates']>0) and np.isfinite(m['rates']).all()
        assert 43 in m['global_delays'] and np.all(abs(m['global_delays']-20.2)>=8.)


def test_query_adaptive_subset_is_subdensity_only_when_original_weights_retained():
    # Exact finite outcome enumeration, independent of the photon simulator.
    probabilities=[[Fraction(9,10),Fraction(1,20),Fraction(1,20)],
                   [Fraction(1,20),Fraction(9,10),Fraction(1,20)]]
    weights=[Fraction(1,2),Fraction(1,2)];total=arb(0);bad=Fraction(0)
    for y in range(3):
        chosen=max(range(2),key=lambda i:weights[i]*probabilities[i][y])
        lower={i:outward_float((arb(p[y].numerator)/p[y].denominator).log(),False) for i,p in enumerate(probabilities)}
        value=weighted_log_lower(lower,weights,[chosen]);total+=arb(value).exp()
        bad+=probabilities[chosen][y]  # Renormalization would remove prior mass.
    assert total<=1 and bad==Fraction(37,20)>1


def test_sparse_certified_numerator_is_below_full_proper_mixture():
    m=coverage_components(proposal(.001),64);h=np.zeros(64,int);h[20]=12;h[42]=6
    _,weighted,_=numeric_scores(h,m,1000);indices=selected_indices(weighted,4)
    sparse=sparse_numerator_lower(h,m,indices,1000)
    full=[]
    for rate,w in zip(m['rates'],m['weights']):
        ell=certified_log_likelihood(h,rate,1000,'first')
        full.append(arb(ell)+(arb(w.numerator)/w.denominator).log())
    shift=max(float(x) for x in full);sum_lower=arb(shift)+sum(((x-arb(shift)).exp() for x in full),arb(0)).log()
    assert arb(sparse['lower'])<=sum_lower
    assert sparse['evaluated_components']<sparse['full_components'] and Fraction(sparse['retained_prior_mass'])<1
