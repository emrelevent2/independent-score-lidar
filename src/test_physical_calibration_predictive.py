from math import comb
from flint import arb
from physical_calibration_predictive import fixed_hazard_log,half_mixture


def test_physical_and_fixed_mixture_predictions_normalize_for_every_risk():
    for r in [0,1,7,30]:
        total=arb(0);mixed=arb(0)
        for k in range(r+1):
            a=fixed_hazard_log(k,r,arb(.07));b=fixed_hazard_log(k,r,arb(.13));m=half_mixture(a,b)
            # Lower/upper directed endpoints enclose every probability.
            total+=comb(r,k)*arb(a['lower']).union(arb(a['upper'])).exp()
            mixed+=comb(r,k)*arb(m['lower']).union(arb(m['upper'])).exp()
        assert total.contains(1) and mixed.contains(1)
