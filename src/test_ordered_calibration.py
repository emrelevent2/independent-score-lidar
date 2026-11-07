from fractions import Fraction as F
from math import comb,factorial
import numpy as np
from ordered_calibration import query_predictor


def binom(n,k,p):return comb(n,k)*p**k*(1-p)**(n-k)


def predictive(n,k,nt,kt):
    # Beta(1,1) predictive, including the binomial coefficient. Exact rationals.
    a,b=kt+1,nt-kt+1
    return F(comb(n,k)*factorial(a+k-1)*factorial(b+n-k-1)*factorial(a+b-1),
        factorial(a+b+n-1)*factorial(a-1)*factorial(b-1))


def test_full_calibration_query_reuse_has_exact_unit_mean():
    n=3
    for p in [F(1,5),F(1,2),F(4,5)]:
        total=F(0);truncated=F(0);bad=F(0)
        for a in range(n+1):
            for b in range(n+1):
                pq=F(a+b+1,2*n+2)  # depends on BOTH calibration halves
                mass=binom(n,a,p)*binom(n,b,p)
                # Synthetic confidence event, data-dependent and shared.
                good=(a+b)>0
                if not good:bad+=mass
                for x in range(n+1):
                    px=binom(n,x,p);qx=binom(n,x,pq)
                    ratio=(predictive(n,b,n,a)/binom(n,b,p)+predictive(n,a,n,b)/binom(n,a,p))/2*qx/px
                    total+=mass*px*ratio
                    if good:truncated+=mass*px*ratio
        assert total==1 and truncated<=1 and bad>=0


def test_query_fitted_numerator_is_not_proper_guard():
    # A predictor fitted on its OWN scored observation is not normalized.
    assert sum(binom(3,k,F(k,3)) for k in range(4))>1


def test_full_calibration_predictor_excludes_raw_query():
    rng=np.random.default_rng(96511)
    h=rng.poisson(.08,(9,9,256));cal=rng.poisson(2.,256)
    first=query_predictor(h,cal,100000);changed=h.copy();changed[4,4]=0
    second=query_predictor(changed,cal,100000)
    assert first['proposal']==second['proposal'] and first['fit']==second['fit']
    assert not first['fit']['scored_calibration_excluded'] and first['fit']['query_excluded']
