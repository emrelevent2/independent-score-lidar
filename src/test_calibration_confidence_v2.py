from fractions import Fraction
from math import comb
from scipy.stats import beta
from calibration_confidence_v2 import binomial_endpoint,binomial_cdf
from flint import arb


def test_exact_small_sample_tail_coverage():
    n=12;delta=Fraction(1,20)
    for p in [Fraction(i,20) for i in range(21)]:
        bu=Fraction(0);bl=Fraction(0)
        for k in range(n+1):
            mass=comb(n,k)*p**k*(1-p)**(n-k)
            if p>Fraction(binomial_endpoint(k,n,delta,True)):bu+=mass
            if p<Fraction(binomial_endpoint(k,n,delta,False)):bl+=mass
        assert bu<=delta and bl<=delta


def test_large_cycle_endpoints_remain_informative_and_certified():
    delta=Fraction(1,3000);budget=arb(1)/3000
    for n in [10000,100000]:
        for k in [0,1,5,10,50,100,500]:
            u=binomial_endpoint(k,n,delta,True)
            assert u<1 and binomial_cdf(k,n,u)<=budget
            proposal=beta.ppf(1-float(delta),k+1,n-k)
            assert abs(u-proposal)<1e-10
            if k:
                l=binomial_endpoint(k,n,delta,False)
                assert l>0 and 1-binomial_cdf(k-1,n,l)<=budget
