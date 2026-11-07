from fractions import Fraction
from math import comb
import numpy as np
import pytest
from flint import arb
from calibration_confidence import (binomial_endpoint,binomial_cdf,window_counts,
                                     leakage_ratio_upper,lobe_confidence)
from envelope_evidence import exact_gaussian


def test_exact_binomial_undercoverage_on_rational_probability_grid():
    # Exact finite-outcome calculation, not a Monte Carlo coverage assertion.
    delta=Fraction(1,20);n=12
    for p in [Fraction(i,20) for i in range(21)]:
        bad_upper=Fraction(0);bad_lower=Fraction(0)
        for k in range(n+1):
            mass=comb(n,k)*p**k*(1-p)**(n-k)
            u=Fraction(binomial_endpoint(k,n,delta,True))
            l=Fraction(binomial_endpoint(k,n,delta,False))
            if p>u:bad_upper+=mass
            if p<l:bad_lower+=mass
        assert bad_upper<=delta and bad_lower<=delta
    p=.125;k=2
    exact=sum(Fraction(comb(n,j))*Fraction(p)**j*(1-Fraction(p))**(n-j) for j in range(k+1))
    value=binomial_cdf(k,n,p)
    assert value.contains(arb(exact.numerator)/exact.denominator)


def test_first_event_window_risk_and_degenerate_confidence():
    h=np.zeros(256,dtype=int);h[5]=17;h[30]=3;h[47]=2
    assert window_counts(h,100,27,37)==(3,83)
    assert window_counts(h,100,43,53)==(2,80)
    assert lobe_confidence(np.zeros(256,dtype=int),0)['eta_upper']==.25
    assert lobe_confidence(np.zeros(256,dtype=int),1000)['eta_upper']==.25
    with pytest.raises(ValueError):lobe_confidence(h,10)


def test_shifted_window_cancellation_and_conservative_leakage_ratio():
    epsilon=leakage_ratio_upper();M=arb(.25)
    for d in [31.5,31.73,32.,32.5]:
        for w in [4.5,4.91,5.5]:
            g=exact_gaussian(272,d,w,False)
            s=exact_gaussian(272,arb(d)+16,w,False)
            a=sum(g[27:38],arb(0));lobe_a=sum(s[43:54],arb(0))
            assert (a-lobe_a).contains(0)
            e=sum(s[27:38],arb(0));assert e/a<=arb(epsilon)
            # With exact hazard information, the conservative algebra must
            # contain eta despite nonzero Gaussian leakage and dark-window tail.
            for eta in [0.,.03125,.125,.25]:
                rate=[arb(.01)*(x+arb(eta)*y)+arb(.005)/256 for x,y in zip(g,s)]
                H0=sum(rate[27:38],arb(0));H1=sum(rate[43:54],arb(0))
                Hd=sum(rate[128:256],arb(0));den=H0-Hd*arb(11)/128
                assert den>0
                upper=H1*(1+M*arb(epsilon))/den
                assert upper>=arb(eta)
