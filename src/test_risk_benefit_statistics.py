"""Independent small probability-space checks, not efficacy fixtures."""
import itertools
import math
import numpy as np
import pytest
from risk_benefit_statistics import rate_bounds,paired_bounds,bernoulli_kl


def test_boundary_counts_have_nonzero_uncertainty_and_correct_roots():
    for n in [1,7,384,1920]:
        for delta in [.001,.025,.1]:
            a=rate_bounds(0,n,delta);b=rate_bounds(n,n,delta)
            assert a[0]==0 and a[1]>0 and b[0]<1 and b[1]==1
            np.testing.assert_allclose(a[1],1-delta**(1/n),atol=2e-12)
            np.testing.assert_allclose(b[0],delta**(1/n),atol=2e-12)
    r=paired_bounds(0,0,1920)
    assert r['lower']<0<r['upper']


def test_exact_poisson_binomial_enumeration_covers_nonidentical_query_rates():
    families=[[0,.01,.2,.6,.99,1,.4,.8],[.001]*8,[.999]*8,[0,1,0,1,0,1,0,1]]
    for probs in families:
        pmf=np.array([1.])
        for p in probs:pmf=np.convolve(pmf,[1-p,p])
        mu=np.mean(probs);n=len(probs)
        for delta in [.001,.025,.1]:
            bounds=np.array([rate_bounds(k,n,delta) for k in range(n+1)])
            assert pmf[bounds[:,0]>mu+1e-14].sum()<=delta+1e-12
            assert pmf[bounds[:,1]<mu-1e-14].sum()<=delta+1e-12


def test_exact_paired_three_outcome_space_covers_heterogeneous_discordances():
    probs=[(.1,.2),(.6,.1),(.01,.4),(.05,.01),(.3,.1)]
    mu=np.mean([g-l for g,l in probs]);failure=0.
    for states in itertools.product(range(3),repeat=len(probs)):
        mass=math.prod(([1-g-l,g,l][state]) for state,(g,l) in zip(states,probs))
        bound=paired_bounds(states.count(1),states.count(2),len(probs),.05,1)
        if bound['lower']>mu+1e-14 or bound['upper']<mu-1e-14:failure+=mass
    assert failure<=.05+1e-12


def test_interior_inversion_and_monotonicity():
    for n in [12,384,1920]:
        previous=[-1.,-1.]
        for k in sorted(set([0,1,2,n//2,n-1,n])):
            a=rate_bounds(k,n,.01);b=rate_bounds(k,n,.001)
            assert 0<=a[0]<=k/n<=a[1]<=1
            assert a[0]>=previous[0] and a[1]>=previous[1]
            assert b[0]<=a[0] and b[1]>=a[1]
            if 0<k<n:
                for value in a:
                    np.testing.assert_allclose(n*bernoulli_kl(k/n,value),-math.log(.01),atol=1e-5)
            previous=a


def test_invalid_counts_and_discordance_overlap_rejected():
    for k,n,d in [(1.1,3,.1),(1,0,.1),(-1,3,.1),(4,3,.1),(0,3,0),(0,3,1),(0,3,np.nan)]:
        with pytest.raises(ValueError):rate_bounds(k,n,d)
    with pytest.raises(ValueError):paired_bounds(3,3,4)
    with pytest.raises(ValueError):paired_bounds(1,0,4,contrasts=0)
