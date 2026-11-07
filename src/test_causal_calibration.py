from fractions import Fraction as F
from math import comb
import numpy as np
from flint import arb
from causal_calibration import prefix_features,score_lobe


def binom(n,k,p):return comb(n,k)*p**k*(1-p)**(n-k)


def test_all_lobe_scoring_with_past_training_and_full_cal_query_has_unit_mean():
    # First-event categories:prefix,lobe,future,none. Future is correlated
    # with the scored lobe through the remaining risk;queryQ may use it.
    N=3;total=F(0)
    for pre in range(N+1):
        R=N-pre;qcal=F(pre+1,N+2)
        for k in range(R+1):
            for future in range(R-k+1):
                mass=binom(N,pre,F(1,5))*binom(R,k,F(1,4))*binom(R-k,future,F(1,3))
                qx=F(pre+k+future+1,N+2)
                for x in range(3):
                    px=binom(2,x,F(2,5));ratio=binom(R,k,qcal)/binom(R,k,F(1,4))*binom(2,x,qx)/px
                    total+=mass*px*ratio
    assert total==1


def test_prefix_mixture_is_proper_for_every_realized_risk():
    prefix=np.zeros(43,int);prefix[30:35]=[1,2,3,2,1];N=25;r=N-int(prefix.sum());probability=arb(0)
    for k in range(r+1):
        q=score_lobe(prefix,N,.12,k)
        probability+=comb(r,k)*arb(q['lower']).union(arb(q['upper'])).exp()
    assert probability.contains(1)


def test_calibration_predictor_api_excludes_its_scored_and_future_bins():
    import pytest
    h=np.zeros(256,int)
    with pytest.raises(ValueError):prefix_features(h,1000,.1)
    a=prefix_features(h[:43],1000,.1);h[43:54]=4;h[128:]=3
    b=prefix_features(h[:43],1000,.1);assert a==b
