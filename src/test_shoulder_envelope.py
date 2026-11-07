from fractions import Fraction
import numpy as np
import pytest
from flint import arb
from shoulder_envelope import (response_basis,relaxed_value_gradient,bound_box,
                               exact_rates,shoulder_null_bound)
from envelope_evidence import _observations,certified_log_likelihood,exact_gaussian
from pilot import first_detection_probabilities


def test_exact_cone_and_two_object_identifiability():
    M=Fraction(1,4)
    for a in [Fraction(0),Fraction(7),Fraction(16)]:
        for eta in [Fraction(0),Fraction(1,20),M]:
            u=a*(1-eta/M);v=a*eta/M
            assert u>=0 and v>=0 and u+v==a and v*M==a*eta
    # Two ideal Gaussian objects at d and d+16 can be exactly one IRF with
    # an unknown lobe, for EVERY photon budget. This is a physical ambiguity.
    main,second=Fraction(16),Fraction(2);eta=second/main
    assert main*eta==second and eta<=M


def test_shape_envelopes_cover_continuous_geometry_and_cap():
    lo,hi=response_basis(32,3.2,5.1,2.5,3.3,.25,4)
    for d in [3.2,4.07,5.1]:
        for w in [2.5,2.94,3.3]:
            main=exact_gaussian(32,d,w,False);side=exact_gaussian(36,arb(d)+4,w,False)
            for j,(g,s) in enumerate(zip(main,side)):
                assert g>=arb(float(lo[j,0])) and g<=arb(float(hi[j,0]))
                x=g+arb(.25)*s
                assert x>=arb(float(lo[j,1])) and x<=arb(float(hi[j,1]))


@pytest.mark.parametrize('family',['first','poisson'])
def test_profile_gradient_concavity_and_certified_box(family):
    rng=np.random.default_rng(7311)
    rates=np.array([float(x) for x in exact_rates(32,7.7,3.,[1.,2.,5.],.25,4,1)])
    h=(rng.multinomial(1000,first_detection_probabilities(rates/1000))[:-1]
       if family=='first' else rng.poisson(rates))
    counts,failures,scale,saturated=_observations(h,1000,family)
    lo,hi=response_basis(32,7.,8.2,2.5,3.3,.25,4)
    x=np.array([1.,2.,5.]);y=np.array([4.,1.,9.]);z=.37*x+.63*y
    value,grad=relaxed_value_gradient(x,counts,failures,lo,hi,scale,family)
    for j in range(3):
        e=np.zeros(3);e[j]=1e-5
        vp=relaxed_value_gradient(x+e,counts,failures,lo,hi,scale,family)[0]
        vm=relaxed_value_gradient(x-e,counts,failures,lo,hi,scale,family)[0]
        assert grad[j]==pytest.approx((vp-vm)/2e-5,abs=1e-6)
    vz=relaxed_value_gradient(z,counts,failures,lo,hi,scale,family)[0]
    vy=relaxed_value_gradient(y,counts,failures,lo,hi,scale,family)[0]
    assert vz>=.37*value+.63*vy-1e-10
    box=bound_box(counts,failures,scale,family,7.,8.2,2.5,3.3,saturated,.25,4)
    assert box.certified and box.lower<=box.upper
    for d in [7.,7.49,8.2]:
        for w in [2.5,2.91,3.3]:
            for theta in [x,y,[0.,0.,5.],[20.,0.,.1],[0.,25.,.1]]:
                ll=certified_log_likelihood(h,exact_rates(32,d,w,theta,.25,4,scale),1000,family,upper=True)
                assert ll<=box.upper


def test_early_stopped_cover_includes_true_null_even_at_delay_edge():
    rng=np.random.default_rng(7312)
    for d,cap in [(3.7,.25),(30.8,0.)]:
        rates=exact_rates(32,d,3.,[1.,2.,3.],cap,4,1000)
        h=rng.multinomial(1000,first_detection_probabilities([float(v) for v in rates]))[:-1]
        bound=shoulder_null_bound(h,1000,(2.5,3.3),cap=cap,shift=4,initial_cells=4,max_boxes=8)
        ll=certified_log_likelihood(h,rates,1000,'first',upper=True)
        assert bound['upper']>=ll and bound['upper']>=bound['feasible_lower']
