import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from flint import arb,ctx
from continuous_rate_profile import numeric_value_gradient,profile_upper,certified_value_gradient


def inputs():
    s=np.array([0,3,11,7,1,0]);e=np.array([40.,37.25,39.5,38.,40.,39.]);f=np.array([.001,.1,.5,.3,.08,.003])
    return s,e,f*.9,f*1.1


def test_profile_gradient_matches_independent_finite_difference():
    s,e,lo,hi=inputs();x=np.array([.6,.7]);v,g=numeric_value_gradient(s,e,lo,hi,*x)
    numerical=approx_derivative(lambda z:numeric_value_gradient(s,e,lo,hi,*z)[0],x).ravel()
    np.testing.assert_allclose(g,numerical,rtol=2e-8,atol=2e-8)


def test_certified_tangent_covers_full_nuisance_orthant_including_very_large_rates():
    ctx.prec=128;s,e,lo,hi=inputs();b=profile_upper(s,e,lo,hi);assert b['tangent_certified']
    for a in [0.,.2,.5,2.,100.,1e8]:
        for bg in [1e-9,.3,1.,50.,1e8]:
            for frac in [0.,.37,1.]:
                shape=lo+frac*(hi-lo);value=arb(0)
                for n,t,f in zip(s,e,shape):
                    lam=arb(a)*arb(float(f))+arb(bg)/len(s)
                    if n:value+=int(n)*lam.log()
                    value-=arb(float(t))*lam
                assert arb(b['upper'])>=value


def test_zero_lower_response_envelope_still_gets_a_finite_global_bound():
    ctx.prec=128;s,e,lo,hi=inputs();b=profile_upper(s,e,np.zeros_like(lo),hi)
    assert np.isfinite(b['upper']) and b['tangent_certified']


def test_invalid_exposure_count_pair_is_not_silently_given_a_finite_certificate():
    with pytest.raises(ValueError,match='Positive count'):profile_upper([1],[0],[.1],[.2])
