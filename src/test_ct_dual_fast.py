import numpy as np
import math
from scipy.integrate import quad
import ct_dual_fast as fast
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times


def test_mixed_exponential_integrals_and_derivatives_match_independent_quadrature():
    for b,c,length in [(0.,0.,1.),(.7,0.,.4),(-.7,1e-10,.8),(.3,-.03,1.),(2.,-1.,.7),(-8.,.05,1.),(5.,3.,.4)]:
        value,db,dc=fast._mixed_moments(b,c,length)
        def e(c,z):return np.expm1(c*z)/c if c else z
        def e1(c,z):return (np.exp(c*z)*(c*z-1)+1)/(c*c) if abs(c*z)>.01 else sum(c**k*z**(k+2)/(math.factorial(k)*(k+2)) for k in range(10))
        expected=quad(lambda z:np.exp(b*z)*e(c,z),0,length,epsabs=1e-12)[0];eb=quad(lambda z:z*np.exp(b*z)*e(c,z),0,length,epsabs=1e-12)[0];ec=quad(lambda z:np.exp(b*z)*e1(c,z),0,length,epsabs=1e-12)[0]
        np.testing.assert_allclose([value,db,dc],[expected,eb,ec],atol=3e-11,rtol=2e-10)


def test_analytic_segments_match_quadrature_full_factors_and_five_parameter_gradients():
    for theta,td,te,seed in [(np.array([2.4,15.3,.8,.15,25.3]),31.7,80.,720381),(np.array([.25,15.3,.1,.035,35.3]),83.7,100.,720382)]:
        rate,j=rates_and_design(theta,64,1.5);obs,_=simulate_two_dead_times(rate,40,td,te,seed);g=fast.prepare_record(obs,2560.,64,td,te);a,ga=fast.full_factor(rate,j,g);b,gb=reference.full_factor(rate,j,g,32);np.testing.assert_allclose(a,b,atol=2e-8);np.testing.assert_allclose(ga,gb,atol=2e-7,rtol=1e-8)


def test_all_phase_rate_sensitivities_and_cached_survival_match_reference():
    rate,_=rates_and_design(np.array([2.4,15.3,.8,.15,25.3]),64,1.5);j=np.eye(64);gap=np.array([80.,87.3,95.2,110.,120.]);f=fast.cached_profile(rate,j,.37,31.7,80.);r=reference.cached_profile(rate,j,.37,31.7,80.,32)
    for survival in [False,True]:
        a,ga=fast.conditional_from_profile(f,gap,survival);b,gb=reference.conditional_from_profile(r,gap,survival);np.testing.assert_allclose(a,b,atol=2e-10);np.testing.assert_allclose(ga,gb,atol=2e-8,rtol=1e-8)


def test_cached_detector_dominant_factors_match_raw_reference():
    rates=np.array([.03,.08,.16,.02]);j=np.eye(4);gaps=np.array([79.,83.7,91.,120.]);profile=fast.cached_profile(rates,j,.37,83.7,40.)
    for survival in [False,True]:
        a,ga=fast.conditional_from_profile(profile,gaps,survival);g=reference.prepare_rows(np.full(4,.37),gaps,4,83.7,40.,np.full(4,survival));b,gb=reference.evaluate_rows(rates,j,g);np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(ga,gb,atol=1e-11,equal_nan=True)


def test_compiled_geometry_matches_independent_python_geometry():
    phases=np.random.default_rng(6135921).uniform(0,64,12);gaps=np.linspace(80,200,12);flags=np.zeros(12,bool);flags[-1]=True;gaps[-1]=92.
    for td,te in [(31.7,80.),(83.7,100.),(83.7,40.)]:
        a=fast.prepare_rows(phases,gaps,64,td,te,flags);b=reference.prepare_rows(phases,gaps,64,td,te,flags)
        for key in a:
            if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            else:assert a[key]==b[key]
