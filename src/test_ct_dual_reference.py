import numpy as np
from scipy.integrate import quad
from ct_dual_reference import prepare_rows,prepare_record,evaluate_rows,full_factor
from homogeneous_dual_deadtime import log_gap_density,gap_survival
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from certified_safe_components import exact_safe_statistics
from ct_dual_reference import cached_profile,conditional_from_profile


def test_homogeneous_transition_and_terminal_match_independent_renewal_formula():
    for td,te,rate in [(31.7,80.,.05),(10.,25.,.2),(83.7,100.,.005),(100.,20.,.05)]:
        gaps=np.r_[max(td,te)+np.array([.01,1.,5.,10.]),td+te+np.array([1.,10.,50.])];rates=np.full(8,rate);design=np.ones((8,1));g=prepare_rows(np.full(len(gaps),.37),gaps,8,td,te);v,_=evaluate_rows(rates,design,g);np.testing.assert_allclose(v,log_gap_density(gaps,rate,td,te),atol=2e-10)
        g=prepare_rows(np.full(len(gaps),.37),gaps,8,td,te,np.ones(len(gaps),bool));v,_=evaluate_rows(rates,design,g);np.testing.assert_allclose(v,np.log(gap_survival(gaps,rate,td,te)),atol=3e-10)


def test_detector_dominant_full_value_and_gradient_match_exact_safe_factor():
    theta=np.array([.8,15.3,.3,.065,27.3]);rate,j=rates_and_design(theta,64,1.5);obs,_=simulate_two_dead_times(rate,100,83.7,40.,307019);geometry=prepare_record(obs,6400.,64,83.7,40.);value,gradient=full_factor(rate,j,geometry);stats=exact_safe_statistics(obs,100,64,83.7);s=stats['counts'];e=np.array([float(v.mid()) for v in stats['exposure']]);np.testing.assert_allclose(value,s@np.log(rate)-e@rate,atol=2e-8);np.testing.assert_allclose(gradient,j.T@(s/rate-e),atol=2e-8)


def test_two_hidden_periodic_gradient_and_quadrature_convergence():
    theta=np.array([2.4,15.3,.8,.15,25.3]);rate,j=rates_and_design(theta,64,1.5);obs,_=simulate_two_dead_times(rate,25,31.7,80.,729501);g=prepare_record(obs,1600.,64,31.7,80.);a,ga=full_factor(rate,j,g,16);b,gb=full_factor(rate,j,g,32);np.testing.assert_allclose(a,b,atol=2e-9);np.testing.assert_allclose(ga,gb,atol=2e-8);finite=[]
    for k in range(5):
        d=np.zeros(5);d[k]=1e-6;r,z=rates_and_design(theta+d,64,1.5);x=full_factor(r,z,g,32)[0];r,z=rates_and_design(theta-d,64,1.5);y=full_factor(r,z,g,32)[0];finite.append((x-y)/2e-6)
    np.testing.assert_allclose(ga,finite,atol=2e-5,rtol=2e-5)


def test_periodic_single_hidden_density_matches_direct_published_integral():
    rate=np.array([.08,.16,.04,.11,.09,.05,.13,.07]);j=np.eye(8);td=5.3;te=8.;phi=.37;gaps=np.array([8.1,10.9,12.,14.]);g=prepare_rows(np.full(4,phi),gaps,8,td,te);actual,_=evaluate_rows(rate,j,g,32)
    def r(t):return rate[int(np.floor((phi+t)%8))]
    def integral(a,b):
        points=[v-phi for v in range(int(np.floor(a+phi)),int(np.ceil(b+phi))+1) if a<v-phi<b];return quad(r,a,b,points=points,epsabs=1e-12)[0]
    expected=[]
    for s in gaps:
        upper=min(te,s-td);knots=sorted(set([td,upper]+[v-phi+shift for shift in [0.,-td] for v in range(-20,40) if td<v-phi+shift<upper]));h=0.
        if upper>td:h=sum(quad(lambda t:r(t)*np.exp(integral(t,t+td)),a,b,epsabs=1e-11)[0] for a,b in zip(knots[:-1],knots[1:]))
        expected.append(np.log(r(s))-integral(td,s)+np.log1p(h))
    np.testing.assert_allclose(actual,expected,atol=2e-10)


def test_empty_record_factor_is_initial_no_photon_probability():
    rate=np.array([.1,.3]);j=np.eye(2);g=prepare_record([],20.,2,3.,7.);value,gradient=full_factor(rate,j,g);np.testing.assert_allclose(value,-4.);np.testing.assert_allclose(gradient,[-10.,-10.])


def test_positive_state_survival_matches_independent_outer_density_quadrature():
    import ct_dual_outer_reference as outer
    theta=np.array([2.4,15.3,.8,.15,25.3]);rate,j=rates_and_design(theta,64,1.5);phases=np.array([.37,15.31,42.1]);gaps=np.array([80.,92.,107.]);g=prepare_rows(phases,gaps,64,31.7,80.,np.ones(3,bool));a,ga=evaluate_rows(rate,j,g,32);b,gb=outer.evaluate_rows(rate,j,g,32);np.testing.assert_allclose(a,b,atol=2e-9);np.testing.assert_allclose(ga,gb,atol=2e-8);assert abs(a[0])<2e-10;np.testing.assert_allclose(ga[0],np.zeros(5),atol=2e-9)


def test_cached_conditional_profile_preserves_density_survival_and_gradients():
    theta=np.array([2.4,15.3,.8,.15,25.3]);rate,j=rates_and_design(theta,64,1.5);gaps=np.array([79.,80.,90.,101.,120.,170.]);profile=cached_profile(rate,j,.37,31.7,80.,32)
    for survival in [False,True]:
        a,ga=conditional_from_profile(profile,gaps,survival);geometry=prepare_rows(np.full(len(gaps),.37),gaps,64,31.7,80.,np.full(len(gaps),survival));b,gb=evaluate_rows(rate,j,geometry,32);np.testing.assert_allclose(a,b,atol=2e-10);np.testing.assert_allclose(ga,gb,atol=2e-9,equal_nan=True)
