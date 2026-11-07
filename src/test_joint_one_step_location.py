import numpy as np
from scipy.optimize import check_grad
from deadtime_scan import pulse,fit_null
from joint_one_step_location import estimate,likelihood_and_fisher,initial_amplitudes


def fixture():
    k=32;e=np.linspace(500,2000,k);rates=.8*pulse(7.3,k,1.5)[0]+.06*pulse(17.1,k,1.5)[0]+.3/k;s=np.rint(e*rates);return s,e


def test_five_parameter_score_and_information_match_independent_direct_design():
    s,e=fixture();theta=np.array([.8,7.3,.3,.06,17.1]);ll,u,info=likelihood_and_fisher(s,e,theta,1.5);f,df=pulse(theta[1],32,1.5);g,dg=pulse(theta[4],32,1.5);rate=theta[0]*f+theta[2]/32+theta[3]*g;j=np.column_stack([f,theta[0]*df,np.ones(32)/32,g,theta[3]*dg]);np.testing.assert_allclose(u,j.T@(s/rate-e),atol=1e-13);root=j*np.sqrt(e/rate)[:,None];np.testing.assert_allclose(info,root.T@root,atol=1e-10)
    assert check_grad(lambda z:likelihood_and_fisher(s,e,z,1.5)[0],lambda z:likelihood_and_fisher(s,e,z,1.5)[1],theta)<1e-3


def test_fixed_center_amplitude_profile_and_one_step_do_not_reduce_factor():
    s,e=fixture();null=fit_null(s,e,1.5);theta,a=initial_amplitudes(s,e,1.5,null,17.);assert a['success'];out=estimate(s,e,1.5,null);assert out['loglik']>=out['initial_loglik']-1e-10 and out['detection_statistic_unchanged'];assert abs(out['delay']-17.1)<.5


def test_location_estimator_does_not_mutate_observed_inputs_or_null_fit():
    s,e=fixture();null=fit_null(s,e,1.5);ss=s.copy();ee=e.copy();theta=null['theta'].copy();estimate(s,e,1.5,null);np.testing.assert_array_equal(s,ss);np.testing.assert_array_equal(e,ee);np.testing.assert_array_equal(null['theta'],theta)


def test_zero_secondary_amplitude_keeps_information_finite():
    s,e=fixture();ll,u,information=likelihood_and_fisher(s,e,np.array([.8,7.3,.3,0.,17.1]),1.5);assert np.isfinite(ll) and np.all(np.isfinite(u)) and np.all(np.isfinite(information));np.testing.assert_array_equal(information[4],np.zeros(5));np.testing.assert_array_equal(information[:,4],np.zeros(5))
