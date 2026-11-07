import numpy as np
from scipy.special import expit
from clock_dual_transition import transition_kernel,full_log_likelihood,log_survival_after
from clock_dual_gradient import kernel_with_gradient,likelihood_and_gradient
from test_clock_dual_transition import observe


def test_full_likelihood_forward_gradient_matches_central_differences_with_censoring():
    design=np.array([[1.,.2,-.7],[.3,1.,.1],[-.4,.2,1.],[.9,-.4,.2]]);theta=np.array([-.8,.2,-.1]);p=expit(design@theta);dp=p[:,None]*(1-p[:,None])*design
    for dd,de in [(1,4),(2,3),(3,1),(0,3),(0,0)]:
        kernel=kernel_with_gradient(p,dp,dd,de)
        for pattern in [[0]*9,[1,0,0,0,0,0,0,0,0],[0,0,1,0,1,0,1,0,1],[1]*9]:
            record=np.asarray(observe(pattern,dd,de),int);value,gradient=likelihood_and_gradient(kernel,record,9);finite=[]
            for i in range(3):
                e=np.zeros(3);e[i]=1e-6;a=full_log_likelihood(transition_kernel(expit(design@(theta+e)),dd,de),record,9);b=full_log_likelihood(transition_kernel(expit(design@(theta-e)),dd,de),record,9);finite.append((a-b)/2e-6)
            np.testing.assert_allclose(gradient,finite,atol=2e-8,rtol=2e-7)


def test_derivative_of_total_transition_probability_is_zero():
    p=np.array([.04,.2,.6,.09]);dp=np.eye(4);kernel=kernel_with_gradient(p,dp,2,11);np.testing.assert_allclose(kernel['short_gradient'].sum(axis=1)+kernel['tail_gradient'],np.zeros((4,4)),atol=2e-14)


def test_long_terminal_survival_stays_finite_in_log_domain():
    p=np.array([.2,.7,.5]);kernel=kernel_with_gradient(p,np.eye(3),1,4);value,gradient=likelihood_and_gradient(kernel,np.array([0]),10000);assert np.isfinite(value) and np.all(np.isfinite(gradient));assert log_survival_after(kernel,0,9999)<-1000
