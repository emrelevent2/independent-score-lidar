import numpy as np
from scipy.optimize import check_grad,minimize
from deadtime_scan import pulse,likelihood_parts,fit_null
from fast_two_return_fit import gaussian_bin,objective,fit_at_start,reports


def test_stable_kernel_matches_integrated_pulse_and_derivative_across_wrap():
    for k,sigma,t in [(32,1.5,31.8),(64,1.5,-.2),(32,2.1,10.37)]:
        expected=pulse(t,k,sigma);actual=np.array([gaussian_bin(j,t,k,sigma) for j in range(k)]).T
        np.testing.assert_allclose(actual,expected,atol=9e-16,rtol=1e-11);np.testing.assert_allclose(actual[0].sum(),1,atol=1e-14)


def test_joint_objective_matches_independent_numpy_likelihood_and_gradient():
    rng=np.random.default_rng(92013);k=32;sigma=1.5;x=np.array([.8,7.37,.3,.06,16.22]);e=rng.uniform(120,400,k);f,df=pulse(x[1],k,sigma);g,dg=pulse(x[4],k,sigma);rate=x[0]*f+x[2]/k+x[3]*g;s=rng.poisson(e*rate).astype(float);ll,u,_=likelihood_parts(s,e,rate,'duration');j=np.column_stack([f,x[0]*df,np.ones(k)/k,g,x[3]*dg]);v,gradient=objective(x,s,e,sigma)
    np.testing.assert_allclose(v,-ll/s.sum(),atol=1e-14);np.testing.assert_allclose(gradient,-j.T@u/s.sum(),atol=1e-13)
    assert check_grad(lambda z:objective(z,s,e,sigma)[0],lambda z:objective(z,s,e,sigma)[1],x)<2e-6


def test_two_return_fit_improves_null_and_matches_reference_optimizer():
    k=32;sigma=1.5;e=np.full(k,2000.);rates=.8*pulse(7.2,k,sigma)[0]+.12*pulse(16.3,k,sigma)[0]+.3/k;s=np.rint(e*rates);fit=fit_null(s,e,sigma);result=fit_at_start(s,e,sigma,fit,16.)
    assert result['loglik']>fit['loglik']+10 and abs(result['delay']-16.3)<.2
    def reference(x):
        a,t,b,c,d=x;f,df=pulse(t,k,sigma);g,dg=pulse(d,k,sigma);ll,u,_=likelihood_parts(s,e,a*f+b/k+c*g,'duration');j=np.column_stack([f,a*df,np.ones(k)/k,g,c*dg]);return -ll/s.sum(),-j.T@u/s.sum()
    a,t,b=fit['theta'];old=minimize(reference,[a,t,b,.02,16.],jac=True,method='L-BFGS-B',bounds=[(0,20),(t-4.5,t+4.5),(1e-6,30),(0,20),(11.5,20.5)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=160))
    np.testing.assert_allclose(result['loglik'],-old.fun*s.sum(),atol=1e-6)


def test_unsupported_context_preserves_local_joint_decision_statistic_and_location():
    k=32;e=np.full(k,1000.);s=np.rint(e*(.8*pulse(7.2,k,1.5)[0]+.04*pulse(16.3,k,1.5)[0]+.3/k));fit=fit_null(s,e,1.5);r=reports(s,e,1.5,fit,dict(primary=7.2,secondary=16.3,two_pulse_loglr=20))
    assert r['samples']['local_joint']==r['samples']['context_max']==r['samples']['context_transport']


def test_supported_context_statistic_matches_explicit_weighted_branches():
    k=32;e=np.full(k,1000.);s=np.rint(e*(.8*pulse(8.2,k,1.5)[0]+.04*pulse(17.3,k,1.5)[0]+.3/k));fit=fit_null(s,e,1.5);r=reports(s,e,1.5,fit,dict(primary=7.2,secondary=16.3,two_pulse_loglr=30));ll=[max(z['loglik'] for z in r['fits'][:-2]),r['fits'][-2]['loglik'],r['fits'][-1]['loglik']];anchor=max(ll);expected=2*(anchor+np.log(sum(w*np.exp(z-anchor) for w,z in zip([.5,.25,.25],ll)))-fit['loglik'])
    np.testing.assert_allclose(r['samples']['context_transport']['score'],expected,atol=2e-12)
    assert r['samples']['context_max']['score']>=r['samples']['context_transport']['score']-2e-12
