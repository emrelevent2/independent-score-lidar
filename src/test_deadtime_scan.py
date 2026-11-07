import numpy as np
from scipy.optimize._numdiff import approx_derivative
from deadtime_exposure import sufficient_statistics,simulate_events
from deadtime_scan import pulse,corr,likelihood_parts,rounded_opportunities,fit_null,scan,multistart_glrt


def test_pulse_derivative_and_fft_all_integer_shifts():
    k=32;sigma=1.5;f,d=pulse(7.3,k,sigma)
    np.testing.assert_allclose(f.sum(),1,atol=1e-14)
    np.testing.assert_allclose(d,approx_derivative(lambda x:pulse(x[0],k,sigma)[0],[7.3]).ravel(),atol=2e-9)
    rng=np.random.default_rng(77);u=rng.normal(size=k);f0,_=pulse(0,k,sigma)
    np.testing.assert_allclose(corr(u,f0),[np.dot(u,pulse(t,k,sigma)[0]) for t in range(k)],atol=1e-14)


def test_path_score_derivative_both_models():
    s=np.array([1.,3.,0.]);e=np.array([8.,9.,5.]);lam=np.array([.1,.5,.2])
    for m in ['duration','ded']:
        _,u,_=likelihood_parts(s,e,lam,m)
        num=approx_derivative(lambda x:likelihood_parts(s,e,x,m)[0],lam).ravel()
        np.testing.assert_allclose(u,num,rtol=2e-7,atol=2e-7)


def test_rounded_opportunity_counter_has_valid_marginal_trials():
    events=np.array([.5,3.25,6.]);s,e=sufficient_statistics(events,2,4,1.5)
    a=rounded_opportunities(events,2,4,1.5)
    np.testing.assert_array_equal(a,[2,1,2,2]);assert np.all(a>=s) and np.all(a>=e)


def test_nuisance_projected_information_matches_direct_schur_complement():
    k=64;sig=1.5;f,_=pulse(15.4,k,sig);events=simulate_events(.8*f+.3/k,1000,83.7,seed=942)
    s,e=sufficient_statistics(events,1000,k,83.7);fit=fit_null(s,e,sig);r=scan(s,e,sig,fit=fit)
    a,t,b=fit['theta'];f,d=pulse(t,k,sig);lam=a*f+b/k;j=np.column_stack([f,a*d,np.ones(k)/k]);_,u,w=likelihood_parts(s,e,lam,'duration')
    g,_=pulse(r['delay'],k,sig);cross=j.T@(w*g);inv=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12)
    ue=g@u-cross@inv@(j.T@u);ie=np.dot(w,g*g)-cross@inv@cross
    np.testing.assert_allclose(r['score'],max(ue,0)**2/ie,rtol=1e-10,atol=1e-10)
    gfit=multistart_glrt(s,e,sig,r);assert gfit['score']>=0
