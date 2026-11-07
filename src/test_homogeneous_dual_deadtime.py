import numpy as np
from scipy.integrate import quad
from homogeneous_dual_deadtime import log_gap_density,gap_survival,log_active_probability


def test_detector_dominant_or_zero_detector_reduces_to_shifted_exponential():
    for td,te in [(100.,40.),(40.,40.),(0.,80.)]:
        s=np.array([0.,39.,40.,79.,80.,100.,130.,200.]);dead=max(td,te);expected=np.where(s>=dead,.05*np.exp(-.05*(s-dead)),0);np.testing.assert_allclose(np.exp(log_gap_density(s,.05,td,te)),expected,atol=1e-15);np.testing.assert_allclose(gap_survival(s,.05,td,te),np.exp(-.05*np.maximum(s-dead,0)),atol=1e-15)


def test_one_hidden_avalanche_matches_the_known_explicit_transition_formula():
    td=10.;te=15.;rate=.07;s=np.linspace(te,70,101);width=np.maximum(np.minimum(te,s-td)-td,0);expected=rate*np.exp(-rate*(s-td))+rate**2*np.exp(-rate*(s-2*td))*width
    np.testing.assert_allclose(np.exp(log_gap_density(s,rate,td,te)),expected,rtol=2e-14,atol=1e-15);np.testing.assert_allclose(np.exp(log_active_probability(rate,td,te)),np.exp(-rate*(te-td)),atol=1e-15)


def test_density_normalizes_with_multiple_hidden_renewals_and_exact_tail():
    for td,te,rate in [(31.7,80.,.05),(10.,35.,.2),(7.,50.,.3),(83.7,100.,.005),(100.,20.,.05),(0.,80.,.1)]:
        end=te+td;breaks=sorted(set([te,end]+([n*td for n in range(1,int(end/td)+1) if te<n*td<end] if td else [])));mass=sum(quad(lambda t:np.exp(log_gap_density(t,rate,td,te)),a,b,epsabs=1e-11,epsrel=1e-11)[0] for a,b in zip(breaks[:-1],breaks[1:]));mass+=np.exp(log_gap_density(end,rate,td,te))/rate;np.testing.assert_allclose(mass,1.,atol=2e-11)


def test_survival_matches_integrated_density_and_safe_tail_hazard():
    rate=.07;td=10.;te=35.
    for t in [35.,37.,41.,44.,45.,50.,100.]:
        integral=quad(lambda x:np.exp(log_gap_density(x,rate,td,te)),te,t,points=[40.] if t>40 else None,epsabs=1e-11)[0];np.testing.assert_allclose(gap_survival(t,rate,td,te),1-integral,atol=3e-11)
    s=np.array([te+td,te+td+10,te+td+50]);np.testing.assert_allclose(np.exp(log_gap_density(s,rate,td,te))/gap_survival(s,rate,td,te),rate,rtol=1e-13)
