import itertools
import numpy as np
from clock_dual_transition import transition_kernel,log_transition_mass,survival_after,full_log_likelihood
from homogeneous_dual_deadtime import gap_survival


def observe(pattern,dd,de):
    available=electronics=0;out=[]
    for t,photon in enumerate(pattern):
        if photon and t>=available:
            available=t+dd+1
            if t>=electronics:out.append(t);electronics=t+de+1
    return tuple(out)


def test_full_clock_likelihood_matches_exhaustive_latent_path_summation():
    p=np.array([.12,.37,.55]);horizon=9
    for dd,de in [(1,4),(2,3),(3,1),(0,3),(0,0)]:
        expected={}
        for pattern in itertools.product([0,1],repeat=horizon):
            probability=float(np.prod([p[t%3] if x else 1-p[t%3] for t,x in enumerate(pattern)]));record=observe(pattern,dd,de);expected[record]=expected.get(record,0)+probability
        table=transition_kernel(p,dd,de)
        for record,probability in expected.items():np.testing.assert_allclose(np.exp(full_log_likelihood(table,np.asarray(record,dtype=int),horizon)),probability,atol=2e-14,rtol=1e-12)
        np.testing.assert_allclose(sum(expected.values()),1,atol=1e-14)


def test_short_mass_and_remaining_live_tail_normalize_every_phase():
    for dd,de in [(1,4),(2,11),(7,50),(10,10),(0,10)]:
        p=np.array([.01,.2,.85,.08]);table=transition_kernel(p,dd,de);np.testing.assert_allclose(table['short'].sum(axis=1)+table['live_at_tail'],np.ones(4),atol=2e-14)
        for phase in range(4):
            length=3000;t=np.arange(1,length+1);mass=np.exp(log_transition_mass(table,phase,t));np.testing.assert_allclose(mass.sum()+survival_after(table,phase,length),1,atol=4e-13)


def test_detector_dominant_kernel_matches_direct_observed_activation_factors():
    p=np.array([.04,.32,.12]);dd=4;de=1;pattern=np.array([1,1,0,0,0,1,0,0,1,0,0,1,1]);record=np.array(observe(pattern,dd,de));table=transition_kernel(p,dd,de);available=0;ll=0.
    for t in range(len(pattern)):
        if t<available:continue
        if t in record:ll+=np.log(p[t%3]);available=t+dd+1
        else:ll+=np.log1p(-p[t%3])
    np.testing.assert_allclose(full_log_likelihood(table,record,len(pattern)),ll,atol=2e-14)


def test_homogeneous_clock_law_converges_to_continuous_renewal_kernel():
    rate=.07;td=10.;te=35.;times=np.arange(36.,61.);reference=gap_survival(times,rate,td,te);errors=[]
    for step in [1.,.25,.0625]:
        table=transition_kernel(np.array([-np.expm1(-rate*step)]),round(td/step)-1,round(te/step)-1);clock=survival_after(table,0,np.rint(times/step).astype(int)-1);errors.append(float(np.max(np.abs(clock-reference))))
    assert errors[-1]<.01 and errors[-1]<errors[0]/5


def test_full_clock_factor_with_initial_and_terminal_terms_converges_to_continuous_law():
    from safe_deadtime_exposure import simulate_two_dead_times
    from homogeneous_dual_deadtime import log_gap_density
    rate=.05;td=31.7;te=80.;horizon=2000;observed,_=simulate_two_dead_times(np.array([rate]),horizon,td,te,317007);continuous=np.log(rate)-rate*observed[0]+np.sum(log_gap_density(np.diff(observed),rate,td,te))+np.log(gap_survival(horizon-observed[-1],rate,td,te));values=[]
    for refinement in [4,10,40]:
        table=transition_kernel(np.array([-np.expm1(-rate/refinement)]),round(td*refinement)-1,round(te*refinement)-1);ticks=np.floor(observed*refinement).astype(int);values.append(full_log_likelihood(table,ticks,horizon*refinement)+len(ticks)*np.log(refinement))
    assert abs(values[-1]-continuous)<.05
