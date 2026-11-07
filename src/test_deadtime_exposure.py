import numpy as np
from scipy.integrate import quad
from deadtime_exposure import periodic_measure,sufficient_statistics,block_statistics,path_loglik,simulate_events


def test_fractional_recovery_and_last_censored_interval():
    s,e=sufficient_statistics([.5,3.25,6.],2,4,1.5)
    np.testing.assert_array_equal(s,[1,0,1,1])
    np.testing.assert_allclose(e,[.75,1.,1.,.75])
    # Durations are not Bernoulli trial counts: S<=E is not required.
    assert s[3]>e[3]


def test_previous_hit_and_large_frame_translation():
    s,e=sufficient_statistics([2.2],1,4,1.5,previous=-.25)
    np.testing.assert_array_equal(s,[0,0,1,0])
    np.testing.assert_allclose(e,[0,.75,.2,.3])
    for offset in [0,1000000000]:
        a,b,_=block_statistics(np.array([0,1])+offset,[3.75,2.2],4.,1.5,1+offset,1,4)
        np.testing.assert_array_equal(a,s);np.testing.assert_allclose(b,e,atol=1e-12)


def test_period_boundary_carry_preserves_timestamp():
    s,e,m=block_statistics([0,2],[4.01,1.],4.,.5,1,2,4)
    assert s.sum()==2 and s[0]==1 and s[1]==1
    assert m['out_of_period_timestamps_preserved']==1
    np.testing.assert_allclose(e.sum(),7.)


def test_interval_aggregation_matches_direct_phase_bin_intersections():
    rng=np.random.default_rng(492)
    lo=rng.uniform(-23,51,100);hi=lo+rng.uniform(0,22,100);k=7
    direct=np.zeros(k)
    for a,b in zip(lo,hi):
        for j in range(int(np.floor(a)),int(np.ceil(b))):
            direct[j%k]+=max(0,min(b,j+1)-max(a,j))
    np.testing.assert_allclose(periodic_measure(lo,hi,k),direct,atol=2e-12)


def test_dead_time_violation_rejected():
    import pytest
    with pytest.raises(ValueError,match='inside'):sufficient_statistics([1.,2.],3,4,1.5)


def test_likelihood_ratio_integrates_to_one_with_random_exposure():
    # Horizon 1, dead>=horizon: either no event, or first event t with E=t.
    p,q=2.3,.7
    no_event=np.exp(-p)*np.exp(path_loglik([0],[1],[q])-path_loglik([0],[1],[p]))
    event=quad(lambda t:p*np.exp(-p*t)*np.exp(path_loglik([1],[t],[q])-path_loglik([1],[t],[p])),0,1)[0]
    np.testing.assert_allclose(no_event+event,1.,atol=1e-12)


def test_simulation_homogeneous_renewal_rate_and_duration_mle():
    events=simulate_events(np.full(8,.2),20000,10.3,seed=719)
    s,e=sufficient_statistics(events,20000,8,10.3)
    assert np.all(np.diff(events)>=10.3-1e-9)
    assert abs(len(events)/160000-1/(10.3+5))<.001
    assert abs(s.sum()/e.sum()-.2)<.008
