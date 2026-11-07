import numpy as np
import ct_dual_reverse as reverse
import ct_dual_fast as forward
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times


def test_reverse_matches_forward_all_rate_pieces_in_four_original_regimes():
    for td,te,seed in [(83.7,40.,1001),(126.7,80.,1002),(31.7,80.,1003),(83.7,100.,1004)]:
        rate,j=rates_and_design([2.4,15.3,.8,.15,25.3],64,1.5);obs,_=simulate_two_dead_times(rate,25,td,te,seed);g=reverse.prepare_record(obs,1600,64,td,te)
        a=reverse.full_factor(rate,np.eye(64),g,return_pieces=True);b=forward.full_factor(rate,np.eye(64),g,return_pieces=True)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=3e-8,rtol=3e-10)
        np.testing.assert_allclose(reverse.full_factor(rate,j,g)[1],a[1]@j,atol=1e-9)


def test_reverse_two_hidden_five_parameter_gradient_against_quadrature_and_finite_difference():
    theta=np.array([2.4,15.3,.8,.15,25.3]);rate,j=rates_and_design(theta,64,1.5);obs,_=simulate_two_dead_times(rate,40,31.7,80.,218591);g=reverse.prepare_record(obs,2560,64,31.7,80.);a,ga=reverse.full_factor(rate,j,g);b,gb=reference.full_factor(rate,j,g,32);np.testing.assert_allclose(a,b,atol=2e-8);np.testing.assert_allclose(ga,gb,atol=2e-7)
    finite=[]
    for h in range(5):
        d=np.zeros(5);d[h]=1e-5;rp,jp=rates_and_design(theta+d,64,1.5);rm,jm=rates_and_design(theta-d,64,1.5);finite.append((reverse.full_factor(rp,jp,g)[0]-reverse.full_factor(rm,jm,g)[0])/2e-5)
    np.testing.assert_allclose(ga,finite,atol=2e-6,rtol=2e-6)


def test_reverse_initial_empty_and_terminal_short_survivals():
    rate=np.array([.03,.12,.08,.19]);j=np.eye(4)
    for observed,total in [([],12.),([.7],1.),([.7],4.91),([.7],7.1),([.7,6.01],7.)]:
        g=reverse.prepare_record(observed,total,4,1.7,4.1);a=reverse.full_factor(rate,j,g,return_pieces=True);b=forward.full_factor(rate,j,g,return_pieces=True)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-10)


def test_reverse_phase_integral_adjoints_match_explicit_basis_integrals():
    rng=np.random.default_rng(3928417);k=7
    for _ in range(40):
        lower=rng.uniform(-100,100);upper=lower+rng.uniform(0,200);d=forward.descriptor(lower,upper,k);direct=np.zeros(k);difference=np.zeros(k);uniform=reverse._add_integral(np.array(d),1.3,direct,difference);actual=direct+uniform+np.cumsum(difference);expected=np.array([1.3*forward._iv(np.eye(k)[h],np.r_[0.,np.cumsum(np.eye(k)[h])],np.array(d)) for h in range(k)]);np.testing.assert_allclose(actual,expected,atol=2e-13)


def test_reverse_at_hidden_path_boundaries_and_zero_dimensional_design():
    rate=np.array([.03,.12,.08,.19,.05,.07,.2]);j=np.eye(7)
    for td,te in [(2.1,6.3),(3.17,8.)]:
        for gap in np.r_[max(td,te),3*td+np.array([-1e-9,0.,1e-9]),td+te+np.array([-1e-9,0.,1e-9])]:
            if gap<max(td,te):continue
            # Literal first timestamp0 preserves exact support-boundary gaps;
            # adding/subtracting a nonbinary offset could round below Te.
            obs=np.array([0.,gap]);g=reverse.prepare_record(obs,obs[-1]+.79,7,td,te);a=reverse.full_factor(rate,j,g,return_pieces=True);b=forward.full_factor(rate,j,g,return_pieces=True)
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-10,rtol=1e-10)
            v,z=reverse.full_factor(rate,np.zeros((7,0)),g);assert z.shape==(0,) and abs(v-a[0])<1e-12
