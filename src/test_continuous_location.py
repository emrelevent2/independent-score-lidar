from fractions import Fraction
import numpy as np
from continuous_location import interval_mass,window_location,two_value_gradient
from shoulder_alternative import pulse
from pilot import first_detection_probabilities


def test_uniform_cells_have_no_five_atom_window_bonus():
    # Width4 covers four unit cells,not five grid atoms at both endpoints.
    assert interval_mass(3.,np.arange(7.),np.ones(7)/7,0.,0.)[0]==4/7


def test_window_optimizer_matches_dense_grid_and_keeps_ties_stable():
    # Identical likelihoods leave these original prior weights unchanged.
    models=dict(rates=np.full((6,8),.001),weights=[Fraction(1,10)]*2+[Fraction(2,10)]*4,
        spatial=np.array([True,True,False,False,False,False]),locations=np.array([3.2,3.2,1.,2.,3.,4.]),global_delays=np.arange(1.,5.))
    r=window_location(np.zeros(8,int),models,3.2)
    dense=np.linspace(0,7,10001);mass=interval_mass(dense,np.arange(1.,5.),np.full(4,.2),3.2,.2)
    assert r['working_window_mass']>=mass.max()-1e-12
    assert r['working_window_mass']>=r['old_working_window_mass']-1e-12


def test_two_pulse_first_event_gradient_and_rate_model():
    z=np.array([80.,12.,6.,20.3,40.2,5.1]);eta=.08;t=np.arange(64)
    rates=(z[0]*pulse(t,z[3],z[5],eta)+z[1]*pulse(t,z[4],z[5],eta)+z[2]/64)/1000
    h=np.random.default_rng(84911).multinomial(1000,first_detection_probabilities(rates))[:-1]
    value,grad=two_value_gradient(z,h,eta)
    expected=np.sum(h*np.log(-np.expm1(-rates))-(1000-np.cumsum(h))*rates)
    assert abs(value-expected)<1e-10
    for j in range(6):
        step=1e-5*max(1.,abs(z[j]));a=z.copy();b=z.copy();a[j]+=step;b[j]-=step
        difference=(two_value_gradient(a,h,eta)[0]-two_value_gradient(b,h,eta)[0])/(2*step)
        np.testing.assert_allclose(grad[j],difference,rtol=2e-5,atol=1e-6)
