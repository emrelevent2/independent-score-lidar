import numpy as np
from flint import arb
from joint_plateau_envelope import calibration_flat,plateau_shared_cover
from conditional_calibration import calibration_upper,saturated_binomial
from envelope_evidence import exact_gaussian,certified_log_likelihood
from pilot import first_detection_probabilities


CF=dict(informative=True,signal_main_lower=.04,signal_main_upper=.06,background_lower=.00001,background_upper=.00003,
        forward_leakage_lower=0.,forward_leakage_upper=.000002)


def test_checked_plateau_is_saturated_at_every_eta_in_cell():
    assert calibration_flat(100,20000,.10,.12,CF)
    assert not calibration_flat(100,20000,0.,.25,CF)
    assert calibration_flat(0,0,0.,.25,CF)
    assert not calibration_flat(20,20,0.,.25,CF)
    assert calibration_flat(20,20,0.,.25,{'informative':False})
    saturated=float(saturated_binomial(100,20000))
    for eta in np.linspace(.10,.12,17):
        upper=calibration_upper(100,20000,float(eta),float(eta),CF)['upper']
        assert abs(upper-saturated)<1e-10


def test_plateau_refinement_retains_the_entire_physical_null():
    rng=np.random.default_rng(7521);eta=.11
    g=exact_gaussian(64,31.73,5.13,False);side=exact_gaussian(80,arb(31.73)+16,5.13,False)[:64]
    rates=[(arb(10)*(x+arb(eta)*y)+arb(5)/64)/1000 for x,y in zip(g,side)]
    h=rng.multinomial(1000,first_detection_probabilities([float(x) for x in rates]))[:-1]
    k,r=110,20000;calrate=arb(.05)*(arb(eta)+arb(.000001))+11*arb(.00002)
    true_ll=arb(certified_log_likelihood(h,rates,1000,'first',upper=True))+k*(-(-calrate).expm1()).log()-(r-k)*calrate
    bound=plateau_shared_cover(h,k,r,CF,initial_eta=8,initial_delay=4,max_boxes=64)
    assert arb(bound['upper'])>=true_ll and bound['calibration_plateau_eta_cells']>0
