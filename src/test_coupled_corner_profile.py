import numpy as np
from flint import arb
from coupled_corner_profile import corner_profile,calibration_tangent,gaussian_mass_lower,amplitude_cap,amplitude_tail_upper
from conditional_calibration import calibration_upper,saturated_binomial
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float
from pilot import first_detection_probabilities
from coupled_corner_envelope import coupled_cover

CF=dict(informative=True,signal_main_lower=.04,signal_main_upper=.06,background_lower=.00001,background_upper=.00003,forward_leakage_lower=0.,forward_leakage_upper=.000002)


def test_profiled_calibration_tangents_majorize_all_eta():
    for center in [.01,.10,.22]:
        value,gradient=calibration_tangent(center,100,20000,CF)
        for eta in np.linspace(0,.25,101):
            upper=outward_float(value+gradient*(arb(float(eta))-arb(center)))
            assert upper+1e-10>=calibration_upper(100,20000,float(eta),float(eta),CF)['upper']


def test_four_corner_tangent_covers_coupled_physical_points():
    rng=np.random.default_rng(203491)
    def rates(a,eta,b,d,w):
        g=exact_gaussian(64,d,w,False);side=exact_gaussian(80,arb(d)+16,w,False)[:64]
        return [(arb(a)*(x+arb(eta)*y)+arb(b)/64)/1000 for x,y in zip(g,side)]
    h=rng.multinomial(1000,first_detection_probabilities([float(x) for x in rates(20.,.1,5.,31.4,5.2)]))[:-1]
    result=corner_profile(h,100,20000,CF,10.,30.,.04,.14,31.3,31.5,5.1,5.3)
    assert result['tangent_certified']
    for _ in range(25):
        a,eta,b,d,w=rng.uniform(10,30),rng.uniform(.04,.14),rng.uniform(0,100),rng.uniform(31.3,31.5),rng.uniform(5.1,5.3)
        ll=arb(certified_log_likelihood(h,rates(a,eta,b,d,w),1000,'first',upper=True))
        rate=arb(.05)*(arb(eta)+arb(.000001))+11*arb(.00002)
        ll+=100*(-(-rate).expm1()).log()-(20000-100)*rate
        assert arb(result['upper'])>=ll


def test_amplitude_tail_keeps_arbitrary_large_signal_domain():
    rng=np.random.default_rng(99845);h=rng.multinomial(1000,np.r_[np.full(64,.003),1-64*.003])[:-1]
    cap=amplitude_cap(h,1000,100,20000,-3000.)
    assert cap is not None and cap['tail_upper']<=-3000
    G=gaussian_mass_lower(64)
    for d in [0.,.49,12.3,31.5,62.7,63.]:
        for w in [4.5,5.5]:
            g=exact_gaussian(64,d,w,False);assert sum(g,arb(0))>=G
            side=exact_gaussian(80,arb(d)+16,w,False)[:64]
            for a in [cap['cap'],2*cap['cap'],10*cap['cap']]:
                rates=[(arb(a)*(x+arb(.25)*y)+arb(30)/64)/1000 for x,y in zip(g,side)]
                ll=arb(certified_log_likelihood(h,rates,1000,'first',upper=True))+saturated_binomial(100,20000)
                assert arb(amplitude_tail_upper(h,1000,cap['cap'],100,20000))>=ll
    assert amplitude_cap(np.array([1000,0]),1000,0,1,-10.) is None


def test_stopped_complete_cover_contains_interior_and_unbounded_tail():
    rng=np.random.default_rng(32294)
    g=exact_gaussian(64,31.4,5.2,False);s=exact_gaussian(80,arb(31.4)+16,5.2,False)[:64]
    rates=[(arb(20)*(x+arb(.1)*y)+arb(5)/64)/1000 for x,y in zip(g,s)]
    h=rng.multinomial(1000,first_detection_probabilities([float(x) for x in rates]))[:-1]
    bound=coupled_cover(h,100,20000,CF,initial_amplitude=2,initial_eta=4,initial_delay=4,max_boxes=64)
    cap=bound['amplitude_cap']['cap']
    for a in [0.,20.,cap*.9,cap,cap*4]:
        for d in [0.,31.4,63.]:
            eta=.1;g=exact_gaussian(64,d,5.2,False);s=exact_gaussian(80,arb(d)+16,5.2,False)[:64]
            rates=[(arb(a)*(x+arb(eta)*y)+arb(5)/64)/1000 for x,y in zip(g,s)]
            ll=arb(certified_log_likelihood(h,rates,1000,'first',upper=True));cr=arb(.05)*(arb(eta)+arb(.000001))+11*arb(.00002)
            ll+=100*(-(-cr).expm1()).log()-(20000-100)*cr
            assert arb(bound['upper'])>=ll
